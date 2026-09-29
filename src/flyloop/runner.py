"""Batched episodes on GPUSimulation driven by BatchedCPG through one captured CUDA graph.
Each episode restores a post-settle physics snapshot instead of calling `sim.reset()`,
which would reallocate the MJWarp data and invalidate the graph."""

import mujoco as mj
import mujoco_warp as mjw
import numpy as np
import warp as wp

from flygym.anatomy import LEGS, BodySegment
from flygym.compose import ActuatorType
from flygym.warp import GPUSimulation
from flygym_demo.complex_terrain import PreprogrammedSteps

from flyloop.cpg import BatchedCPG
from flyloop.records import _record, _record_qpos
from flyloop.body import make_world


@wp.kernel
def _increment(counter: wp.array(dtype=wp.int32)):  # type: ignore
    counter[0] = counter[0] + 1


@wp.kernel
def _record_contacts(
    sensordata: wp.array2d(dtype=wp.float32),  # type: ignore
    adr: wp.array(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    every: int,
    out: wp.array4d(dtype=wp.float32),  # type: ignore  (world, record, leg, [found, |F|])
):
    w, leg = wp.tid()
    s = counter[0]
    if s % every == 0:
        a = adr[leg]
        f = wp.vec3(sensordata[w, a + 1], sensordata[w, a + 2], sensordata[w, a + 3])
        out[w, s // every, leg, 0] = sensordata[w, a]
        out[w, s // every, leg, 1] = wp.length(f)


class Runner:
    def __init__(self, cut_legs, n_worlds, max_steps, noise_sigma=0.02, asym_bias=0.0,
                 record_every=10, record_qpos=False, record_contacts=False):
        self.fly, world, _ = make_world(cut_legs=cut_legs, add_camera=False)
        self.n_worlds, self.every = n_worlds, record_every
        sim = self.sim = GPUSimulation(world, n_worlds)
        self.timestep = sim.timestep
        self.cpg = BatchedCPG(self.fly, n_worlds, sim.timestep, noise_sigma, asym_bias)
        self.param_names = self.cpg.names

        # Settle once in the default pose with adhesion on (as tutorial 4a), then snapshot.
        sim.reset()
        dofs = self.fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        pose = PreprogrammedSteps().default_pose_by_dof_order(dofs).astype(np.float32)
        sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, np.tile(pose, (n_worlds, 1)))
        sim.set_leg_adhesion_states(
            self.fly.name, np.ones((n_worlds, len(self.fly.get_legs_order())), np.float32))
        sim.warmup()
        d = sim.mjw_data
        self._state = [a for a in (d.qpos, d.qvel, d.act, d.qacc_warmstart, d.time, d.ctrl)
                       if a.size > 0]
        self._snapshot = [wp.clone(a) for a in self._state]

        n_rec = max_steps // record_every + 1
        self.max_steps = max_steps
        self.pos = wp.zeros((n_worlds, n_rec), dtype=wp.vec3f)
        self.quat = wp.zeros((n_worlds, n_rec), dtype=wp.quatf)
        body = int(sim._internal_bodyids_by_fly[self.fly.name][
            self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))])
        counter = self.cpg.counter
        rec = [(_record, n_worlds, [d.xpos, d.xquat, body, counter, record_every],
                [self.pos, self.quat])]
        self.qpos = self.contacts = None
        if record_qpos:
            self.qpos = wp.zeros((n_worlds, n_rec, sim.mj_model.nq), dtype=wp.float32)
            rec.append((_record_qpos, (n_worlds, sim.mj_model.nq),
                        [d.qpos, counter, record_every], [self.qpos]))
        if record_contacts:  # all 6 legs incl. stumps (sensor subtree = coxa)
            sensors = world.legpos_to_groundcontactsensors_by_fly[self.fly.name]
            adr = [sim.mj_model.sensor_adr[mj.mj_name2id(
                sim.mj_model, mj.mjtObj.mjOBJ_SENSOR, sensors[leg].name)] for leg in LEGS]
            self.contacts = wp.zeros((n_worlds, n_rec, len(LEGS), 2), dtype=wp.float32)
            rec.append((_record_contacts, (n_worlds, len(LEGS)),
                        [d.sensordata, wp.array(adr, dtype=wp.int32), counter, record_every],
                        [self.contacts]))
        self._rec = rec
        self._setup_extra()
        with wp.ScopedCapture() as cap:
            self._graph_body()
        self._graph = cap.graph

    def _setup_extra(self):
        """Hook for subclasses: allocate extra per-step buffers before graph capture."""

    def _graph_body(self):
        """One control + physics step (captured once, replayed every step)."""
        self._control_and_step()
        self._launch_records()

    def _control_and_step(self):
        self.cpg.launch()
        self.sim.set_actuator_inputs(self.fly.name, ActuatorType.POSITION, self.cpg.angles)
        self.sim.set_leg_adhesion_states(self.fly.name, self.cpg.adhesion)
        self.sim.step()
        wp.launch(_increment, dim=1, outputs=[self.cpg.counter])

    def _launch_records(self):
        for kernel, dim, inputs, outputs in self._rec:
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs)

    def run(self, params, seeds, n_steps):
        """One episode from the settled snapshot. params (n_worlds, n_params); seeds set
        initial CPG phases and motor noise. Returns records every `record_every` steps."""
        assert n_steps <= self.max_steps
        self._start_episode(params, seeds)
        for _ in range(n_steps):
            wp.capture_launch(self._graph)
        return self._collect(n_steps)

    def _start_episode(self, params, seeds):
        for dst, src in zip(self._state, self._snapshot):
            wp.copy(dst, src)
        # Recompute derived fields (xpos, ...) for the restored state; forward() may
        # overwrite the warm start, so restore that again.
        mjw.forward(self.sim.mjw_model, self.sim.mjw_data)
        wp.copy(self.sim.mjw_data.qacc_warmstart, self._snapshot[self._state.index(
            self.sim.mjw_data.qacc_warmstart)])
        self.cpg.reset(params, seeds)
        self._launch_records()  # record index 0 = start state

    def _collect(self, n_steps):
        n = n_steps // self.every + 1
        out = {"pos": self.pos.numpy()[:, :n], "quat": self.quat.numpy()[:, :n]}
        if self.qpos is not None:
            out["qpos"] = self.qpos.numpy()[:, :n]
        if self.contacts is not None:
            out["contacts"] = self.contacts.numpy()[:, :n]
        return out
