"""Tripod CPG (the hand-built leg rhythm) + PreprogrammedSteps, batched over worlds as Warp
kernels and driven by per-world parameters (amplitudes, phases, posture, turn asymmetry). CPG state and spline evaluation are float64 so the outputs match the numpy
CPGController (flygym_demo) to float32 rounding."""

import numpy as np
import warp as wp

from flygym.anatomy import LEGS
from flygym.compose import ActuatorType
from flygym_demo.complex_terrain import PreprogrammedSteps
from flygym_demo.complex_terrain.common import dof_spec_to_jointdof
from flygym_demo.complex_terrain.cpg_controller import get_cpg_biases

FREQ, COUPLING, CONVERGENCE = 12.0, 10.0, 20.0  # make_tripod_cpg_network defaults
POSTURE_DOFS = {"coxa_roll": ("thorax", "coxa", "roll"),
                "femur_pitch": ("coxa", "trochanterfemur", "pitch")}
N_LEGS = 6
PHASE0 = N_LEGS  # params: amp[0:6], phase[6:12], posture[12:12+2*n_intact], asym[-1]

vec6d = wp.types.vector(length=6, dtype=wp.float64)


def param_names(fly):
    names = [f"amp_{l}" for l in LEGS] + [f"phase_{l}" for l in LEGS]
    names += [f"post_{l}_{p}" for l in fly.get_legs_order() for p in POSTURE_DOFS]
    return names + ["asym"]


def param_groups(names):
    return {g: [i for i, n in enumerate(names) if n.startswith(g[:4])]
            for g in ["amp", "phase", "posture", "asym"]}


def initial_phases(seeds):
    """CPGNetwork's initial phases for each seed (first draw of RandomState(seed))."""
    return np.stack([np.random.RandomState(s).random(N_LEGS) * 2 * np.pi for s in seeds])


@wp.kernel
def _cpg_step(
    phases: wp.array2d(dtype=wp.float64),  # type: ignore
    mags: wp.array2d(dtype=wp.float64),  # type: ignore
    params: wp.array2d(dtype=wp.float64),  # type: ignore
    weights: wp.array2d(dtype=wp.float64),  # type: ignore
    biases: wp.array2d(dtype=wp.float64),  # type: ignore
    side: wp.array(dtype=wp.float64),  # type: ignore
    asym_idx: int,
    asym_bias: wp.float64,
    dt: wp.float64,
    omega: wp.float64,
    alpha: wp.float64,
):
    w = wp.tid()
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    asym = params[w, asym_idx] + asym_bias
    dth = vec6d()
    dr = vec6d()
    for i in range(6):
        s = zero
        for j in range(6):
            s += mags[w, j] * weights[i, j] * wp.sin(phases[w, j] - phases[w, i] - biases[i, j])
        dth[i] = omega + s
        amp = wp.max((one + params[w, i]) * (one + side[i] * asym), zero)
        dr[i] = alpha * (amp - mags[w, i])
    for i in range(6):
        phases[w, i] = phases[w, i] + dth[i] * dt
        mags[w, i] = mags[w, i] + dr[i] * dt


@wp.func
def _wrap(ph: wp.float64):
    two_pi = wp.float64(2.0 * wp.pi)
    return ph - wp.floor(ph / two_pi) * two_pi


@wp.kernel
def _targets(
    phases: wp.array2d(dtype=wp.float64),  # type: ignore
    mags: wp.array2d(dtype=wp.float64),  # type: ignore
    params: wp.array2d(dtype=wp.float64),  # type: ignore
    coef: wp.array4d(dtype=wp.float64),  # type: ignore  (leg, interval, dof_k, power)
    knots: wp.array(dtype=wp.float64),  # type: ignore
    neutral: wp.array2d(dtype=wp.float64),  # type: ignore
    dof_leg: wp.array(dtype=wp.int32),  # type: ignore
    dof_k: wp.array(dtype=wp.int32),  # type: ignore
    dof_post: wp.array(dtype=wp.int32),  # type: ignore
    sigma: wp.float64,
    seeds: wp.array(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    out: wp.array2d(dtype=wp.float32),  # type: ignore
):
    w, j = wp.tid()
    leg = dof_leg[j]
    k = dof_k[j]
    t = _wrap(phases[w, leg] + params[w, PHASE0 + leg])
    n_int = knots.shape[0] - 1
    seg = wp.min(wp.int32(t / knots[1]), n_int - 1)
    dx = t - knots[seg]
    psi = ((coef[leg, seg, k, 0] * dx + coef[leg, seg, k, 1]) * dx + coef[leg, seg, k, 2]) * dx
    psi = psi + coef[leg, seg, k, 3]
    val = neutral[leg, k] + mags[w, leg] * (psi - neutral[leg, k])
    if dof_post[j] >= 0:
        val = val + params[w, dof_post[j]]
    if sigma > wp.float64(0.0):
        state = wp.rand_init(seeds[w], counter[0] * out.shape[1] + j)
        val = val + sigma * wp.float64(wp.randn(state))
    out[w, j] = wp.float32(val)


@wp.kernel
def _adhesion(
    phases: wp.array2d(dtype=wp.float64),  # type: ignore
    params: wp.array2d(dtype=wp.float64),  # type: ignore
    swing_end: wp.array(dtype=wp.float64),  # type: ignore
    adh_legs: wp.array(dtype=wp.int32),  # type: ignore
    out: wp.array2d(dtype=wp.float32),  # type: ignore
):
    w, a = wp.tid()
    leg = adh_legs[a]
    t = _wrap(phases[w, leg] + params[w, PHASE0 + leg])
    out[w, a] = wp.where(t > wp.float64(0.0) and t < swing_end[leg], 0.0, 1.0)


class BatchedCPG:
    """Per-world CPG state + parameters on the GPU. `launch()` advances the CPG one
    step and writes joint targets (`angles`) and adhesion (`adhesion`); it is
    graph-capturable."""

    def __init__(self, fly, n_worlds, timestep, noise_sigma=0.0, asym_bias=0.0):
        steps = PreprogrammedSteps()
        self.n_worlds, self.timestep = n_worlds, timestep
        self.noise_sigma, self.asym_bias = noise_sigma, asym_bias
        self.names = param_names(fly)
        dofs = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        spec_index = {
            dof_spec_to_jointdof(leg, spec): (li, k)
            for li, leg in enumerate(LEGS)
            for k, spec in enumerate(steps.dofs_per_leg)
        }
        dof_leg, dof_k = zip(*(spec_index[d] for d in dofs))
        post_index = {
            dof_spec_to_jointdof(l, POSTURE_DOFS[p]): self.names.index(f"post_{l}_{p}")
            for l in fly.get_legs_order()
            for p in POSTURE_DOFS
        }
        dof_post = [post_index.get(d, -1) for d in dofs]

        coef = np.stack([steps._psi_funcs[l].c.transpose(1, 2, 0) for l in LEGS])
        biases = get_cpg_biases("tripod")
        side = [1.0 if l[0] == "l" else -1.0 for l in LEGS]  # +asym: bigger left steps
        f64 = lambda a: wp.array(np.asarray(a, np.float64), dtype=wp.float64)  # noqa: E731
        i32 = lambda a: wp.array(np.asarray(a, np.int32), dtype=wp.int32)  # noqa: E731
        self._coef, self._knots = f64(coef), f64(steps._psi_funcs["lf"].x)
        self._neutral = f64([steps.neutral_pos[l].ravel() for l in LEGS])
        self._weights, self._biases = f64((biases > 0) * COUPLING), f64(biases)
        self._side = f64(side)
        self._swing_end = f64([steps.swing_period[l][1] for l in LEGS])
        self._dof_leg, self._dof_k, self._dof_post = i32(dof_leg), i32(dof_k), i32(dof_post)
        self._adh_legs = i32([LEGS.index(l) for l in fly.get_legs_order()])

        self.phases = wp.zeros((n_worlds, N_LEGS), dtype=wp.float64)
        self.mags = wp.zeros((n_worlds, N_LEGS), dtype=wp.float64)
        self.params = wp.zeros((n_worlds, len(self.names)), dtype=wp.float64)
        self.seeds = wp.zeros(n_worlds, dtype=wp.int32)
        self.counter = wp.zeros(1, dtype=wp.int32)
        self.angles = wp.zeros((n_worlds, len(dofs)), dtype=wp.float32)
        self.adhesion = wp.zeros((n_worlds, len(self._adh_legs)), dtype=wp.float32)

    def reset(self, params, seeds):
        """Copy per-world parameters (n_worlds, n_params) and seeds into the fixed buffers."""
        wp.copy(self.params, wp.array(np.asarray(params, np.float64), dtype=wp.float64))
        wp.copy(self.phases, wp.array(initial_phases(seeds), dtype=wp.float64))
        self.mags.zero_()
        wp.copy(self.seeds, wp.array(np.asarray(seeds, np.int32), dtype=wp.int32))
        self.counter.zero_()

    def launch(self):
        n = self.n_worlds
        wp.launch(_cpg_step, dim=n, inputs=[
            self.phases, self.mags, self.params, self._weights, self._biases, self._side,
            len(self.names) - 1, wp.float64(self.asym_bias), wp.float64(self.timestep),
            wp.float64(2 * np.pi * FREQ), wp.float64(CONVERGENCE)])
        wp.launch(_targets, dim=self.angles.shape, inputs=[
            self.phases, self.mags, self.params, self._coef, self._knots, self._neutral,
            self._dof_leg, self._dof_k, self._dof_post, wp.float64(self.noise_sigma),
            self.seeds, self.counter], outputs=[self.angles])
        wp.launch(_adhesion, dim=self.adhesion.shape, inputs=[
            self.phases, self.params, self._swing_end, self._adh_legs], outputs=[self.adhesion])
