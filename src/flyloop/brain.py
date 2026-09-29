"""Batched LIF model of the whole central brain (Shiu et al. 2024, FlyWire v783) on the GPU.

B independent flies share the static wiring; synapses selected as plastic carry per-fly weights.
Constants are transcribed from Shiu's `model.py`. The update order matches Brian2's schedule at
dt = 0.1 ms: state update (exact, 'linear') -> threshold -> synaptic delivery (delay 1.8 ms) and
Poisson input -> reset. As in Brian2, v and g ("unless refractory") are conditional-write: any
input arriving while the target is refractory, or on the step it spikes, is dropped. Delivery is event-driven: spiking neurons are appended to a ring of spike
lists and their out-rows are scattered 18 steps later. All kernels are CUDA-graph capturable.
"""

import numpy as np
import warp as wp

from flyloop import data

# Shiu model.py default_params (mV, ms, Hz)
V0, V_RST, V_TH = -52.0, -52.0, -45.0
T_MBR, TAU, T_RFC, T_DLY = 20.0, 5.0, 2.2, 1.8
W_SYN, R_POI, F_POI = 0.275, 150.0, 250.0
W_POI = W_SYN * F_POI  # a Poisson event adds 68.75 mV to v: one input event -> one spike
DT = 0.1  # ms (Brian2 default)
DELAY = int(round(T_DLY / DT))  # 18 steps
RING = DELAY + 1  # spike lists for steps n-18 .. n
RFC_STEPS = int(T_RFC / DT + 1e-3)  # Brian2 timestep(): 22
LANES = 32
G_FLUSH = 1e-20  # mV; |g| below this is set to 0 so resting neurons take the early exit


@wp.kernel
def _neurons(
    v: wp.array2d(dtype=wp.float32),  # type: ignore
    g: wp.array2d(dtype=wp.float32),  # type: ignore
    last: wp.array2d(dtype=wp.int32),  # type: ignore
    rfc: wp.array(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    e1: float,
    e2: float,
    c12: float,
    bias: wp.array(dtype=wp.float32),  # type: ignore  constant input (mV), 0 in Shiu's model
    ring: wp.array2d(dtype=wp.int32),  # type: ignore
    ring_n: wp.array(dtype=wp.int32),  # type: ignore
    counts: wp.array2d(dtype=wp.int32),  # type: ignore
):
    b, i = wp.tid()
    vi = v[b, i]
    gi = g[b, i]
    if gi == 0.0 and vi == V0 and bias[i] == 0.0:
        return
    n = counter[0]
    if n - last[b, i] < rfc[i]:  # refractory: v and g frozen ("unless refractory")
        return
    vi = V0 + (vi - V0) * e1 + gi * c12 + bias[i] * (1.0 - e1)
    gi = gi * e2
    if wp.abs(gi) < G_FLUSH:
        gi = 0.0
    if vi > V_TH:
        v[b, i] = V_RST
        g[b, i] = 0.0
        last[b, i] = n
        counts[b, i] = counts[b, i] + 1
        s = n % RING
        k = wp.atomic_add(ring_n, s, 1)
        if k < ring.shape[1]:
            ring[s, k] = b * v.shape[1] + i
        return
    v[b, i] = vi
    g[b, i] = gi


@wp.kernel
def _poisson(
    v: wp.array2d(dtype=wp.float32),  # type: ignore
    last: wp.array2d(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    inp: wp.array(dtype=wp.int32),  # type: ignore
    shut: wp.array(dtype=wp.int32),  # type: ignore  steps after a spike with writes dropped
    prob: wp.array2d(dtype=wp.float32),  # type: ignore  (fly, input) rate * dt
    rng: wp.array2d(dtype=wp.uint32),  # type: ignore
):
    b, m = wp.tid()
    p = prob[b, m]
    if p <= 0.0:
        return
    st = rng[b, m]
    r = wp.randf(st)
    rng[b, m] = st
    i = inp[m]
    if r < p and counter[0] - last[b, i] >= shut[m]:  # shut is per input (m), not per neuron
        v[b, i] = v[b, i] + W_POI


@wp.kernel
def _deliver(
    g: wp.array2d(dtype=wp.float32),  # type: ignore
    last: wp.array2d(dtype=wp.int32),  # type: ignore
    counter: wp.array(dtype=wp.int32),  # type: ignore
    ring: wp.array2d(dtype=wp.int32),  # type: ignore
    ring_n: wp.array(dtype=wp.int32),  # type: ignore
    shut: wp.array(dtype=wp.int32),  # type: ignore
    rowptr: wp.array(dtype=wp.int32),  # type: ignore
    col: wp.array(dtype=wp.int32),  # type: ignore
    w: wp.array(dtype=wp.float32),  # type: ignore
    prow: wp.array(dtype=wp.int32),  # type: ignore
    pcol: wp.array(dtype=wp.int32),  # type: ignore
    wpl: wp.array2d(dtype=wp.float32),  # type: ignore  (fly, plastic synapse)
):
    k, lane = wp.tid()
    n = counter[0]
    if n < DELAY:
        return
    s = (n - DELAY) % RING
    if k >= wp.min(ring_n[s], ring.shape[1]):
        return
    code = ring[s, k]
    nn = g.shape[1]
    b = code // nn
    i = code - b * nn
    # g += w, dropped for targets that are refractory or spiked this step
    for e in range(rowptr[i] + lane, rowptr[i + 1], LANES):
        j = col[e]
        if n - last[b, j] >= shut[j]:
            wp.atomic_add(g, b, j, w[e])
    for e in range(prow[i] + lane, prow[i + 1], LANES):
        j = pcol[e]
        if n - last[b, j] >= shut[j]:
            wp.atomic_add(g, b, j, wpl[b, e])


@wp.kernel
def _tick(
    counter: wp.array(dtype=wp.int32),  # type: ignore
    ring_n: wp.array(dtype=wp.int32),  # type: ignore
    max_n: wp.array(dtype=wp.int32),  # type: ignore
):
    n = counter[0]
    max_n[0] = wp.max(max_n[0], ring_n[n % RING])
    counter[0] = n + 1
    ring_n[(n + 1) % RING] = 0  # its spikes (step n-18) were delivered this step


def _csr(pre, post, w, n):
    order = np.argsort(pre, kind="stable")
    rowptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=rowptr[1:])
    return rowptr.astype(np.int32), post[order].astype(np.int32), w[order].astype(np.float32), order


class Brain:
    """n_flies copies of the brain. `inputs`: indices of neurons that can receive Poisson drive
    (their refractory period is 0, as Shiu's `poi()` sets). `plastic`: boolean mask over
    `data.edges()` of synapses with per-fly weights (`wpl`, initialised to the static weight)."""

    def __init__(self, n_flies, inputs, plastic=None, cap=None, graph_steps=50, seed=0,
                 wiring=None, bias=None):
        wp.synchronize()  # a physics sim's in-flight GPU work (e.g. warmup) corrupts a brain built during it
        pre, post, cnt, n = wiring or (*data.edges(), len(data.neurons()))  # wiring: for tests
        self.n = n
        self.b, self.graph_steps = n_flies, graph_steps
        w = cnt * W_SYN
        plastic = np.zeros(len(pre), bool) if plastic is None else plastic
        rowptr, col, ws, _ = _csr(pre[~plastic], post[~plastic], w[~plastic], n)
        prow, pcol, pw, porder = _csr(pre[plastic], post[plastic], w[plastic], n)
        self.plastic_edges = np.flatnonzero(plastic)[porder]  # edge index of each plastic slot
        self.w0 = pw  # static weight of each plastic slot
        i32, f32 = (lambda a: wp.array(a, dtype=wp.int32)), (lambda a: wp.array(a, dtype=wp.float32))
        self._rowptr, self._col, self._w = i32(rowptr), i32(col), f32(ws)
        self._prow, self._pcol = i32(prow), i32(np.concatenate([pcol, [0]]))
        self.wpl = f32(np.tile(np.concatenate([pw, [0]]), (n_flies, 1)))

        self.inputs = np.asarray(inputs, np.int64)
        rfc = np.full(n, RFC_STEPS, np.int32)
        rfc[self.inputs] = 0
        self._rfc, self._inp = i32(rfc), i32(self.inputs)
        shut = np.maximum(rfc, 1)  # the spike step itself is not writable either
        self._shut, self._shut_in = i32(shut), i32(shut[self.inputs])
        self._bias = f32(np.zeros(n) if bias is None else bias)  # mV; not part of Shiu's model
        self.prob = wp.zeros((n_flies, len(self.inputs)), dtype=wp.float32)
        self._seed = seed

        e1, e2 = np.exp(-DT / T_MBR), np.exp(-DT / TAU)
        self._e = (float(e1), float(e2), float(TAU / (T_MBR - TAU) * (e1 - e2)))
        self.cap = cap or 512 * n_flies
        self.v = wp.zeros((n_flies, n), dtype=wp.float32)
        self.g = wp.zeros((n_flies, n), dtype=wp.float32)
        self.last = wp.zeros((n_flies, n), dtype=wp.int32)
        self.counts = wp.zeros((n_flies, n), dtype=wp.int32)
        self.counter = wp.zeros(1, dtype=wp.int32)
        self.ring = wp.zeros((RING, self.cap), dtype=wp.int32)
        self.ring_n = wp.zeros(RING, dtype=wp.int32)
        self.max_n = wp.zeros(1, dtype=wp.int32)
        self.rng = wp.zeros((n_flies, len(self.inputs)), dtype=wp.uint32)
        self.reset()
        with wp.ScopedCapture() as cap:
            for _ in range(graph_steps):
                self._step()
        self._graph = cap.graph

    def reset(self, seed=None):
        """Rest state (v = v0, g = 0, not refractory); spike counts and step counter zeroed.
        Plastic weights are left alone (see `reset_weights`)."""
        seed = self._seed if seed is None else seed
        self.v.fill_(V0)
        self.g.zero_()
        self.last.fill_(-(10**9))
        for a in (self.counts, self.counter, self.ring_n, self.max_n, self.prob):
            a.zero_()
        ss = np.random.SeedSequence(seed).generate_state(self.rng.size, np.uint32)
        wp.copy(self.rng, wp.array(ss.reshape(self.rng.shape), dtype=wp.uint32))

    def reset_weights(self):
        wp.copy(self.wpl, wp.array(np.tile(np.concatenate([self.w0, [0]]), (self.b, 1)),
                                   dtype=wp.float32))

    def set_rates(self, rates_hz):
        """Poisson rates (n_flies, n_inputs) in Hz for the `inputs` neurons."""
        p = np.asarray(rates_hz, np.float32) * (DT * 1e-3)
        wp.copy(self.prob, wp.array(np.broadcast_to(p, self.prob.shape).copy(), dtype=wp.float32))

    def _step(self):
        b, n = self.b, self.n
        e1, e2, c12 = self._e
        wp.launch(_neurons, dim=(b, n), inputs=[self.v, self.g, self.last, self._rfc, self.counter,
                                               e1, e2, c12, self._bias, self.ring, self.ring_n,
                                               self.counts])
        wp.launch(_poisson, dim=self.prob.shape, inputs=[
            self.v, self.last, self.counter, self._inp, self._shut_in, self.prob, self.rng])
        wp.launch(_deliver, dim=(self.cap, LANES), inputs=[
            self.g, self.last, self.counter, self.ring, self.ring_n, self._shut, self._rowptr, self._col,
            self._w, self._prow, self._pcol, self.wpl])
        wp.launch(_tick, dim=1, inputs=[self.counter, self.ring_n, self.max_n])

    def run(self, n_steps):
        """Advance n_steps (a multiple of graph_steps) brain steps of 0.1 ms."""
        assert n_steps % self.graph_steps == 0
        for _ in range(n_steps // self.graph_steps):
            wp.capture_launch(self._graph)

    def check_overflow(self):
        m = int(self.max_n.numpy()[0])
        assert m <= self.cap, f"spike list overflow: {m} spikes in one step > cap {self.cap}"
        return m
