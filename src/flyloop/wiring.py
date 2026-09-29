"""Null model: the connectome with its connections scrambled."""

import numpy as np

from flyloop import data


def shuffled_wiring(seed=0):
    """Degree-preserving shuffle: every connection keeps its presynaptic neuron, sign and synapse
    count; the postsynaptic targets are permuted across all connections."""
    pre, post, cnt = data.edges()
    return pre, np.random.default_rng(seed).permutation(post), cnt, len(data.neurons())
