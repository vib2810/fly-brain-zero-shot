"""FlyWire v783 wiring (Shiu et al. 2024 files) and FlyWire cell-type annotations.

Neuron index = row of Shiu's `Completeness_783.csv` (Brian2 index). Sides are FlyWire's `side`
annotation: soma side for brain neurons, nerve-entry side for sensory neurons.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
COMP = DATA / "shiu" / "Completeness_783.csv"
CONN = DATA / "shiu" / "Connectivity_783.parquet"
ANNOT = DATA / "flywire" / "Supplemental_file1_neuron_annotations.tsv"

# Shiu et al. labellar sugar GRNs (their example.ipynb), used to validate the brain: 150 Hz -> MN9 ~70 Hz
SUGAR_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_TYPE = "CB0701"  # Shiu's MN9 720575940660219265 is CB0701 in v783


@lru_cache
def neurons():
    """Neuron table in Brian2 index order: root_id plus the FlyWire annotation columns."""
    ids = pd.read_csv(COMP, index_col=0).index.to_numpy()
    a = pd.read_csv(ANNOT, sep="\t", low_memory=False).drop_duplicates("root_id")
    cols = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"]
    return pd.DataFrame({"root_id": ids}).merge(a[cols], on="root_id", how="left")


@lru_cache
def edges():
    """(pre, post, signed synapse count) as int arrays, one row per Shiu connection."""
    c = pd.read_parquet(CONN, columns=["Presynaptic_Index", "Postsynaptic_Index",
                                       "Excitatory x Connectivity"])
    return (c["Presynaptic_Index"].to_numpy(np.int64), c["Postsynaptic_Index"].to_numpy(np.int64),
            c["Excitatory x Connectivity"].to_numpy(np.int64))


def neuron_sets():
    """Named index arrays used here: Shiu's sugar GRNs."""
    return {"sugar": np.flatnonzero(neurons().root_id.isin(SUGAR_IDS).to_numpy())}
