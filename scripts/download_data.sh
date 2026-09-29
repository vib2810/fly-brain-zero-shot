#!/usr/bin/env bash
# Downloads the three inputs this repo does not ship (pinned to the commits the results used):
#   data/shiu/     Shiu et al. 2024 connectivity of the FlyWire v783 brain (~100 MB)
#   data/flywire/  FlyWire cell-type annotations (Schlegel et al. 2024)
#   data/flyvis/   FlyVis pretrained optic-lobe models (Lappalainen et al. 2024)
set -euo pipefail
cd "$(dirname "$0")/.."

SHIU=https://github.com/philshiu/Drosophila_brain_model/raw/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
ANNOT=https://github.com/flyconnectome/flywire_annotations/raw/8587524c1748ce5ef2080822a2fc890fc03bf597

mkdir -p data/shiu data/flywire data/flyvis
curl -fL -o data/shiu/Connectivity_783.parquet "$SHIU/Connectivity_783.parquet"
curl -fL -o data/shiu/Completeness_783.csv "$SHIU/Completeness_783.csv"
curl -fL -o data/flywire/Supplemental_file1_neuron_annotations.tsv \
  "$ANNOT/supplemental_files/Supplemental_file1_neuron_annotations.tsv"

md5sum -c <<'SUMS'
3d802fd542b5d18570ba1ba0bb0abed9  data/shiu/Connectivity_783.parquet
031a4b6eac490bda17c75b65ccaa7906  data/shiu/Completeness_783.csv
719904abad876c68ace1b5690c9b9b63  data/flywire/Supplemental_file1_neuron_annotations.tsv
SUMS

FLYVIS_ROOT_DIR="$PWD/data/flyvis" flyvis download-pretrained --skip_large_files
echo "done: data/ is ready"
