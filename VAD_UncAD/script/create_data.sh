export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/data_converter/vad_nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/ \
    --out-dir ./data_processed/ \
    --extra-tag nuscenes \
    --version v1.0

