export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/sparsedrive_small_stage2_modified.py \
	--result-path work_dirs/sparsedrive_small_stage2/results.pkl