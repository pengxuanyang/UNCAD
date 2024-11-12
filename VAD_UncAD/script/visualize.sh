export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/analysis_tools/visualization.py \
	--result-path test/.../results_nusc.pkl \
	--save-path vis_result
	