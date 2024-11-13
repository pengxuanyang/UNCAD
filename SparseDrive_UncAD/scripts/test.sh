bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2_modified.py \
    ckpt/sparsedrive_stage2_UncAD.pth \
    8 \
    --deterministic \
    --eval bbox 
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl