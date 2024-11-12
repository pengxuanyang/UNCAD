CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash ./tools/dist_test.sh \
    projects/configs/VAD/VAD_base_e2e_modified.py \
    ckpt/VAD_UncAD.pth \
    8 \
    --tmpdir tmp \
    --eval bbox \
