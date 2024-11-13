# stage1
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage1_modified.py \
   8 \
   --deterministic 

# stage2
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_small_stage2_modified.py \
   8 \
   --deterministic 