# ISE train script for Market1501
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u examples/ise_train_usl.py \
-b 256 \
-a vit_small \
-d market1501 \
--iters 200 \
--momentum 0.2 \
--eps 0.6 \
--self-norm \
--hw-ratio 2 \
--conv-stem \
--num-instances 8 \
--use-hard \
--logs-dir ../logs/ISE_Market1501/vit_small_ics_cfs_lup \
--data-dir ../data/ \
--step-size 30 \
--epochs 70 \
--save-step 20 \
--eval-step 1 \
--sample-type hard \
--use_support \
--lp_loss_weight 0.1 \
--pretrained_path ../model/vit_small_ics_cfs_lup.pth \





