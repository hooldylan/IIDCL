# ISE train script for MSMT17
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u examples/ise_train_usl.py \
-b 256 \
-a vit_small \
-d msmt17 \
--iters 200 \
--momentum 0.2 \
--eps 0.7 \
--self-norm \
--hw-ratio 2 \
--conv-stem \
--num-instances 8 \
--use-hard \
--logs-dir ../logs/ISE_Msmt17/vit_small_ics_cfs_lup \
--data-dir ../data/ \
--step-size 20 \
--epochs 50 \
--save-step 20 \
--eval-step 1 \
--sample-type ori \
--use_support \
--lp_loss_weight 0.01 \
--pretrained_path ../model/vit_small_ics_cfs_lup.pth \

