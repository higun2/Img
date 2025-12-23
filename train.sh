torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 256 --blr 5e-5 \
--epochs 200 --warmup_epochs 5 \
--gen_bsz 256 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir /datasets/v-geonhuison/JiT/results/test_5-1 \
--data_path /datasets/v-geonhuison/imagenet --online_eval
