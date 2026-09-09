python -m torch.distributed.run --nproc_per_node=8 eval_diffgemma.py \
  --dataset gsm8k \
  --batch_size 4 \
  --gen_length 256 \
  --max_denoising_steps 128 \
  --model_path /data/oss_bucket_0/models/google/diffusiongemma-26B-A4B-it \
  --output_dir eval_results/gsm8k