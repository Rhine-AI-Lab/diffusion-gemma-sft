python -m torch.distributed.run --nproc_per_node=8 eval_diffgemma.py \
  --dataset sudoku \
  --batch_size 8 \
  --gen_length 128 \
  --max_denoising_steps 64 \
  --model_path /data/oss_bucket_0/models/google/diffusiongemma-26B-A4B-it \
  --output_dir /data/oss_bucket_0/eval_results/sudoku