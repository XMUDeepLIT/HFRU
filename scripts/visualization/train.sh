
export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch \
  --multi_gpu \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port 29507 \
  --num_processes 4 \
  --mixed_precision bf16 \
  ./src/IP-Adapter/train.py \
  --qwen_model_path ./model/Qwen2.5-VL-3B-Instruct \
  --clip_model_path ./src/IP-Adapter/models/image_encoder \
  --data_file ./data/dataset/VGGFace2/train_image_paths.json \
  --image_root ./data/dataset/VGGFace2/train \
  --output_dir ./mapper \
  --train_batch_size 256 \
  --num_train_epochs 10 \
  --save_steps 500 \
  --logging_steps 10 \
  --dataloader_num_workers 16 \
  --gradient_accumulation_steps 1 \
  --wandb_run_name "qwen25vl"
