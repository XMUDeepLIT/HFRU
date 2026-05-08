CUDA_VISIBLE_DEVICES=0 \
python ./src/IP-Adapter/test.py \
    --projector_path ./model/qwen_to_clip_projector.pt \
    --qwen_model_path ./model/Qwen2.5-VL-3B-Instruct \
    --base_model_path ./model/stable-diffusion-xl-base-1.0 \
    --image_encoder_path ./src/IP-Adapter/models/image_encoder \
    --ip_ckpt ./src/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin \
    --image_dir ./data/test_visualization \
    --output_path ./results/visualization/ \
    --qwen_prompt ""