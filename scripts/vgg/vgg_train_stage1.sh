FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train \
    ./src/LlamaFactory/examples/train_full/vgg.yaml \
	dataset_dir=./src/LlamaFactory/data