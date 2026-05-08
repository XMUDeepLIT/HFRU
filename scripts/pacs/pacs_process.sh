python src/preprocess/pacs.py \
	--model-path "./model/Qwen2.5-VL-3B-Instruct" \
	--model-path2 "./model/Qwen3-8B" \
	--input-parquet "./data/dataset/pacs/data/train-00000-of-00001.parquet" \
	--output-dir "./src/LlamaFactory/data" \
	--pacs-verl-dir "./data/pacs" \
	"$@"
