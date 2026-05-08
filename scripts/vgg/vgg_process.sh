python src/preprocess/vgg.py \
	--model-path "./model/Qwen2.5-VL-3B-Instruct" \
	--input-parquet "./data/dataset/vgg/vgg_select.parquet" \
	--output-dir "./src/LlamaFactory/data" \
	--parquet-output-dir "./data/vgg" \
	"$@"
