python src/test/vgg_test.py \
	--input-parquet ./data/vgg/test.parquet \
	--model-path ./model/Qwen2.5-VL-3B-Instruct \
	--hallu-model-path ./model/Qwen3-8B \
	--output-root ./results/vgg2 \
	"$@"
