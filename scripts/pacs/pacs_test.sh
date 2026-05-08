python src/test/pacs_test.py \
	--input_parquets \
		./data/pacs/val.parquet \
		./data/pacs/test.parquet \
	--model_path ./model/Qwen2.5-VL-3B-Instruct \
	--hallu_model_path ./model/Qwen3-8B \
	--output_root ./results/pacs \
	"$@"
