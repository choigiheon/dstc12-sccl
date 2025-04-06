device="mps"
dataset_file="./dstc12-data/AppenBanking/all_sampled.jsonl"
llm_name="Qwen/Qwen2.5-1.5B"
n_clusters=5

python scripts/run_theme_detection.py \
    --dataset-file $dataset_file \
    --llm-name $llm_name \
    --n-clusters $n_clusters \
    --cluster-label-map "./cluster_label_map.json" \
    --device $device