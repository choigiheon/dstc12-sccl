device="mps"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
llm_name="Qwen/Qwen2.5-1.5B"
n_clusters=29

python scripts/run_theme_detection.py \
    --dataset-file $dataset_file \
    --llm-name $llm_name \
    --n-clusters $n_clusters \
    --cluster-label-map "./dstc12-data/AppenBanking/topclus/cluster_label_map.json" \
    --device $device