device="cuda"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
llm_name="mistralai/Mistral-7B-Instruct-v0.3"
n_clusters=29

python scripts/run_theme_detection.py \
    --dataset-file $dataset_file \
    --llm-name $llm_name \
    --n-clusters $n_clusters \
    --cluster-label-map "./cluster_label_map.json" \
    --device $device