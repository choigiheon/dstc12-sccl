device="cuda"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
result_file="./appen_banking_predicted.jsonl"
preference_file="./dstc12-data/AppenBanking/preference_pairs.json"
model_name="sentence-transformers/all-mpnet-base-v2"
pre_train_epoch=0
inter_train_epoch=0
joint_train_epoch=5
n_clusters=29
batchsize=25
lr=5e-6


python3 sccl/cluster.py \
    --device $device \
    --model-name $model_name \
    --dropout 0.1 \
    --dataset-file $dataset_file \
    --result-file $result_file \
    --max-length 100 \
    --batch-size $batchsize \
    --lr $lr \
    --lr-scale 10 \
    --pre-train-epoch $pre_train_epoch \
    --inter-train-epoch $inter_train_epoch \
    --joint-train-epoch $joint_train_epoch \
    --temperature 0.5 \
    --n-clusters $n_clusters \
    --alpha 1.0 \
    --n-init 100 \
    --print-freq 10 \
    --eval-interval 100 \
    --preference-file $preference_file \
    --negative-alpha 5.0