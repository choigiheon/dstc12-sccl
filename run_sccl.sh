device="cuda"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
result_file="./appen_banking_predicted.jsonl"
preference_file="./dstc12-data/AppenBanking/preference_pairs.json"
model_name="BAAI/bge-base-en-v1.5"
pre_train_epoch=10
inter_train_epoch=100
joint_train_epoch=50
n_clusters=15
batchsize=40
lr=5e-7


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