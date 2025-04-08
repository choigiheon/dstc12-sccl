device="mps"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
result_file="./appen_banking_predicted.jsonl"
preference_file="./dstc12-data/AppenBanking/preference_pairs.json"
model_name="sentence-transformers/all-MiniLM-L6-v2"
pre_train_epoch=10
joint_train_epoch=10
n_clusters=16
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
    --joint-train-epoch $joint_train_epoch \
    --temperature 0.5 \
    --n-clusters $n_clusters \
    --alpha 1.0 \
    --n-init 100 \
    --print-freq 1 \
    --eval-interval 1 \
    --preference-file $preference_file