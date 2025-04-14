device="mps"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
result_file="./appen_banking_predicted.jsonl"
model_name="sentence-transformers/all-mpnet-base-v2"
pre_train_epoch=3
joint_train_epoch=100
n_clusters=15
batchsize=32
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
    --lr-scale 100 \
    --eta 10 \
    --pre-train-epoch $pre_train_epoch \
    --joint-train-epoch $joint_train_epoch \
    --temperature 0.5 \
    --n-clusters $n_clusters \
    --alpha 1.0 \
    --n-init 100 \
    --print-freq 1 \
    --eval-freq 100 \