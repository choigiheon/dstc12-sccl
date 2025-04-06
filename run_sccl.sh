device="mps"
dataset_file="./dstc12-data/AppenBanking/all.jsonl"
result_file="./appen_banking_predicted.jsonl"
model_name="sentence-transformers/all-mpnet-base-v2"
pre_train_epoch=0
joint_train_epoch=5
n_clusters=16
batchsize=40
lr=5e-4


python3 sccl/cluster.py \
    --device $device \
    --model-name $model_name \
    --dropout 0.1 \
    --dataset-file $dataset_file \
    --result-file $result_file \
    --max-length 100 \
    --batch-size $batchsize \
    --lr $lr \
    --pre-train-epoch $pre_train_epoch \
    --joint-train-epoch $joint_train_epoch \
    --lr-scale 10000 \
    --augtype simcse \
    --temperature 0.5 \
    --eta 100 \
    --n-clusters $n_clusters \
    --alpha 1.0 \
    --n-init 100 \
    --kmeans-interval 1 \
    --print-freq 1 \
    --eval-interval 1 \