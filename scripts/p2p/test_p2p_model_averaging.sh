#!/usr/bin/env bash

# Modify flags --model_averaging_device=[gpu|cpu] and --request_mode=[sync|async]
# Training on synthetic data
train() {
    BATCH=$1
    echo "[BEGIN TRAINING KEY] training-p2p"
    kungfu-prun  -np 4 -H 127.0.0.1:4 -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 \
        --num_epochs=1 \
        --eval=False \
        --forward_only=False \
        --print_training_accuracy=True \
        --num_gpus=1 \
        --num_warmup_batches=20 \
        --batch_size=${BATCH} \
        --momentum=0.9 \
        --weight_decay=0.0001 \
        --staged_vars=False \
        --optimizer=p2p_averaging \
        --variable_update=kungfu \
        --kungfu_strategy=none \
        --model_averaging_device=gpu \
        --request_mode=async \
        --peer_selection_strategy=roundrobin \
        --use_datasets=True \
        --distortions=False \
        --fuse_decode_and_crop=True \
        --resize_method=bilinear \
        --display_every=100 \
        --checkpoint_every_n_epochs=False \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/data/kungfu/checkpoints-lbr/checkpoint \
        --data_format=NCHW \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
    echo "[END TRAINING KEY] training-p2p"
}

validate() {
    for worker in 0 1 2 3
    do
    echo "[BEGIN VALIDATION KEY] validation-lbr-${RUN}-worker-${worker}"
    python3 tf_cnn_benchmarks.py --eval=True --forward_only=False --model=resnet32 --data_name=cifar10 \
        --data_dir=/data/cifar-10/cifar-10-batches-py \
        --variable_update=replicated --data_format=NCHW --use_datasets=False --num_batches=50 --eval_batch_size=50 \
        --num_gpus=4 --use_tf_layers=True \
        --checkpoint_directory=/data/kungfu/checkpoints-lbr/checkpoint-worker-${worker}/v-000001 --checkpoint_interval=0.25 \
        --checkpoint_every_n_epochs=True 
    echo "[END VALIDATION KEY] validation-lbr-${RUN}-worker-${worker}"
    done
}


train 64
validate