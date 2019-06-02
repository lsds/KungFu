#!/usr/bin/env bash

# P2P Model Averaging flags --model_averaging_device=[gpu|cpu] and --request_mode=[sync|async]
# Training on synthetic data
train() {
    DEVICE=$1
    MODE=$2
    kungfu-prun  -np 4 -H 127.0.0.1:4 -timeout 1000000s \
        python3 tf_cnn_benchmarks.py --model=resnet32 --data_name=cifar10 \
        --num_batches=50 \
        --eval=False \
        --forward_only=False \
        --print_training_accuracy=True \
        --num_gpus=1 \
        --num_warmup_batches=20 \
        --batch_size=64 \
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
        --display_every=50 \
        --checkpoint_every_n_epochs=False \
        --checkpoint_interval=0.25 \
        --checkpoint_directory=/data/kungfu/checkpoints-lbr/checkpoint \
        --data_format=NCHW \
        --batchnorm_persistent=True \
        --use_tf_layers=True \
        --winograd_nonfused=True
}

train cpu sync
train gpu sync
train cpu async
train gpu async