#!/usr/bin/env bash

killall python

# ps0
echo "launch ps 0"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=0 &

# ps1
echo "launch ps 1"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py  --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=ps --task_index=1 &

# worker0
echo "launch worker 0"
CUDA_VISIBLE_DEVICES=0 python ps_benchmark.py  --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=0 &

# worker1
echo "launch worker 1"
CUDA_VISIBLE_DEVICES=1 python ps_benchmark.py  --ps_hosts=localhost:2222,localhost:2223 --worker_hosts=localhost:2224,localhost:2225 --job_name=worker --task_index=1