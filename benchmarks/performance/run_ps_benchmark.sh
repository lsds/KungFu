#!/usr/bin/env bash

killall python

PS_HOSTS="localhost:2220,localhost:2221,localhost:2222,localhost:2223"
WORKER_HOSTS="localhost:2224,localhost:2225,localhost:2226,localhost:2227"

# ps0
echo "launch ps 0"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &

# ps1
echo "launch ps 1"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=1 &

# ps2
echo "launch ps 2"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=2 &

# ps3
echo "launch ps 3"
CUDA_VISIBLE_DEVICES=-1 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=3 &

# worker0
echo "launch worker 0"
CUDA_VISIBLE_DEVICES=0 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 &

# worker1
echo "launch worker 1"
CUDA_VISIBLE_DEVICES=1 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 &

# worker2
echo "launch worker 2"
CUDA_VISIBLE_DEVICES=2 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=2 &

# worker3
echo "launch worker 3"
CUDA_VISIBLE_DEVICES=3 python ps_benchmark.py  --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=3 &

wait