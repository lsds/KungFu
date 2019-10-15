# Distributed Training Benchmark
Used Tensorflow version is 1.13.1

## Parameter Servers
Use the following shell script to run the parameter server benchmark.
```bash
# Configure two local parameter servers
PS_HOSTS="localhost:2220"

# Configure four training workers
WORKER_HOSTS="localhost:2230,localhost:2231,localhost:2232,localhost:2233"

# Start parameter server on CPU
CUDA_VISIBLE_DEVICES="" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &

# Start workers on different GPUs
CUDA_VISIBLE_DEVICES="0" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 &
CUDA_VISIBLE_DEVICES="1" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 &
CUDA_VISIBLE_DEVICES="2" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=2 &
CUDA_VISIBLE_DEVICES="3" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=3 &
```

## Horovod
### Install
There are three steps to install Horovod.
1. install OpenMPI with the [script](https://raw.githubusercontent.com/tensorlayer/openpose-plus/master/scripts/install-mpi.sh)
2. install g++ version 4.8 with `bash sudo apt install g++-4.8`
3. install Horovod with `bash pip3 install horovod`

Use the following command to run the Horovod benchmark.
```bash
horovodrun -np 4 python3 benchmark_horovod.py
```
Used Horovod version is 0.18.1

## KungFu
Assuming that the KungFu repository is located at ~/KungFu use the following command to run the KungFu benchmark.
```bash
~/KungFu/bin/kungfu-run -np 4 -timeout 1h python3 benchmark_kungfu.py
```