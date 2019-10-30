# Distributed Training Benchmark

Distributed training benchmark of KungFu, Horovod and Parameter Servers.

## Intro

This benchmark requires TensorFlow <=1.13.2, KungFu and Horovod.
We have run this benchmark on two clusters: one has two DGX-1 machines (each has 8 V100) and one has 16 P100 machines. You can see the benchmark result [here](result/).

In the following, we provide sample commands to run the benchmark.
We assume the benchmark runs on a server with 4 GPUs.
The benchmark imports models from [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications). You can freely choose different models
and batch sizes.

## Synchronous Training

In the synchronous training benchmark, we compare KungFu with Horovod.

### KungFu

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark_kungfu.py --kf-optimizer=sync-sgd --model=ResNet50 --batch-size=64
```

### Horovod

Horovod requires OpenMPI ([install script](https://raw.githubusercontent.com/tensorlayer/openpose-plus/master/scripts/install-mpi.sh)).
We use Horovod 0.16.1 (`pip3 install horovod==0.16.1`). Use the following command to run the Horovod benchmark.

```bash
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 benchmark_horovod.py --model=ResNet50 --batch-size=64
```

## Asynchronous Training

Asynchronous training is useful for addressing network bottlenecks, stragglers and unpredictable availability of resource (e.g., AWS spot instances). In the asynchronous benchmark, we compare KungFu with Parameter Servers.

### KungFu

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark_kungfu.py --kf-optimizer=async-sgd --model=ResNet50 --batch-size=64
```

### Parameter Servers

Use the following shell script to run the parameter server benchmark.

```bash
# Configure 1 local parameter server (We suggest users to have a 1:1 ratio between parameter servers and workers)
PS_HOSTS="localhost:2220"

# Configure four training workers
WORKER_HOSTS="localhost:2230,localhost:2231,localhost:2232,localhost:2233"

# Start parameter server on CPU
CUDA_VISIBLE_DEVICES="" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &

# Start workers on different GPUs
CUDA_VISIBLE_DEVICES="0" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 --model=ResNet50 --batch-size=64 &
CUDA_VISIBLE_DEVICES="1" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 --model=ResNet50 --batch-size=64 &
CUDA_VISIBLE_DEVICES="2" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=2 --model=ResNet50 --batch-size=64 &
CUDA_VISIBLE_DEVICES="3" python3 benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=3 --model=ResNet50 --batch-size=64 &
```
