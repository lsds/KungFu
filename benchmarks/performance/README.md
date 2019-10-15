# Distributed Training Benchmark

Distributed training benchmark of KungFu, Horovod and Parameter Servers.

We assume the benchmark runs on a server with 4 GPUs. The Tensorflow version is 1.13.1.
The benchmark imports models from [tf.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications). You can freely choose different models
and batch sizes.

## Synchronous Training

In the synchronous training benchmark, we compare KungFu with Horovod.

### Horovod

There are three steps to install Horovod.

- Install OpenMPI with the [script](https://raw.githubusercontent.com/tensorlayer/openpose-plus/master/scripts/install-mpi.sh)
- Install Horovod with `pip3 install horovod==0.16.1`

Use the following command to run the Horovod benchmark.

```bash
mpirun -np 4 python3 benchmark_horovod.py --model=ResNet50 --batch-size=64
```

### KungFu sync

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark_kungfu.py --kungfu=sync-sgd --model=ResNet50 --batch-size=64
```

## Asynchronous Training

Asynchronous training is useful for addressing network bottlenecks, stragglers and unpredictable availability of resource (e.g., AWS spot instances). In the asynchronous benchmark, we compare KungFu with Parameter Servers.

### Parameter Servers

Use the following shell script to run the parameter server benchmark.

```bash
# Configure 1 local parameter server (You can create more parameter servers)
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

### KungFu async

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark_kungfu.py --kungfu=async-sgd --model=ResNet50 --batch-size=64
```
