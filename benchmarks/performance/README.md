# Distributed Training Benchmark

## Parameter Servers

```bash
# Configure two local parameter servers
PS_HOSTS="localhost:2220,localhost:2221"

# Configure two training workers
WORKER_HOSTS="localhost:2224,localhost:2225"

# Start parameter servers on CPUs
CUDA_VISIBLE_DEVICES="" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &
CUDA_VISIBLE_DEVICES="" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=1 &

# Start workers on different GPUs
CUDA_VISIBLE_DEVICES="0" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 &
CUDA_VISIBLE_DEVICES="1" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 &

wait
```
