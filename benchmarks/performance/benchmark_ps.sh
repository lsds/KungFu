# Configure two local parameter servers
PS_HOSTS="localhost:2220"

# Configure two training workers
WORKER_HOSTS="localhost:2230,localhost:2231,localhost:2232,localhost:2233"

# Start parameter servers on CPUs
CUDA_VISIBLE_DEVICES="" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=ps --task_index=0 &

# Start workers on different GPUs
CUDA_VISIBLE_DEVICES="0" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=0 &
CUDA_VISIBLE_DEVICES="1" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=1 &
CUDA_VISIBLE_DEVICES="2" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=2 &
CUDA_VISIBLE_DEVICES="3" python benchmark_ps.py --ps_hosts=$PS_HOSTS --worker_hosts=$WORKER_HOSTS --job_name=worker --task_index=3 &
