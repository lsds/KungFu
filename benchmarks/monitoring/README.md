# Training Monitoring Benchmark

## Gradient Variance

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark.py --kf-optimizer=variance --model=ResNet50 --batch-size=64
```

## Gradient Noise Scale

Use the following command to run the KungFu benchmark.

```bash
kungfu-run -np 4 python3 benchmark.py --kf-optimizer=noise-scale --model=ResNet50 --batch-size=64
```
