

# Evaluation reproduce
Run from the root of the project:


## Garbench

**MAX Area**

```python custom_benchmark/run_garbanche.py --garbage_rates 0.8 0.85 0.9 0.95 --cache-config-path custom_benchmark/config/garbench_maxarea_config.yaml```

**Baseline (LRU)**
```python custom_benchmark/run_garbanche.py --garbage_rates 0.8 0.85 0.9 0.95 --cache-config-path custom_benchmark/config/garbench_baseline_config.yaml```


## PAWS

**MAX AREA**

```python custom_benchmark/run_paws_bench.py  --cache-config-path custom_benchmark/config/paws_conf.yaml```

**Baseline (LRU)**
```python custom_benchmark/run_paws_bench.py  --cache-config-path custom_benchmark/config/paws_baseline_conf.yaml```