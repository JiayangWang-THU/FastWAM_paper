# LIBERO Horizon 验证（一键运行）

这个目录用于做你说的第一步实验：

> 先验证 **horizon 长度** 在 LIBERO 不同任务上的影响；并支持在推理阶段开启/关闭 dream（通过 `infer_joint` 生成未来视频分支）对比。

## 这套脚本做了什么

- 自动 sweep 多个 horizon（例如 `5,9,13,17`，均满足 `T % 4 == 1`）
- 每个 horizon 自动调用：
  - `experiments/libero/run_libero_manager.py`
  - `experiments/libero/summarize_results.py`
- 最终自动汇总成：
  - `horizon_sweep_summary.csv`
  - `horizon_sweep_summary.json`

## 快速开始（One-liner）

```bash
python exoeriments/libero_horizon/run_horizon_sweep.py \
  --task libero_uncond_2cam224_1e-4 \
  --ckpt ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats ./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --horizons 5,9,13,17 \
  --suite libero_spatial \
  --num-gpus 1 \
  --dream-mode off \
  --output-root ./evaluate_results/horizon_sweep
```

## 开启 dream 推理（对比组）

```bash
python exoeriments/libero_horizon/run_horizon_sweep.py \
  --task libero_uncond_2cam224_1e-4 \
  --ckpt ./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats ./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --horizons 5,9,13,17 \
  --suite libero_spatial \
  --num-gpus 1 \
  --dream-mode on \
  --output-root ./evaluate_results/horizon_sweep
```

`dream-mode on` 会设置：
- `EVALUATION.visualize_future_video=true`（走 `infer_joint`）
- `EVALUATION.test_action_with_infer_action=false`（避免额外一致性检查导致的 seed 依赖）

## 输出结构

```text
<output-root>/
  horizon_sweep_summary.csv
  horizon_sweep_summary.json
  horizon_5/
  horizon_9/
  horizon_13/
  horizon_17/
```

每个 `horizon_x/` 下都有 `summary.json`（由 `summarize_results.py` 生成）。

## 注意

- horizon 必须满足 `T % 4 == 1`。
- 这个实验是“先看长度影响”的 baseline 版本；它不包含 ADG gate/critic 训练逻辑。
