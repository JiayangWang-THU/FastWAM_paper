import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_horizons(text: str) -> list[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("`--horizons` is empty.")
    for h in values:
        if h <= 0:
            raise ValueError(f"Invalid horizon {h}, must be > 0.")
        if h % 4 != 1:
            raise ValueError(f"Invalid horizon {h}, must satisfy T % 4 == 1.")
    return values


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_manager_cmd(
    task: str,
    ckpt: str,
    dataset_stats: str,
    output_dir: Path,
    num_gpus: int,
    suite: str,
    horizon: int,
    dream_mode: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/libero/run_libero_manager.py",
        f"task={task}",
        f"ckpt={ckpt}",
        f"EVALUATION.dataset_stats_path={dataset_stats}",
        f"EVALUATION.output_dir={str(output_dir)}",
        f"EVALUATION.task_suite_name={suite}",
        f"EVALUATION.num_video_frames={horizon}",
        f"MULTIRUN.num_gpus={num_gpus}",
    ]

    if dream_mode == "on":
        cmd.extend(
            [
                "EVALUATION.visualize_future_video=true",
                "EVALUATION.test_action_with_infer_action=false",
            ]
        )
    else:
        cmd.append("EVALUATION.visualize_future_video=false")

    return cmd


def build_summarize_cmd(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "experiments/libero/summarize_results.py",
        "--output_dir",
        str(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep LIBERO eval over horizon lengths.")
    parser.add_argument("--task", required=True, help="Hydra task name, e.g. libero_uncond_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--dataset-stats", required=True, help="dataset_stats.json path")
    parser.add_argument("--horizons", required=True, help="Comma-separated horizons, e.g. 5,9,13,17")
    parser.add_argument("--suite", default="libero_spatial", help="LIBERO suite name")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dream-mode", choices=["on", "off"], default="off")
    parser.add_argument("--output-root", default="./evaluate_results/horizon_sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        run_dir = output_root / f"horizon_{horizon}"
        run_dir.mkdir(parents=True, exist_ok=True)

        manager_cmd = build_manager_cmd(
            task=args.task,
            ckpt=args.ckpt,
            dataset_stats=args.dataset_stats,
            output_dir=run_dir,
            num_gpus=int(args.num_gpus),
            suite=args.suite,
            horizon=horizon,
            dream_mode=args.dream_mode,
        )
        run_cmd(manager_cmd, dry_run=args.dry_run)

        summarize_cmd = build_summarize_cmd(run_dir)
        run_cmd(summarize_cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        summary = load_summary(run_dir / "summary.json")
        overall = summary.get("overall", {})

        row = {
            "horizon": horizon,
            "dream_mode": args.dream_mode,
            "suite": args.suite,
            "average_success_rate": overall.get("average_success_rate"),
            "average_task_time": overall.get("average_task_time"),
            "total_time": overall.get("total_time"),
            "average_future_video_psnr": overall.get("average_future_video_psnr", None),
            "run_dir": str(run_dir),
        }
        aggregate_rows.append(row)

    if args.dry_run:
        print("[DRY-RUN] Skip writing aggregate files.")
        return

    aggregate_json = output_root / "horizon_sweep_summary.json"
    aggregate_csv = output_root / "horizon_sweep_summary.csv"

    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate_rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "horizon",
        "dream_mode",
        "suite",
        "average_success_rate",
        "average_task_time",
        "total_time",
        "average_future_video_psnr",
        "run_dir",
    ]
    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"[DONE] Aggregate JSON: {aggregate_json}")
    print(f"[DONE] Aggregate CSV : {aggregate_csv}")


if __name__ == "__main__":
    main()
