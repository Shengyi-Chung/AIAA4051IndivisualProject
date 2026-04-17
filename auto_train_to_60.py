import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_ACC = 0.60
KEEP_TOP_K = 3
PROGRESS_PATH = ROOT / "auto60_progress.json"


@dataclass
class Trial:
    tag: str
    rank: int
    alpha: int
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    dropout: float
    padding_side: str
    target_modules: str

    @property
    def model_dir(self) -> Path:
        return ROOT / f"model_auto60-{self.tag}"

    @property
    def output_dir(self) -> Path:
        return ROOT / f"outputs_auto60-{self.tag}"


TRIALS = [
    Trial("base-r32-e2-lr1e4-qkvo-right", 32, 64, 2, 1e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r32-e3-lr1e4-qkvo-right", 32, 64, 3, 1e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r32-e2-lr15e5-qkvo-right", 32, 64, 2, 1.5e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r32-e3-lr15e5-qkvo-right", 32, 64, 3, 1.5e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r64-e2-lr1e4-qkvo-right", 64, 128, 2, 1e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r32-e2-lr2e4-qkvo-right", 32, 64, 2, 2e-4, 4, 4, 0.1, "right", "q_proj,k_proj,v_proj,o_proj"),
    Trial("r32-e2-lr1e4-qv-right", 32, 64, 2, 1e-4, 4, 4, 0.1, "right", "q_proj,v_proj"),
    Trial("r32-e3-lr1e4-qv-right", 32, 64, 3, 1e-4, 4, 4, 0.1, "right", "q_proj,v_proj"),
]


def run_trial(trial: Trial) -> float:
    cmd = [
        "python",
        "train.py",
        "--rank",
        str(trial.rank),
        "--alpha",
        str(trial.alpha),
        "--epochs",
        str(trial.epochs),
        "--learning_rate",
        str(trial.lr),
        "--batch_size",
        str(trial.batch_size),
        "--grad_accum",
        str(trial.grad_accum),
        "--dropout",
        str(trial.dropout),
        "--padding_side",
        trial.padding_side,
        "--target_modules",
        trial.target_modules,
        "--mask_question",
        "false",
        "--use_wandb",
        "false",
        "--wandb_mode",
        "disabled",
        "--output_dir",
        str(trial.model_dir),
        "--report_dir",
        str(trial.output_dir),
    ]

    print(f"\n===== RUN {trial.tag} =====")
    subprocess.run(cmd, cwd=ROOT, check=True)

    metrics_path = trial.output_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics.json: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    acc = float(metrics.get("accuracy", 0.0))
    print(f"[RESULT] {trial.tag} accuracy={acc:.4f}")
    return acc


def keep_top_k(results: list[dict], k: int) -> None:
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    keep_tags = {row["tag"] for row in sorted_results[:k]}

    for row in sorted_results[k:]:
        tag = row["tag"]
        if tag in keep_tags:
            continue

        model_dir = ROOT / f"model_auto60-{tag}"
        output_dir = ROOT / f"outputs_auto60-{tag}"

        if model_dir.exists():
            shutil.rmtree(model_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)


def save_progress(results: list[dict], reached: bool, best_tag: str | None) -> None:
    payload = {
        "target_accuracy": TARGET_ACC,
        "reached": reached,
        "best_tag": best_tag,
        "results": sorted(results, key=lambda x: x["accuracy"], reverse=True),
    }
    PROGRESS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    results: list[dict] = []
    reached = False
    best_tag = None

    for trial in TRIALS:
        try:
            acc = run_trial(trial)
        except subprocess.CalledProcessError as e:
            print(f"[FAILED] {trial.tag}: exit={e.returncode}")
            continue
        except Exception as e:
            print(f"[FAILED] {trial.tag}: {e}")
            continue

        results.append({"tag": trial.tag, "accuracy": acc})
        keep_top_k(results, KEEP_TOP_K)

        if acc >= TARGET_ACC:
            reached = True
            best_tag = trial.tag
            break

    if results and best_tag is None:
        best_tag = sorted(results, key=lambda x: x["accuracy"], reverse=True)[0]["tag"]

    save_progress(results, reached, best_tag)

    print("\n===== SUMMARY =====")
    for row in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{row['tag']}: {row['accuracy']:.4f}")

    if reached:
        print(f"TARGET REACHED with {best_tag}")
        return 0

    print(f"TARGET NOT REACHED. Best={best_tag}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())