"""Run the whole pipeline with one command, from the raw download to the results tables.

    uv run python -m src.run_all                     # everything, val only (about 15 min)
    uv run python -m src.run_all --dry-run           # print the steps, run nothing
    uv run python -m src.run_all --from eval         # restart from a stage
    uv run python -m src.run_all --skip-existing     # skip steps whose outputs already exist
    uv run python -m src.run_all --no-tests          # leave out pytest at the end
    uv run python -m src.run_all --score-test        # at the freeze only (Sun 15 Nov), see below

Stages, in order: data, features, eda, models, eval, interpret, tests.

The test split is never touched without --score-test. With it, the models stage also writes test
scores and the later stages add the test tables next to the val ones. The protocol is to score
test once, so the run refuses to start if test score files already exist.

Each step is the same module the README documents, run in its own Python process. That keeps this
file a list of steps: it computes nothing itself, and it works anywhere Python does (no bash).
It does not cover R2's ablation or label-delay runs, which have their own commands.

Reruns are meant to reproduce the committed tables, but check before committing: the end of the run
lists every file under reports/ that now differs from git. hyperparams.csv always differs because
it stores the time of the run.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from src.config import CLEAN_FILE, FIG_DIR, GRAPH_FILE, RAW_FILE, ROOT, SCORE_DIR, SPLIT_DIR, TABLE_DIR

STAGES = ("data", "features", "eda", "models", "eval", "interpret", "tests")
MAIN_MODELS = ("lr", "mlp", "lgbm")
FEATURE_SETS = ("txn", "txn_graph")


@dataclass(frozen=True)
class Step:
    stage: str
    module: str
    args: tuple = ()
    outputs: tuple = ()             # --skip-existing skips the step if all of these exist
    scores_test: bool = False       # writes test-split score files

    @property
    def label(self):
        return " ".join((self.module, *self.args))


def score_files(models, split):
    return tuple(SCORE_DIR / f"{m}_{f}_{split}.parquet" for m in models for f in FEATURE_SETS)


def model_steps(score_test):
    extra, split = (("--score-test",), "test") if score_test else ((), "val")
    return [
        Step("models", "src.models.run_main", extra, score_files(MAIN_MODELS, split), score_test),
        Step("models", "src.models.run_main", ("--models", "rf", *extra), score_files(("rf",), split), score_test),
    ]


def eval_steps(split):
    """Metrics, figure and formatted tables for one split. RF stays in its own results_rf_* family."""
    rf_main = f"reports/tables/results_rf_{split}.csv"
    rf_paired = f"reports/tables/results_rf_paired_{split}.csv"
    rf_md = f"reports/tables/results_rf_{split}.md"
    return [
        Step("eval", "src.eval.metrics", ("--split", split), (TABLE_DIR / f"results_main_{split}.csv",)),
        Step("eval", "src.eval.metrics", ("--split", split, "--models", "rf", "--name", "rf"),
             (ROOT / rf_main,)),
        Step("eval", "src.eval.figures", ("--split", split), (FIG_DIR / f"budget_curve_{split}.png",)),
        Step("eval", "src.eval.business", ("--split", split), (TABLE_DIR / f"business_{split}.csv",)),
        Step("eval", "src.models.results_table", ("--split", split), (TABLE_DIR / f"results_main_{split}.md",)),
        Step("eval", "src.models.results_table",
             ("--split", split, "--main", rf_main, "--paired", rf_paired, "--out", rf_md), (ROOT / rf_md,)),
    ]


def interpret_steps(split):
    """Case studies first: interpret writes contributions only for shortlists that already exist."""
    def candidates(model, pct):
        return (TABLE_DIR / f"case_study_candidates_{model}_{split}_{pct}pct.csv",)
    steps = [
        Step("interpret", "src.models.case_studies", ("--model", "lgbm", "--split", split), candidates("lgbm", 1)),
        Step("interpret", "src.models.case_studies", ("--model", "lgbm", "--split", split, "--budget", "0.02"),
             candidates("lgbm", 2)),
        Step("interpret", "src.models.case_studies", ("--model", "lr", "--split", split), candidates("lr", 1)),
        Step("interpret", "src.models.interpret", ("--model", "lgbm", "--split", split),
             (TABLE_DIR / f"case_study_contribs_{split}.csv",)),
    ]
    if split == "val":              # LR coefficients come from the fitted model, so one split is enough
        steps.append(Step("interpret", "src.models.interpret", ("--model", "lr"),
                          (TABLE_DIR / "lr_coefficients_txn_graph.csv",)))
    return steps


def build_plan(score_test=False, start="data", tests=True):
    """Every step to run, in order. EDA comes after the features because figure 07 needs them."""
    plan = [
        Step("data", "src.data.download", (), (RAW_FILE,)),
        Step("data", "src.data.clean", (), (CLEAN_FILE,)),
        Step("data", "src.data.split", (), tuple(SPLIT_DIR / f"{s}.parquet" for s in ("train", "val", "test"))),
        Step("features", "src.features.graph", (),
             (GRAPH_FILE, GRAPH_FILE.with_name("features_graph_delay1.parquet"))),
        Step("eda", "src.eda", (), (FIG_DIR / "07_normal_looking_fraud_vs_legit.png",)),
        *model_steps(score_test),
    ]
    for split in ("val", "test") if score_test else ("val",):
        plan += eval_steps(split)
    for split in ("val", "test") if score_test else ("val",):
        plan += interpret_steps(split)
    if tests and importlib.util.find_spec("pytest"):    # pytest isn't in requirements.txt (dev only)
        plan.append(Step("tests", "pytest", ("-q",)))
    return [s for s in plan if STAGES.index(s.stage) >= STAGES.index(start)]


def is_done(step):
    return bool(step.outputs) and all(p.exists() for p in step.outputs)


def run_step(step):
    env = {**os.environ, "PYTHONUTF8": "1"}     # scripts print non-ASCII; Windows consoles default to cp1252
    return subprocess.run([sys.executable, "-m", step.module, *step.args], cwd=ROOT, env=env).returncode


def changed_results():
    """Files under reports/ that differ from git. Empty if this isn't a git checkout."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "reports"], cwd=ROOT,
                             capture_output=True, text=True)
    except FileNotFoundError:
        return []
    return [line[3:] for line in out.stdout.splitlines()] if out.returncode == 0 else []


def flushed_print(message):
    print(message, flush=True)      # keep our lines in order with the output of the child processes


def execute(plan, skip_existing=False, runner=run_step, log=flushed_print):
    """Run the steps in order and stop at the first failure. Returns the exit code."""
    t_all = time.time()
    for i, step in enumerate(plan, 1):
        name = f"[{i}/{len(plan)}] {step.stage}: {step.label}"
        if skip_existing and is_done(step):
            log(f"{name}  (skipped, outputs exist)")
            continue
        log(f"\n{name}")
        t = time.time()
        code = runner(step)
        if code != 0:
            log(f"\nFAILED with exit code {code}: {step.label}")
            log(f"Fix it, then restart with: python -m src.run_all --from {step.stage}")
            return code
        log(f"done in {time.time() - t:.0f}s")
    log(f"\nAll {len(plan)} steps finished in {(time.time() - t_all) / 60:.1f} min")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", choices=STAGES, default="data", help="first stage to run")
    p.add_argument("--skip-existing", action="store_true", help="skip steps whose outputs already exist")
    p.add_argument("--dry-run", action="store_true", help="print the steps and exit")
    p.add_argument("--no-tests", action="store_true", help="leave out pytest at the end")
    p.add_argument("--score-test", action="store_true", help="also score and evaluate test; at the freeze only")
    a = p.parse_args(argv)

    plan = build_plan(a.score_test, a.start, tests=not a.no_tests)
    if a.dry_run:
        for i, step in enumerate(plan, 1):
            done = "  (outputs exist)" if is_done(step) else ""
            print(f"{i:>2}. {step.stage:<10} {step.label}{done}")
        return 0

    will_score_test = any(s.scores_test and not (a.skip_existing and is_done(s)) for s in plan)
    existing = sorted(SCORE_DIR.glob("*_test.parquet"))
    if will_score_test and existing:
        print(f"Refusing to score test again: {len(existing)} test score files already exist, starting with "
              f"{existing[0].name}.\nThe protocol is to score test once. Use --from eval to redo the "
              "evaluation only. To really rescore, delete those files first and tell the group.")
        return 2

    code = execute(plan, a.skip_existing)
    if code == 0:
        changed = changed_results()
        if changed:
            print(f"\n{len(changed)} files under reports/ differ from git. Check them before committing:")
            print("\n".join(f"  {path}" for path in changed))
        else:
            print("\nNothing under reports/ differs from git.")
    return code


if __name__ == "__main__":
    sys.exit(main())
