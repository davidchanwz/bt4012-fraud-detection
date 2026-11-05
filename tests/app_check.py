"""Headless check for the dashboard: renders each tab under many sidebar selections, no browser needed.

    uv run python tests/app_check.py                 # every tab, main selections (~15 s)
    uv run python tests/app_check.py budget --full   # one tab, all 96 selections
    uv run python tests/app_check.py --empty         # also with no tables and no data/processed

Without --full each tab gets 48 selections (every model and split, compare mode or subset "all").
--full adds every features x subset pair: 96 per tab, about 1.5 minutes for all six. Prints one
line per tab and exits non-zero if any render raises. Used by tests/test_app.py.
"""
import itertools
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
TABS = ["budget", "scoreboard", "eda", "cases", "fp_profile", "ablation"]

SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from app import data as D
from app.tabs import {tab} as T
T.render(D.Selection(split={split!r}, model={model!r}, feature_sets={fs!r}, subset={subset!r}))
"""


def _quiet_streamlit():
    """AppTest runs outside a browser session, so Streamlit logs 'missing ScriptRunContext' (and a
    cache notice) on every render. Harmless, but hundreds of lines of it hide the result."""
    from streamlit.logger import get_logger
    # AppTest resets Streamlit's log levels on every run, so a level change doesn't stick; a filter does.
    for name in ("streamlit.runtime.scriptrunner_utils.script_run_context",
                 "streamlit.runtime.caching.cache_data_api"):
        lg = get_logger(name)
        if not any(isinstance(f, _DropWarnings) for f in lg.filters):
            lg.addFilter(_DropWarnings())


class _DropWarnings(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR


def selections(full=False):
    models = ["lr", "mlp", "lgbm", "rf"]
    fss = [("txn", "txn_graph"), ("txn",), ("txn_graph",)]
    subsets = ["all", "normal_looking", "normal_looking_strict", "normal_looking_q90"]
    combos = list(itertools.product(["val", "test"], models, fss, subsets))
    return combos if full else [c for c in combos if c[2] == fss[0] or c[3] == "all"]


def check(tab, quick=True):
    from streamlit.testing.v1 import AppTest
    _quiet_streamlit()
    t0 = time.time()
    combos = selections(full=not quick)
    failures = []
    for split, model, fs, subset in combos:
        at = AppTest.from_string(SCRIPT.format(root=str(ROOT), tab=tab, split=split, model=model, fs=fs,
                                               subset=subset), default_timeout=60).run()
        if at.exception:
            failures.append(((split, model, fs, subset), at.exception[0].value))
    where = "no tables" if os.environ.get("DASHBOARD_TABLE_DIR") else "reports/tables"
    print(f"  {tab:<11} [{where}] {len(combos) - len(failures)}/{len(combos)} selections OK "
          f"({time.time() - t0:.0f}s)", flush=True)
    for sel, err in failures[:10]:
        print("    FAIL", sel, "->", err[:300], flush=True)
    return not failures


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tabs = TABS if not args or args[0] == "all" else args
    quick = "--full" not in sys.argv
    if "--child" in sys.argv:                      # the --empty run, in a fresh process
        sys.exit(0 if all([check(t, quick) for t in tabs]) else 1)
    print(f"Rendering {len(tabs)} tab(s), {len(selections(not quick))} sidebar selections each", flush=True)
    ok = all([check(t, quick) for t in tabs])
    if "--empty" in sys.argv:
        # A fresh process, because app.data reads the table folder once, at import.
        with TemporaryDirectory() as d:
            env = dict(os.environ, DASHBOARD_TABLE_DIR=d, DASHBOARD_CLEAN_FILE=str(Path(d) / "none.parquet"))
            r = subprocess.run([sys.executable, __file__, *tabs, "--child", *(["--full"] if not quick else [])],
                               env=env, cwd=ROOT)
            ok = ok and r.returncode == 0
    print("All OK" if ok else "FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
