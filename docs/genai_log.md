# GenAI use log

Log GenAI help as you go (date, who, tool, what for, what you checked). This feeds the GenAI acknowledgement in the report appendix. Everyone must be able to explain every part of the submission.

| Date | Who | Tool | What for | How it was checked |
|---|---|---|---|---|
| 1 Nov 2026 | Aayush (R3) | Claude Code (Claude Sonnet / Opus) | Wrote `CLAUDE.md` and `docs/r3_plan.md`. Wrote `src/models/` (`train`, `run_main`, `interpret`, `case_studies`, `results_table`, `mock_scores`), `tests/test_models.py`, `tests/test_reporting.py`, and the §3.3 draft and §4.1 skeleton | Unit tests (no test-split access without `--score-test`, fit only on train rows, deterministic reruns). Reran on real data. Read the code. Checked the `g1_cust_n_prev` effect by hand |
