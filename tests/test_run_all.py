"""Checks for the one-command pipeline, mainly that it can't touch the test split by accident."""
import importlib.util

import pytest

from src import run_all as R
from src.run_all import Step, build_plan, execute


def modules(plan):
    return [s.module for s in plan]


def position(plan, module, *args):
    for i, s in enumerate(plan):
        if s.module == module and all(a in s.args for a in args):
            return i
    raise AssertionError(f"{module} {args} not in the plan")


def touches_test(step):
    return "test" in step.args or "--score-test" in step.args


def test_default_plan_never_touches_test():
    assert not [s.label for s in build_plan() if touches_test(s) or s.scores_test]


def test_score_test_covers_every_test_step():
    plan = build_plan(score_test=True)
    assert [s.scores_test for s in plan].count(True) == 2           # main models and RF
    for module, args in [("src.eval.metrics", ("--split", "test")),
                         ("src.eval.metrics", ("--split", "test", "--name", "rf")),
                         ("src.eval.figures", ("--split", "test")),
                         ("src.models.results_table", ("--split", "test")),
                         ("src.models.case_studies", ("--split", "test", "lgbm", "0.02")),
                         ("src.models.case_studies", ("--split", "test", "lr")),
                         ("src.models.interpret", ("--split", "test"))]:
        position(plan, module, *args)


def test_order_respects_dependencies():
    plan = build_plan(score_test=True)
    # figure 07 needs the graph features
    assert position(plan, "src.features.graph") < position(plan, "src.eda")
    # the plan's documented order for the case viewer: shortlists before contributions
    assert position(plan, "src.models.case_studies", "test") < position(plan, "src.models.interpret", "test")
    assert position(plan, "src.models.run_main") < position(plan, "src.eval.metrics")
    # results_table formats what metrics wrote
    assert position(plan, "src.eval.metrics", "test") < position(plan, "src.models.results_table", "test")


def test_every_module_exists():
    for step in build_plan(score_test=True):
        if step.module != "pytest":
            assert importlib.util.find_spec(step.module), step.module


def test_from_drops_earlier_stages():
    assert {s.stage for s in build_plan(start="eval", tests=False)} == {"eval", "interpret"}


def test_no_tests_flag():
    assert "pytest" not in modules(build_plan(tests=False))


def test_execute_stops_at_first_failure():
    plan = [Step("data", "a"), Step("data", "b"), Step("data", "c")]
    ran = []

    def fail_on_b(step):
        ran.append(step.module)
        return 3 if step.module == "b" else 0

    assert execute(plan, runner=fail_on_b, log=lambda *_: None) == 3
    assert ran == ["a", "b"]


def test_skip_existing_needs_every_output(tmp_path):
    there = tmp_path / "there"
    there.touch()
    plan = [Step("data", "complete", (), (there,)),
            Step("data", "partial", (), (there, tmp_path / "missing")),
            Step("data", "no_outputs")]
    ran = []
    runner = lambda step: ran.append(step.module) or 0
    assert execute(plan, skip_existing=True, runner=runner, log=lambda *_: None) == 0
    assert ran == ["partial", "no_outputs"]


def test_refuses_to_score_test_twice(tmp_path, monkeypatch, capsys):
    (tmp_path / "lr_txn_test.parquet").touch()
    monkeypatch.setattr(R, "SCORE_DIR", tmp_path)
    monkeypatch.setattr(R, "execute", lambda *a, **k: pytest.fail("should not have started"))
    assert R.main(["--score-test"]) == 2
    assert "Refusing" in capsys.readouterr().out


def test_redoing_test_evaluation_is_allowed(tmp_path, monkeypatch):
    (tmp_path / "lr_txn_test.parquet").touch()
    monkeypatch.setattr(R, "SCORE_DIR", tmp_path)
    monkeypatch.setattr(R, "execute", lambda *a, **k: 0)
    monkeypatch.setattr(R, "changed_results", lambda: [])
    assert R.main(["--score-test", "--from", "eval"]) == 0
