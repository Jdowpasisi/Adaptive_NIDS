"""C18: every table / figure in reports/ has a registered producer, and the registry is consistent."""

import pytest

from xnids import reporting as R
from xnids.utils import paths


def test_every_report_file_has_a_producer():
    files = R.outputs()
    if not files:
        pytest.skip("no reports yet")
    missing = [str(p.relative_to(paths.REPORTS)) for p in files if R.step_for(p) is None]
    assert not missing, f"no step in xnids.reporting.STEPS produces: {missing}"


def test_registry_is_consistent():
    names = [s.name for s in R.STEPS]
    assert len(names) == len(set(names))
    for s in R.STEPS:
        assert s.kind in ("mlflow", "derived", "pipeline")
        if s.kind == "mlflow":
            assert s.experiments, s.name                  # an MLflow step must say which runs it is built from
        script = s.cmd[1] if s.cmd[0] == "bash" else s.cmd[0]
        assert (paths.REPO / script).exists(), script
