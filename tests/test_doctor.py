"""`omnia doctor` must report every reader and never crash, whatever is installed."""
from omnia_sdk import doctor


def test_report_lists_every_reader():
    rows = doctor.report()
    assert {r["module"] for r in rows} == {"openslide", "pydicom", "wsidicom", "nibabel", "torch"}
    for r in rows:
        assert isinstance(r["ok"], bool) and r["detail"]


def test_missing_module_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(doctor, "CHECKS", [("Nothing", "definitely_not_a_module_xyz", ["x"], "x")])
    assert doctor.main(fix=False) == 1
