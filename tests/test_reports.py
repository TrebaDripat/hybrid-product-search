import json
import subprocess
import sys
from pathlib import Path

from hybrid_search.evaluation import source_info

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "quality-v2"


def run_comparison(before, after):
    return subprocess.run(
        [sys.executable, ROOT / "scripts" / "compare_runs.py", before, after],
        capture_output=True,
        text=True,
    )


def test_comparison_rebuilds_wands_metrics_and_rejects_unpaired_data(tmp_path):
    before, after = REPORTS / "before-wands.json", REPORTS / "after-wands.json"
    paired = run_comparison(before, after)
    assert paired.returncode == 0, paired.stderr
    assert "| hybrid | after | 0.8041 | 0.9583 | 0.1434 |" in paired.stdout
    other = run_comparison(REPORTS / "before-demo.json", after)
    assert other.returncode == 2 and "dataset_sha256" in other.stderr
    broken = json.loads(after.read_text())
    broken["samples"].pop()
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(broken))
    incomplete = run_comparison(before, path)
    assert incomplete.returncode == 2 and "missing or duplicate" in incomplete.stderr


def test_source_revision_follows_the_tracked_module(tmp_path, monkeypatch):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    module = checkout / "evaluation.py"
    module.write_text("value = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "add", module.name], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "Fixture module",
        ],
        cwd=checkout,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    monkeypatch.chdir(tmp_path)
    assert source_info(module) == {"git_revision": revision, "working_tree_dirty": False}
    monkeypatch.chdir(checkout)
    assert source_info(tmp_path / "wheel.py")["git_revision"] == "unavailable in this deployment"
