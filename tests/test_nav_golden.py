"""The Python navigator must keep matching tests/golden/nav_cases.json, and
models/nav_config.json must match configs/default.yaml.

The phone app runs the same golden cases against its own implementation. If a rule is
changed on purpose: update docs/navigation_spec.md, run scripts/export_nav_config.py and
scripts/make_nav_golden.py, and review the diff of both JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_nav_config  # noqa: E402
import make_nav_golden  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "golden" / "nav_cases.json").read_text())
NAV_JSON = ROOT / "models" / "nav_config.json"


def test_nav_config_json_is_up_to_date_with_the_yaml() -> None:
    assert json.loads(NAV_JSON.read_text()) == export_nav_config.build_nav_config(), (
        "models/nav_config.json is stale: run python scripts/export_nav_config.py")


def test_nav_config_has_the_naming_rule() -> None:
    nav = json.loads(NAV_JSON.read_text())["navigator"]
    assert nav["name_groups"] == {"post": ["traffic_cone", "bollard", "fire_hydrant", "pole"]}
    assert nav["name_min_conf"] == 0.6 and nav["phrases"]["unknown"] == "Obstacle"


def test_golden_file_shape() -> None:
    cases = GOLDEN["cases"]
    assert len(cases) == 30 and len({c["id"] for c in cases}) == 30
    texts = [e["text"] for c in cases for e in c["expected"]]
    assert any(t.startswith("Obstacle") for t in texts), "the naming rule must be covered"
    assert any(e["priority"] == "urgent" for c in cases for e in c["expected"])
    assert sum(1 for c in cases if not c["expected"]) >= 5, "silent cases must be covered"


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["id"] for c in GOLDEN["cases"]])
def test_python_matches_golden(case: dict[str, Any]) -> None:
    nav = make_nav_golden.load_nav(NAV_JSON)
    assert make_nav_golden.run_case(case, nav) == case["expected"]


def test_generator_inputs_match_the_stored_cases() -> None:
    """The stored inputs are exactly what the generator defines (no hand edits)."""
    stored = [{k: v for k, v in c.items() if k != "expected"} for c in GOLDEN["cases"]]
    assert stored == make_nav_golden.cases()
