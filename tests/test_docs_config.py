"""Every configuration key a stage accepts is documented on its page."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / "bff" / "workflows"
DOCS = ROOT / "docs" / "configuration"
PAGES = {
    "build": "build.md",
    "campaign": "snippets/campaign-options.md",
    "sample_parameters": "sample-parameters.md",
    "validate": "validate.md",
    "build_qoi_datasets": "build-qoi-datasets.md",
    "fit_lgp": "fit-lgp.md",
    "learn": "learn.md",
}


def accepted_keys(module: Path) -> set[str]:
    """String literals in ``allowed=``/``stage_keys=`` and ``*_KEYS``/``*_ROLES``."""
    keys: set[str] = set()
    for node in ast.walk(ast.parse(module.read_text())):
        values = []
        if isinstance(node, ast.keyword) and node.arg in {"allowed", "stage_keys"}:
            values.append(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.endswith(("_KEYS", "_ROLES"))
            for target in node.targets
        ):
            values.append(node.value)
        for value in values:
            keys.update(
                item.value
                for item in ast.walk(value)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return keys


@pytest.mark.parametrize("stage", sorted(PAGES))
def test_configuration_page_documents_every_key(stage: str) -> None:
    keys = accepted_keys(WORKFLOWS / stage / "config.py")
    assert keys, f"no configuration keys found for {stage}"
    page = re.sub(r"```.*?```", "", (DOCS / PAGES[stage]).read_text(), flags=re.S)
    documented = {
        part
        for span in re.findall(r"`([^`]+)`", page)
        for part in re.split(r"[.\[\]<>\s]+", span)
    }
    assert sorted(keys - documented) == []
