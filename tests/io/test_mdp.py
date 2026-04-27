from pathlib import Path

from bff.io.mdp import get_n_frames_target, patch_mdp, read_mdp, write_mdp


def test_read_write_mdp_preserves_comments_and_patches_values(tmp_path: Path) -> None:
    src = tmp_path / "in.mdp"
    src.write_text(
        "; comment\n"
        "\n"
        "integrator = md\n"
        "nsteps = 1000\n"
        "nstxout-compressed = 100\n"
    )

    content = read_mdp(src)

    assert list(content)[:2] == ["C000", "B000"]
    assert content["integrator"] == "md"

    patched = tmp_path / "patched.mdp"
    patch_mdp(src, {"nsteps": 2000, "dt": "0.002"}, patched)
    patched_content = read_mdp(patched)

    assert patched_content["nsteps"] == "2000"
    assert patched_content["dt"] == "0.002"
    assert get_n_frames_target(patched) == (20, 100)

    rewritten = tmp_path / "rewritten.mdp"
    write_mdp(content, rewritten)
    assert rewritten.read_text().startswith("; comment\n\n")


def test_get_n_frames_target_returns_none_without_steps(tmp_path: Path) -> None:
    mdp = tmp_path / "empty.mdp"
    mdp.write_text("integrator = md\n")

    assert get_n_frames_target(mdp) == (None, None)
