from pathlib import Path

from bff.io.colvars import write_mdp_with_colvars
from bff.io.mdp import patch_mdp, read_mdp, write_mdp


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

    rewritten = tmp_path / "rewritten.mdp"
    write_mdp(content, rewritten)
    assert rewritten.read_text().startswith("; comment\n\n")


def test_colvars_path_is_relative_to_gromacs_working_directory(
    tmp_path: Path,
) -> None:
    production = tmp_path / "systems/acetate/production.mdp"
    production.parent.mkdir(parents=True)
    production.write_text("integrator = md\n")
    run_dir = tmp_path / "campaign"
    sample_dir = run_dir / "samples/000/acetate"
    sample_dir.mkdir(parents=True)
    colvars = sample_dir / "bias.colvars.dat"
    colvars.write_text("colvar {}\n")
    output = sample_dir / "production-colvars.mdp"

    write_mdp_with_colvars(
        production,
        colvars,
        output,
        working_dir=run_dir,
    )

    content = read_mdp(output)
    assert content["colvars-active"] == "yes"
    assert content["colvars-configfile"] == "samples/000/acetate/bias.colvars.dat"
