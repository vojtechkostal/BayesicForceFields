from pathlib import Path

from bff.io.colvars import write_mdp_with_colvars
from bff.io.mdp import patch_mdp, read_mdp


def test_read_mdp_ignores_comments_and_normalizes_keys(tmp_path: Path) -> None:
    src = tmp_path / "in.mdp"
    src.write_text(
        "; comment\n"
        "   ; indented comment\n"
        "\n"
        "integrator = md\n"
        "nstxout_compressed = 500 ; 1 ps\n"
        "Tcoupl = v-rescale\n"
    )

    assert read_mdp(src) == {
        "integrator": "md",
        "nstxout-compressed": "500",
        "tcoupl": "v-rescale",
    }


def test_patch_mdp_replaces_and_appends_keeping_other_lines(tmp_path: Path) -> None:
    src = tmp_path / "in.mdp"
    src.write_text("; header\n\nnsteps = 1000 ; old\ndt = 0.001\n")
    patched = tmp_path / "patched.mdp"

    patch_mdp(src, {"nsteps": 2000, "dt": "0.002", "colvars_active": "yes"}, patched)

    lines = patched.read_text().splitlines()
    assert lines[:2] == ["; header", ""]
    assert read_mdp(patched) == {
        "nsteps": "2000",
        "dt": "0.002",
        "colvars-active": "yes",
    }


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
