"""Download or copy example workflows for local use."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
from importlib import metadata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .. import __version__

_REPOSITORY = "https://github.com/vojtechkostal/BayesicForceFields"
_ARCHIVE_API = "https://api.github.com/repos/vojtechkostal/BayesicForceFields/tarball"


def _installed_version() -> str:
    try:
        return metadata.version("bfflearn")
    except metadata.PackageNotFoundError:
        return __version__


def _default_ref() -> str:
    return f"v{_installed_version()}"


def _local_examples_dir() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"
    if examples_dir.is_dir() and (repo_root / ".git").exists():
        return examples_dir
    return None


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)


def _copy_local_examples(source_dir: Path, output_dir: Path) -> Path:
    repo_root = source_dir.parent
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                source_dir.name,
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not list example files in the source checkout."
        ) from exc

    tracked_files = [Path(path) for path in result.stdout.split("\0") if path]
    if not tracked_files:
        raise RuntimeError("The source checkout does not contain examples.")

    output_dir.mkdir(parents=True)
    for tracked_file in tracked_files:
        source_path = repo_root / tracked_file
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_dir)
        target_path = output_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return output_dir


def _download_archive(ref: str, archive_path: Path) -> None:
    request = Request(
        f"{_ARCHIVE_API}/{ref}",
        headers={"User-Agent": "bfflearn-example-fetcher"},
    )
    try:
        with urlopen(request) as response, archive_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except HTTPError as exc:
        raise RuntimeError(
            f"Could not download examples for ref '{ref}' from {_REPOSITORY}."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not reach GitHub to download examples. "
            "Check your internet connection and try again."
        ) from exc


def _extract_examples_archive(archive_path: Path, output_dir: Path) -> Path:
    files_written = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] != "examples":
                continue

            relative_path = Path(*parts[2:])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(
                    "Downloaded examples archive contains an unsafe path."
                )
            target_path = output_dir / relative_path

            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted, target_path.open("wb") as out_file:
                shutil.copyfileobj(extracted, out_file)
            files_written += 1

    if files_written == 0:
        raise RuntimeError("Downloaded archive did not contain an examples directory.")

    return output_dir


def fetch_examples(
    output_dir: Path,
    *,
    ref: str | None = None,
    force: bool = False,
) -> tuple[Path, str]:
    """
    Fetch all repository examples into ``output_dir``.

    If running from a source checkout, examples are copied from the local
    repository. Otherwise, the example tree is downloaded from GitHub for the
    requested ref or the installed package version tag.
    """

    output_dir = output_dir.expanduser().resolve()
    _prepare_output_dir(output_dir, force=force)

    local_examples = _local_examples_dir()
    if local_examples is not None:
        return _copy_local_examples(local_examples, output_dir), "local checkout"

    resolved_ref = ref or _default_ref()
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "examples.tar.gz"
        _download_archive(resolved_ref, archive_path)
        return (
            _extract_examples_archive(archive_path, output_dir),
            f"{_REPOSITORY} @ {resolved_ref}",
        )
