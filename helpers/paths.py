from __future__ import annotations

from pathlib import Path
import shutil


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_path(name: str) -> Path:
    return repo_root() / name


def resolve_output_dir(default_subdir: str, requested: str | None = None) -> Path:
    if requested:
        output_dir = Path(requested)
    else:
        output_dir = repo_root() / "outputs" / default_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clean_output_dir(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
