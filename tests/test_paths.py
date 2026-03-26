from pathlib import Path

from helpers.paths import clean_output_dir


def test_clean_output_dir_preserves_gitkeep(tmp_path: Path):
    preserved = tmp_path / ".gitkeep"
    preserved.write_text("", encoding="utf-8")
    stale_file = tmp_path / "old_plot.png"
    stale_file.write_text("stale", encoding="utf-8")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "artifact.txt").write_text("nested", encoding="utf-8")

    clean_output_dir(tmp_path)

    assert preserved.exists()
    assert not stale_file.exists()
    assert not nested_dir.exists()
