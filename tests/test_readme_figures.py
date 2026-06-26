from __future__ import annotations

from scripts import make_readme_figures


def test_readme_figure_sources_exist():
    assert (make_readme_figures.TABLE_DIR / "table_gate_similarity_mirror_vs_nonmirror.csv").exists()
    assert make_readme_figures.FIG_DIR.exists()
