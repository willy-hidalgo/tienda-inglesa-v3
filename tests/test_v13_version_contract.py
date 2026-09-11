from pathlib import Path
import settings


def test_version_is_major_architecture_reset():
    assert settings.APP_VERSION == "13.3.3"
    assert settings.UPDATE_BLOCK_OPTIONS == (1, 7, 14, 28)
    assert settings.METRIC_HORIZON_DAYS == 28


def test_documentation_is_consolidated_to_seven_markdown_files():
    """Validate only canonical project documentation.

    Operational README files under data/, notebooks/, environments, etc. are not
    part of the consolidated documentation set and must not make this contract fail.
    """
    root = Path(__file__).resolve().parent.parent
    docs = sorted(
        [p.relative_to(root).as_posix() for p in root.glob("*.md")]
        + [p.relative_to(root).as_posix() for p in (root / "docs").glob("*.md")]
    )
    assert docs == [
        "CHANGELOG.md",
        "README.md",
        "docs/ARCHITECTURE.md",
        "docs/DASHBOARD.md",
        "docs/EXPERIMENTS.md",
        "docs/MODEL.md",
        "docs/VALIDATION.md",
    ]
