"""Tests for the CHANGELOG release-notes extractor used by the release workflow."""

from textwrap import dedent

from scripts.extract_release_notes import extract_release_notes


def test_extracts_named_version_section() -> None:
    changelog = dedent(
        """\
        # Changelog

        ## [0.2.0] - 2026-06-25

        ### Added
        - A new thing

        ### Fixed
        - A broken thing

        ## Entries

        ### 2026-06-17
        - Some dated entry
        """
    )
    notes = extract_release_notes(changelog, "0.2.0")
    assert notes is not None
    assert notes.startswith("## [0.2.0] - 2026-06-25")
    assert "A new thing" in notes
    assert "A broken thing" in notes
    # Dated entries below "## Entries" must not leak into the release notes.
    assert "Some dated entry" not in notes


def test_section_terminates_at_next_version_header() -> None:
    changelog = dedent(
        """\
        # Changelog

        ## [0.3.0] - 2026-07-01

        ### Added
        - Newer thing

        ## [0.2.0] - 2026-06-25

        ### Added
        - Older thing
        """
    )
    notes = extract_release_notes(changelog, "0.3.0")
    assert notes is not None
    assert "Newer thing" in notes
    # Must stop before the next version section.
    assert "Older thing" not in notes
    assert "0.2.0" not in notes


def test_ignores_header_inside_code_fence() -> None:
    """A version header shown as documentation inside a fence is not a real section."""
    changelog = dedent(
        """\
        # Changelog

        ## Format

        Release sections use a version header:

        ```markdown
        ## [0.2.0] - 2026-06-25
        ```

        ## Entries

        ### 2026-06-17
        - Some dated entry
        """
    )
    assert extract_release_notes(changelog, "0.2.0") is None


def test_returns_none_for_missing_version() -> None:
    changelog = dedent(
        """\
        # Changelog

        ## [0.1.0] - 2025-01-01

        Baseline release.

        ## Entries
        """
    )
    assert extract_release_notes(changelog, "9.9.9") is None


def test_fenced_block_inside_section_is_preserved() -> None:
    """A code fence within a real version section should be kept verbatim."""
    changelog = dedent(
        """\
        # Changelog

        ## [0.2.0] - 2026-06-25

        ### Added
        - Example usage:

        ```bash
        pip install agentfarm==0.2.0
        ```

        ## Entries
        """
    )
    notes = extract_release_notes(changelog, "0.2.0")
    assert notes is not None
    assert "pip install agentfarm==0.2.0" in notes
    assert "```bash" in notes


def test_section_terminates_at_entries_boundary() -> None:
    changelog = dedent(
        """\
        # Changelog

        ## [0.2.0] - 2026-06-25

        ### Added
        - Kept line

        ## Entries

        ### 2026-06-17
        - Dropped line
        """
    )
    notes = extract_release_notes(changelog, "0.2.0")
    assert notes is not None
    assert "Kept line" in notes
    assert "Dropped line" not in notes
