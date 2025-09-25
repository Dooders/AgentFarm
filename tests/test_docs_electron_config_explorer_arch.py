import os


DOC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'electron', 'config_explorer_architecture.md'))
ROOT_README = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'README.md'))
DOCS_README = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'README.md'))


def read_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_architecture_doc_exists():
    assert os.path.exists(DOC_PATH), f"Missing architecture doc at {DOC_PATH}"


def test_architecture_doc_has_required_sections():
    content = read_file(DOC_PATH)
    required_headers = [
        '## High-Level Architecture',
        '## Module Responsibilities (Renderer)',
        '## Module Responsibilities (Main)',
        '## IPC Contracts',
        '## State Management (Renderer)',
        '## Electron Process Concerns',
        '## Migration Plan (Phased)',
        '## Risks & Mitigations',
        '## Acceptance Criteria Mapping',
    ]
    for header in required_headers:
        assert header in content, f"Expected section header not found: {header}"


def test_root_readme_links_architecture_doc():
    content = read_file(ROOT_README)
    assert 'docs/electron/config_explorer_architecture.md' in content, 'Root README missing link to architecture doc'


def test_docs_readme_links_architecture_doc():
    content = read_file(DOCS_README)
    assert 'electron/config_explorer_architecture.md' in content, 'Docs README missing link to architecture doc'

