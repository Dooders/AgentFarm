# Contributing to AgentFarm

Thank you for your interest in contributing to AgentFarm! This document provides guidelines and instructions to help you contribute effectively.

## Ways to Contribute

- **Bug Reports**: Report bugs or issues you encounter
- **Feature Requests**: Suggest new features or improvements
- **Documentation**: Help improve or expand documentation
- **Code Contributions**: Submit code improvements or new features

## Branching Model

**`dev` is the default integration branch — all contributions target `dev`, not `main`.**

- **`dev`**: The active development branch. Base your work on `dev` and open all pull requests against `dev`.
- **`main`**: The stable, release branch. It is updated only by maintainers (typically by merging `dev`). Please do **not** open pull requests against `main`.

When you fork or clone, make sure your feature branch starts from the latest `dev`:

```bash
git checkout dev
git pull upstream dev   # or `origin dev` if working directly on the repo
git checkout -b my-feature-branch
```

## Getting Started

1. **Fork the repository** to your GitHub account
2. **Clone your fork** to your local machine
3. **Add the upstream remote**: `git remote add upstream https://github.com/Dooders/AgentFarm.git`
4. **Create and activate a virtual environment** (recommended): `python -m venv venv` then `source venv/bin/activate`
5. **Install dependencies and the package in editable mode**: `pip install -r requirements.txt` and `pip install -e .`
6. **Create a new branch from `dev`** for your contribution (see [Branching Model](#branching-model))
7. **Make your changes** following our coding standards
8. **Test your changes** (for example `pytest` from the repository root)
9. **Commit your changes** with clear, descriptive commit messages
10. **Push to your fork** and submit a pull request **targeting the `dev` branch**

## Pull Request Process

1. **Target the `dev` branch** — pull requests against `main` will be asked to retarget
2. Ensure your branch is up to date with the latest `dev` before opening or updating your PR
3. Ensure your code follows the project's coding standards
4. Update documentation as necessary
5. Include a clear description of the changes in your pull request
6. Link any relevant issues in your pull request description
7. Be responsive to feedback and be willing to make changes if requested

## When will my PR ship?

All contributor PRs merge into **`dev`**. Releases cut from `dev` into **`main`** on a milestone or time-based cadence tracked via [GitHub Milestones](https://github.com/Dooders/AgentFarm/milestones).

- Your PR ships in the **next release** after it merges to `dev`, assuming that release's milestone is not already frozen.
- Maintainers may label PRs `release-blocker` or `breaking-change` when triaging.
- For the full maintainer release process, see [docs/RELEASE.md](docs/RELEASE.md).

## Coding Standards

- Follow PEP 8 style guidelines for Python code (line length 120, as configured for Ruff/Pylint)
- Lint your changes with `ruff check .` (and optionally `pylint farm`) before opening a PR
- Run the test suite with `pytest` from the repository root; add or update tests under `tests/` for behavior changes
- Write meaningful comments and docstrings
- Keep functions and methods small and focused on a single task
- Favor SOLID, DRY, KISS, and composition over inheritance when adding or refactoring code

## Reporting Bugs

When reporting bugs, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Any relevant logs, screenshots, or error messages
- Information about your environment (OS, Python version, etc.)

## Communication

- For quick questions, use the project's [GitHub Discussions](https://github.com/Dooders/AgentFarm/discussions)
- For major changes, open an issue first to discuss what you would like to change

## Code of Conduct

- Be respectful and inclusive in your language and actions
- Be constructive in providing feedback
- Focus on the issue, not the person
- Be open to learning and improving
- We're all at different places in our knowledge and skills, so be patient and supportive

Thank you for contributing to AgentFarm and helping to make it better! 