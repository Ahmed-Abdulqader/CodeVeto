# Contributing to CodeVeto

Thank you for your interest in contributing to **CodeVeto**! 

CodeVeto is built for developers who demand explicit control, security, and intentionality. We are not building for "vibe coders" who rely on auto-pilot; we are building for engineers who want to accelerate their workflow without compromising agency, security, or codebase integrity. As such, we expect contributions to be deliberate, well-tested, and strictly aligned with our core philosophy.

Please read this document carefully before submitting your first contribution.

---

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Development Workflow](#development-workflow)
- [Code Style & Quality](#code-style--quality)
- [Security Considerations](#security-considerations)
- [Pull Request Guidelines](#pull-request-guidelines)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Python >= 3.14** (as specified in `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)**: The fast Python package and project manager.

---

## Development Workflow

Follow these steps to set up your environment and contribute effectively. **Deviating from this workflow may result in your Pull Request being closed.**

### 1. Find and Assign an Issue
- Browse the [open issues](https://github.com/Ahmed-Abdulqader/CodeVeto/issues).
- Comment on the issue you want to work on to **assign it to yourself**. 
- *Note:* Please wait for a maintainer's acknowledgment before starting work to avoid duplicated effort. **Do not submit PRs for unassigned issues.**

### 2. Fork and Clone
- Fork the repository to your GitHub account.
- Clone your fork locally:
  ```bash
  git clone https://github.com/YOUR_USERNAME/CodeVeto.git
  cd CodeVeto
  ```

### 3. Set Up the Environment
- Start by syncing the project dependencies using `uv`. This will create a virtual environment and install all required packages (including development tools like `ruff` and `pytest`):
  ```bash
  uv sync
  ```

### 4. Create a Branch
- Create a descriptive branch for your work. Use a prefix like `feature/`, `fix/`, or `docs/`:
  ```bash
  git checkout -b feature/your-descriptive-branch-name
  ```

### 5. Develop and Test
- Make your changes intentionally. 
- If your contribution adds new functionality or fixes a bug, **write or update tests** in the `tests/` directory.
- Run the test suite to ensure nothing is broken:
  ```bash
  uv run pytest
  ```

---

## Code Style & Quality

We use **[Ruff](https://docs.astral.sh/ruff/)** for both formatting and linting to ensure a consistent, high-quality, and secure codebase. 

**Before you commit any changes, you MUST run the following commands:**

1. **Format the code:**
   ```bash
   uv run ruff format
   ```
2. **Check for linting errors:**
   ```bash
   uv run ruff check
   ```
   *(If `ruff check` reports any issues, fix them before proceeding. Safe issues can be auto-fixed using `uv run ruff check --fix`)*

> [!IMPORTANT]
> Pull Requests that fail `ruff` checks, do not follow the project's formatting standards, or lack adequate tests will be requested for changes or closed without review.

---

## Security Considerations

CodeVeto’s core value proposition is **security and explicit developer control**. 
- Any changes affecting file access, command execution, context reading, or the "Veto" permission layer will undergo strict scrutiny.
- Ensure your code does not introduce silent failures, implicit permissions, background telemetry, or unauthorized file reads.
- For reporting security vulnerabilities, please refer to our [SECURITY.md](./SECURITY.md).

---

## Pull Request Guidelines

When you are ready to submit your work:

1. **Commit your changes** with clear, descriptive, and conventional commit messages (e.g., `feat: add explicit read permission prompt` or `fix: prevent silent file overwrite`).
2. **Push your branch** to your forked repository.
3. **Open a Pull Request** against the `main` branch of the original repository.
4. **Reference the Issue**: In your PR description, explicitly reference the issue you were assigned (e.g., `Closes #12` or `Fixes #42`).
5. **Describe your changes**: Briefly explain what you did, why you did it, and how a reviewer can test it.
6. **Ensure CI passes**: All GitHub Actions checks (including `ruff` formatting/linting and `pytest`) must pass before a maintainer will review your PR.

---

## 🙏 Acknowledgments

Thank you for helping us build a truly developer-owned AI tooling ecosystem. Your contributions make CodeVeto better for everyone who values control, transparency, and security in their development workflow.
