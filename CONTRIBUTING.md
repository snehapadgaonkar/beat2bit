# Contributing

<a href="#bug-report">Bug Report</a> •
<a href="#request-feature">Request Feature</a> •
<a href="#coding-convention">Coding Convention</a> •
<a href="#git-commit">Git Commit</a> •
<a href="#pull-request">Pull Request</a>

Thank you for taking the time to contribute to Beat2Bit! This document explains how to report bugs, request features, and submit pull requests.

---

## Bug Report

If you find a bug, please open a [GitHub Issue](https://github.com/snehapadgaonkar/beat2bit/issues) with the following information:

- **Description** — a clear summary of the bug
- **Steps to reproduce** — the exact sequence of actions that triggers the bug
- **Expected behaviour** — what you expected to happen
- **Actual behaviour** — what actually happened
- **Environment** — Python version, TensorFlow version, OS, Node.js version (if frontend)
- **Logs / screenshots** — any relevant error output

---

## Request Feature

Open a [GitHub Issue](https://github.com/snehapadgaonkar/beat2bit/issues) labelled `enhancement` and include:

- **Motivation** — why is this feature useful to the project?
- **Proposed solution** — how you think it could be implemented
- **Alternatives considered** — other approaches you thought of

---

## Coding Convention

### Python (ML pipeline)

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Type-hint all public functions
- Docstrings on all public functions and classes
- Keep training scripts runnable end-to-end with `python scripts/train_optimised.py`
- Never commit model weights (`.keras`, `.tflite`, `.h5`) — they are git-ignored
- Never commit raw or processed data arrays (`.npy`) — they are git-ignored

### TypeScript / React (frontend)

- Use functional components with explicit prop types
- Tailwind CSS for all styling — no inline styles
- All new components must work at `sm`, `md`, and `lg` breakpoints (responsive)
- Keep components in `frontend/components/`; page-level logic in `frontend/app/`
- Run `npm run lint` before committing

---

## Git Commit

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short summary>
```

| Type | When to use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Refactor without feature/fix |
| `test` | Adding or fixing tests |
| `chore` | Build process, dependencies |

**Examples:**

```
feat(training): add post-pruning recalibration on imbalanced data
fix(frontend): correct FLOPs formatter for sub-10 MFLOP models
docs(readme): update results table with real MIT-BIH numbers
```

---

## Pull Request

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the coding conventions above.

3. **Add tests** for any new ML code in `tests/`. Run the suite:
   ```bash
   pytest tests/ -v
   ```

4. **Run the frontend linter:**
   ```bash
   cd frontend && npm run lint
   ```

5. **Commit** using the Conventional Commits format above.

6. **Push** your branch and open a Pull Request against `main`.

7. In the PR description, include:
   - What the PR does
   - Any benchmark numbers that changed (before/after if applicable)
   - Screenshots for UI changes

PRs that add new benchmark results should include the full summary table output from `scripts/train_optimised.py`.
