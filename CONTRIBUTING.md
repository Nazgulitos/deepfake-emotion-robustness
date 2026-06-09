# Contributing

Thank you for helping improve this research repository.

## Scope
This project is intended to preserve the original experiment behavior. Contributions should focus on documentation, reproducibility, packaging, tests, and release hygiene unless a change to the research pipeline is explicitly requested.

## Before You Open a Pull Request
1. Create a clean branch for your work.
2. Install the project dependencies with `uv sync`.
3. Run the relevant commands for the code you changed.
4. Run the test suite when possible:

```bash
make test
```

5. Update documentation if your change affects setup, configuration, or outputs.

## Style Notes
- Keep changes focused and avoid unrelated refactors.
- Preserve existing APIs, file names, and experiment behavior.
- Prefer small, reviewable commits.
- Add docstrings and comments only where they clarify non-obvious behavior.
- Use `uv.lock` as the canonical lockfile when checking dependency state.

## Pull Request Checklist
- [ ] The change does not alter model behavior unless explicitly intended.
- [ ] Tests relevant to the change were run.
- [ ] README or supporting docs were updated if needed.
- [ ] New files and outputs are explained clearly in the PR description.