# RATER Challenge Solution

This repository contains the modeling code, experiments, and supporting materials for my RATER Challenge project.

## Final solution (4th place)

Start with the **[final solution](docs/RaterSolution_IoannisMeintanis.pdf)** for the project's approach and results.

## Project structure

- [`src/`](src/) - installable RATER code and legacy experiment modules.
- [`scripts/`](scripts/) - training, inference, and post-processing scripts.
- [`det_utils/`](det_utils/) - separate GPU-cluster code used with Determined AI; kept independent of the local RATER package.
- [`nbs/`](nbs/) - historical experiment and analysis notebooks.
- [`solutions/`](solutions/) - copied reference implementations and configurations.
- [`docs/`](docs/) - final solution, project reports, and solution notes.
- [`Files/Resources/`](Files/Resources/) - supporting project resources.
- [`pyproject.toml`](pyproject.toml) - package build and test configuration.
- [`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt) - environment and package dependency specifications.

## Getting started

Install the dependencies for your execution environment, then install the project and development test tools in editable mode:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest
```

The Determined AI cluster workflow under [`det_utils/`](det_utils/) is maintained separately. Local competition data, checkpoints, and generated artifacts are not tracked; see [`.gitignore`](.gitignore).
