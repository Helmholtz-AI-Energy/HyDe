# HyDe - Hyperspectral Denoising

Hyperspectral Denoising algorithm toolbox in Python

## Project Status

[![license: BSD-3](https://img.shields.io/badge/License-BSD3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Description

Image denoising is the task of recovering the true unknown image from a degraded observed image. It plays an important role in a variety of applications, for example in remote sensing imaging systems in lithological mapping. Hyperspectral Denoising is a Python toolbox aiming to provide, as the name suggests, denoising algorithms for hyperspectral image data. In particular, we provide:

* A wide variety of hyperspectral denoising algorithms (see Features for details)
* GPU acceleration for all algorithms
* An inuitive pythonic API design
* PyTorch compatibility

# Features

* Automatic Hyperspectral Image Restoration Using Sparse and Low-Rank Modeling ([HyRes](https://ieeexplore.ieee.org/document/8098642))
* Hyperspectral Mixed Gaussian and Sparse Noise Reduction ([HyMiNoR](https://ieeexplore.ieee.org/document/8760540))

## Future Features

* [BM3D](https://www.cs.tut.fi/~foi/GCF-BM3D/)
* [FastHyDe](https://arxiv.org/pdf/2103.06842.pdf)
* [L1HyMixDe](https://ieeexplore.ieee.org/document/9040508) or [repo](https://github.com/LinaZhuang/L1HyMixDe)
* [WSRRR](https://ieeexplore.ieee.org/document/6736073)
* [OTVCA](https://ieeexplore.ieee.org/document/7530874)
* [FORPDN](https://ieeexplore.ieee.org/document/6570741)

Requirements
------------

Hyperspectral denoises makes heavy use of PyTorch


## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `hyde` with the help of [conda]:
   ```
   python -m venv hyde_venv
   ```
2. activate the new environment with:
   ```
   source hyde_venv/bin/activate
   ```
3. Install requirements
   ```
   pip install -r requirements.txt -e .
   ```

Optional and needed only once after `git clone`:

4. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

5. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build system configuration. Do not change!
├── references              <- Data dictionaries, manuals, and all other materials.
├── scripts                 <- Analysis and production scripts which import the
│                              actual Python package, e.g. train_model.py.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `pip install -e .` to install for development or
|                              or create a distribution with `tox -e build`.
├── src
│   └── hyde                <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## License

Hyperspectral Denoising is distributed under the BSD-3 license, see our [LICENSE](LICENSE.txt) file.

## Acknowledgements

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under the Helmholtz AI platform grant.*

---

<div align="center">
    <a href="https://www.helmholtz.ai/"><img src="logos/helmholtzai_logo.jpg" height="45px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="logos/kit_logo.svg" height="45px" hspace="3%" vspace="20px"></a><a href="https://www.hzdr.de/db/Cms?pOid=32948&pNid=2423"><img src="logos/hif_logo.png" height="45px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="logos/helmholtz_logo.svg" height="45px" hspace="3%" vspace="20px"></a>
</div>

---

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.0.1 and the [dsproject extension] 0.6.1.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
