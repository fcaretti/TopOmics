# topomics

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/fcaretti/topomics/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/topomics

Single-cell multiomics topic modeling

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install topomics:

<!--
1) Install the latest release of `topomics` from [PyPI][]:

```bash
pip install topomics
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/fcaretti/topomics.git@main
```

Or even better, download, activate your environment, enter the repo and then run
```bash
pip install -e .'[spatial, amortized, test, docs]'
```
or a subset of these if you prefer a lighter installation.

## Suggest changes

For now, simply add points or subpoints to ROADMAP.md.

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out.
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/fcaretti/topomics/issues
[tests]: https://github.com/fcaretti/topomics/actions/workflows/test.yaml
[documentation]: https://topomics.readthedocs.io
[changelog]: https://topomics.readthedocs.io/en/latest/changelog.html
[api documentation]: https://topomics.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/topomics
