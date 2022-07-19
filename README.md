[![Documentation Status](https://readthedocs.org/projects/alhambra-mixes/badge/?version=latest)](https://alhambra-mixes.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://img.shields.io/codecov/c/github/cgevans/mixes)](https://pypi.org/project/alhambra-mixes/)
[![GitHub Workflow
Status](https://img.shields.io/github/workflow/status/cgevans/mixes/Python%20tests)]((https://pypi.org/project/alhambra-mixes/))
[![PyPI](https://img.shields.io/pypi/v/alhambra-mixes)](https://pypi.org/project/alhambra-mixes/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alhambra-mixes)](https://pypi.org/project/alhambra-mixes/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6861213.svg)](https://doi.org/10.5281/zenodo.6861213)


This package, alhambra_mixes, is a separate package containing the `alhambra.mixes`
library from
[alhambra][alhambra]
modified to be more compatible with Python < 3.10.  Continued development on
mixes will take place here, and alhambra will be made to depend on this.  **The
name may change soon to something more unique.**

The mixes package is a Python library to systematically, efficiently, and safely
design recipes for mixes of many components, intended primarily for DNA
computation experiments.  The library recursively tracks individual components
through layers of intermediate mixes, performs checks to ensure those layers of
mixes are possible, and allows verification that final samples will contain the
correct components at the correct concentrations. Incorporating reference
information from files such as DNA synthesis order details, the library
generates recipes for straightforward pipetting, even in mixes with many
components at different concentrations spread across several plates.

Our [our documentation][mixref] is in the process of being written; we also have [a tutorial notebook][tutorial] (WIP).


[alhambra]: https://github.com/DNA-and-Natural-Algorithms-Group/alhambra
[mixref]: https://alhambra-mixes.readthedocs.io/
[tutorial]: https://github.com/cgevans/mixes/blob/main/tutorial.ipynb
