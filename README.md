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

For documentation, at the moment, see [our API reference][mixref],
or [our tutorial notebook][tutorial] (WIP).


[alhambra]: https://github.com/DNA-and-Natural-Algorithms-Group/alhambra
[mixref]: https://alhambra-mixes.readthedocs.io/
[tutorial]: https://github.com/cgevans/mixes/blob/main/tutorial.ipynb
