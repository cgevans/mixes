from re import L
from .mixes import *

__all__ = (
    "Q_",
    "FV",
    "FC",
    "S",
    "C",
    "Ref",
    "Mix",
    #    "µM",
    "uM",
    "nM",
    "mM",
    "nL",
    #   "µL",
    "uL",
    "mL",
    "save_mixes",
    "load_mixes",
)

FV = FixedVolume
FC = FixedConcentration
S = Strand
C = Component
Ref = Reference
Mix = Mix

µM = ureg("µM")
uM = ureg("uM")
nM = ureg("nM")
mM = ureg("mM")
nL = ureg("nL")
µL = ureg("µL")
uL = ureg("uL")
mL = ureg("mL")
