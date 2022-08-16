from re import L

from .actions import *
from .components import *
from .mixes import *
from .references import *
from .units import *

__all__ = (
    "Q_",
    "FV",
    "FC",
    "EC",
    "TC",
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
    "ureg",
)

FV = FixedVolume
FC = FixedConcentration
EC = EqualConcentration
TC = ToConcentration
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
