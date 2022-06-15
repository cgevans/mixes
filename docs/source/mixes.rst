Mixes
=====

Introductory notes
------------------

- A *Component* is something that goes into a mix, and has a source concentration.  It may be a generic component, a strand with a sequence, or a mix.  It must implement the :class:`AbstractComponent` class.
- An *Action* describes how a component or set of components is to be added to a mix.  It may specify that each component be added to get a target concentration in the mix, for example, or that a fixed volume of each component be added.
- A *Mix* is a collection of Actions, each covering some Components.  It may have a fixed volume, or that may be determined by the components.  It may also have a fixed effective concentration (for use as a component), or that may be determined by a particular component.
- A reference DataFrame can be used to add and check information about components.

Component classes
-----------------

.. currentmodule:: alhambra.mixes

.. autosummary::cd

  Component
  Strand
  Mix

Action classes
--------------

.. autosummary::

  FixedConcentration
  FixedVolume
  MultiFixedConcentration
  MultiFixedVolume
  FixedRatio

Mixes
-----

.. autosummary::

  Mix
  Mix.table
  Mix.all_components

Functions for references
------------------------

.. autosummary::

    Reference
    Reference.update
    Reference.compile

Abstract class details
----------------------

To extend

.. autosummary::

  AbstractComponent
  AbstractAction

Implementation details
----------------------

.. autosummary::

  WellPos
  MixLine
