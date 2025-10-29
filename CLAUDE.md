# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`riverine` is a Python library for designing recipes for mixes of many components, primarily for DNA computation experiments. The library recursively tracks individual components through layers of intermediate mixes, performs validation checks, and generates recipes for pipetting.

## Development Commands

### Testing
- `hatch test` - Run tests (primary command)
- `hatch test --cover` - Run tests with coverage
- `hatch test --python 3.11` - Run tests on specific Python version
- `hatch test --randomize` - Run tests in random order

### Code Quality
- `black src/ tests/` - Format code
- `ruff check src/ tests/` - Lint code  
- `mypy src/riverine` - Type checking

### Documentation
- `cd docs && make html` - Build documentation

### Build and Install
- `hatch build` - Build package
- `pip install -e .` - Install in development mode
- `pip install -e .[echo]` - Install with ECHO liquid handler support (requires Python ≥ 3.10)

## Architecture

### Core Classes
- **Component/Strand** (`components.py`): Individual components like DNA strands
- **AbstractAction** (`actions.py`): Base class for mix actions (FixedVolume, FixedConcentration, etc.)
- **Mix** (`mixes.py`): Container for components and actions that generates mixing recipes
- **Experiment** (`experiments.py`): High-level container that manages multiple mixes and tracks volumes
- **Reference** (`references.py`): Handles loading component data from files (CSV, Excel)

### Key Patterns
- Uses `attrs` for class definitions with validation
- Pint library for unit handling (volumes in μL, concentrations in nM/μM)
- Polars/Pandas DataFrames for tabular data processing
- Optional ECHO liquid handler integration via `kithairon` library
- Extensive use of type hints and validation

### File Structure
- `src/riverine/` - Main package code
- `tests/` - Test files with data samples
- `docs/` - Sphinx documentation
- `tutorial.ipynb` - Interactive tutorial

### Units System
- Volume units: μL (`uL`), with `Q_()` for quantity creation
- Concentration units: nM, μM (`nM`, `uM`)
- Uses Pint's `ureg` unit registry

### Testing Strategy
- Tests are in `tests/` directory
- Uses hatch for test execution with pytest backend
- Coverage reporting with `--cover` flag
- Multi-version testing via hatch matrix
- Test data files in `tests/data/`
- Supports both with and without ECHO functionality

### Common Patterns
- Mix creation typically involves: creating components → defining actions → creating Mix → calling `make()`
- Reference loading from CSV/Excel files for component definitions
- Volume tracking and validation throughout the mixing process
- Table formatting for human-readable output using `tabulate`