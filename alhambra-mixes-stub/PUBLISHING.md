# Publishing the alhambra-mixes stub package

This stub package eases the transition from the old name (`alhambra-mixes`) to the new name (`riverine`).

## How to publish

1. Make sure riverine is already published to PyPI
2. Update the version number in `pyproject.toml` if needed
3. Build the package:
   ```bash
   cd alhambra-mixes-stub
   uv build
   ```

4. Publish to PyPI:
   ```bash
   uv publish
   # or
   twine upload dist/*
   ```

## What this package does

When users install `alhambra-mixes`, they will:
1. Get `riverine` installed as a dependency
2. See a deprecation warning when they `import alhambra`
3. Have all riverine functionality available through the `alhambra` namespace

This allows existing code to continue working while encouraging migration to the new name.

