"""
setup.py — legacy shim for tools that don't support pyproject.toml yet.
All canonical metadata lives in pyproject.toml.
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
