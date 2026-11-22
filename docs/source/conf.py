# Configuration file for the Sphinx documentation of PyShockFlow.
# Build locally from the docs folder with:  make html  (or  make.bat html  on Windows)

import os
import sys

# make the packages in src/ importable for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

project = "PyShockFlow"
author = "Matthias Van Mechelen"
copyright = "2026, Matthias Van Mechelen, Francesco Neri and Matteo Pini"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

# docstrings in the code follow the NumPy convention
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "none"

exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": True,
}

numfig = True
