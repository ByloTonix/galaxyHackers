import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath(".."))

project = "galaxy"
copyright = ""
author = "Svetlana Voskresenskaia, Nedezda Fomicheva, Arkady P., Matvei Zekhov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "autoapi.extension",
    # "sphinx.ext.mathjax",
    "sphinx.ext.imgmath"
]

imgmath_latex = 'latex'
# To change the image format (png or svg), use:
imgmath_image_format = 'svg'

autoapi_dirs = ["../galaxy"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# html_theme_options = {
#     "navigation_depth": 4,
#     "collapse_navigation": False,
#     "sticky_navigation": True,
# }
