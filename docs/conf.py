import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "galaxyHackers"
copyright = "FreeMakers"
author = "Svetlana Voskresenskaia, Nedezda Fomicheva, Arkady P., Matvei Zekhov"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "autoapi.extension",
    "sphinx.ext.imgmath",
]

imgmath_latex = "latex"
imgmath_image_format = "png"

autoapi_dirs = ["../galaxy"]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "black",
        "color-brand-content": "#404040",
    },
    "dark_css_variables": {
        "color-brand-primary": "white",
        "color-brand-content": "#d0d0d0",
    },
}

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]


html_show_sphinx = False
