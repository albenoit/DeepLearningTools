# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'DeepLearningTools'
copyright = '2023, Alexandre Benoit'
author = 'Alexandre Benoit'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration', 
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'nord'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = '_static/logo_listic.png'
html_theme_options = {
    "light_css_variables": {
        "font-stack": "Roboto, sans-serif",
        "font-stack--monospace": "Open Sans, monospace",
        # "color-brand-primary": "",
        # "color-background-secondary": "#24038F"
    },
    "announcement":
        "<em>Deeplearningtools is currently in development.</em>",
    "source_repository": "https://github.com/albenoit/DeepLearningTools",
    "source_edit_link": "https://github.com/albenoit/DeepLearningTools",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_logo": "main_light.png",
    "dark_logo": "main_dark.png",
}