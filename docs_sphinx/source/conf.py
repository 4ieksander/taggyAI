# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../taggy'))
sys.path.insert(0, os.path.abspath('../../taggy/utils'))

project = 'Taggy'
copyright = '2025, Aleksander Okrasa'
author = 'Aleksander Okrasa'
release = 'v1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_click',
    'sphinx.ext.intersphinx'
    ]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torchvision': ('https://pytorch.org/vision/stable/', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
    'click': ('https://click.palletsprojects.com/en/8.1.x/', None),
    'rich': ('https://rich.readthedocs.io/en/stable/', None),
    'clip': ('https://github.com/openai/CLIP', None),
}

templates_path = ['_templates']
exclude_patterns = []

language = 'pl'
locale_dirs = ['locales/']
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
