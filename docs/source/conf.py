# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# Path relative from docs/source to the project root where python files are
sys.path.insert(0, os.path.abspath('../../src/python'))
# Add the project root to the path for notebooks
sys.path.insert(0, os.path.abspath('../..'))

# Import nbsphinx for notebook support
import nbsphinx

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NSphere'
copyright = '2025, Kris Sigurdson' # Adjust year/name if needed
author = 'Kris Sigurdson'
# release = '0.1' # Optional: Add your project release version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add your extensions
extensions = [
    'sphinx.ext.autodoc',      # Include documentation from docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'breathe',                 # Bridge between Doxygen and Sphinx
    'nbsphinx',                # Support for Jupyter notebooks
    'sphinx.ext.mathjax',      # Enable MathJax for LaTeX rendering
]

# Configure MathJax to load from a CDN (Content Delivery Network) for LaTeX rendering
# Using MathJax version 3
mathjax_path = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'

# Disable smart quotes to prevent '--' from becoming em-dashes incorrectly
smartquotes = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The examples directory is configured in nbsphinx_path below

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] # Keep this if you have custom static files
html_output_path = '../'

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_warnings = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Breathe Configuration ---------------------------------------------------
breathe_projects = {
   "NSphere": "../doxygen/xml/" # Path to Doxygen XML output relative to conf.py
}
breathe_default_project = "NSphere"

# -- nbsphinx Configuration ---------------------------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during the build
nbsphinx_allow_errors = True  # Display cells with errors
nbsphinx_kernel_name = 'python3'  # Use Python 3 kernel

# Include notebooks outside of source directory
import sphinx
from pathlib import Path
notebook_src_dir = Path(os.path.abspath('../../examples'))

# Set up paths for notebooks - both the original location and the temporary notebooks directory
notebook_original_dir = str(notebook_src_dir)
notebook_temp_dir = os.path.abspath('./examples/notebooks')

# Create the notebooks directory if it doesn't exist yet
os.makedirs(notebook_temp_dir, exist_ok=True)

nbsphinx_allow_errors = True
nbsphinx_path = [notebook_original_dir, notebook_temp_dir]

# Define custom resolver for notebook references
def setup(app):
    # Set up debug logging
    from sphinx.util import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Extra nbsphinx paths: {str(notebook_src_dir)}")

# Template for notebook header/footer - simple informational note
nbsphinx_prolog = """
.. note::

   This example notebook demonstrates NSphere visualization techniques.
   The original notebook can be found in the examples directory of the NSphere project.
"""
