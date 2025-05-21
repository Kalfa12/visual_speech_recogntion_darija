import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Adds the project root to Python path

project = 'visual_speech_recogntion_darija'
copyright = '2025, Your Name/Organization' # Update with your info
author = 'Your Name/Organization' # Update with your info

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'nbsphinx',  # For Jupyter notebook integration
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

# (Optional) nbsphinx configuration
nbsphinx_execute = 'never' # 'always' or 'auto' if you want notebooks to be run during doc build
nbsphinx_allow_errors = True

# Intersphinx mapping (example for Python and PyTorch)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}
