# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/config

import os
import sys

# Add the project root to the path so Sphinx can import pycf
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -------------------------------------------------------

project = 'PyCF'
copyright = '2024, Sebastian Horvath'
author = 'Sebastian Horvath'

# The short X.Y version
try:
    from pycf.__version__ import __version__
    version = __version__
    # For release, use just major.minor
    release = __version__
except ImportError:
    version = 'dev'
    release = 'dev'

# -- General configuration -------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (in the form of modules or `sphinx` submodules).
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',          # Support for Google/NumPy style docstrings
    'sphinx.ext.intersphinx',       # Link to other Sphinx docs
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.mathjax',           # Support LaTeX math
    'sphinx_rtd_theme',             # ReadTheDocs theme (if installed)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that shouldn't be included
# when building the documentation.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and app pages.
try:
    html_theme = 'sphinx_rtd_theme'
except ImportError:
    html_theme = 'default'

# Theme options are theme-specific and used by html_theme_path.
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = []

# The name for this set of documents.
html_title = f"PyCF {release} Documentation"

# A shorter title for the navigation bar.
html_short_title = "PyCF"

# The name of an image file (within the static path) to use as favicon of the
# docs. This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "(C) Copyright ..." text is shown in the HTML footer.
html_show_copyright = True

# If true, an OpenSearch description file is output, and all these functions
# have a `rel="search"` link to it.
html_use_opensearch = ''

# If nonempty, this is the file name suffix for generated HTML files.
html_file_suffix = '.html'

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of (source start file,
# target name, title, author, documentclass [howto*, report, manual, or own
# class]).
latex_documents = [
    ('index', 'PyCF.tex', u'PyCF Documentation',
     u'Sebastian Horvath', 'manual'),
]

# -- Options for manual page output ---------------------------------------------

# One entry per manual page. List of (source start file, name, description,
# authors, manual section).
man_pages = [
    ('index', 'pycf', u'PyCF Documentation', [u'Sebastian Horvath'], 1)
]

# -- Options for Texinfo output ------------------------------------------------

texinfo_documents = [
    ('index', 'PyCF', u'PyCF Documentation',
     u'Sebastian Horvath', 'PyCF', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output ---------------------------------------------------

epub_title = u'PyCF'
epub_author = u'Sebastian Horvath'
epub_publisher = u'Sebastian Horvath'
epub_copyright = u'2024, Sebastian Horvath'

# -- Extension configuration ---------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

autodoc_mock_imports = []

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

# -- Options for todo extension ------------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
