# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/config

import os
import sys

# Add the project root to the path so Sphinx can import pycf
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -------------------------------------------------------

project = "PyCF"
copyright = "2024, Sebastian Horvath"
author = "Sebastian Horvath"

# The short X.Y version
try:
    from pycf.__version__ import __version__

    version = __version__
    # For release, use just major.minor
    release = __version__
except ImportError:
    version = "dev"
    release = "dev"

# -- General configuration -------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (in the form of modules or `sphinx` submodules).
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.autosummary",  # Compact API summary tables
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # Link to other Sphinx docs
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.mathjax",  # Support LaTeX math
    "sphinx_rtd_theme",  # ReadTheDocs theme (if installed)
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that shouldn't be included
# when building the documentation.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "legacy/**"]

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and app pages.
try:
    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "default"

# Theme options are theme-specific and used by html_theme_path.
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
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
html_use_opensearch = ""

# If nonempty, this is the file name suffix for generated HTML files.
html_file_suffix = ".html"

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of (source start file,
# target name, title, author, documentclass [howto*, report, manual, or own
# class]).
latex_documents = [
    ("index", "PyCF.tex", "PyCF Documentation", "Sebastian Horvath", "manual"),
]

# -- Options for manual page output ---------------------------------------------

# One entry per manual page. List of (source start file, name, description,
# authors, manual section).
man_pages = [("index", "pycf", "PyCF Documentation", ["Sebastian Horvath"], 1)]

# -- Options for Texinfo output ------------------------------------------------

texinfo_documents = [
    (
        "index",
        "PyCF",
        "PyCF Documentation",
        "Sebastian Horvath",
        "PyCF",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output ---------------------------------------------------

epub_title = "PyCF"
epub_author = "Sebastian Horvath"
epub_publisher = "Sebastian Horvath"
epub_copyright = "2024, Sebastian Horvath"

# -- Extension configuration ---------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "show-inheritance": True,
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
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "string": "str",
    "integer": "int",
    "boolean": "bool",
    "callable": "typing.Callable",
    "dictionary": "dict",
    "dictionaries": "dict",
    "sequence": "typing.Sequence",
    "ndarray": "numpy.ndarray",
    "np.ndarray": "numpy.ndarray",
    "np.array": "numpy.ndarray",
    "numpy array": "numpy.ndarray",
    "array": "numpy.ndarray",
    "iterable": "Iterable",
    # Internal short names used in docstrings → fully qualified targets
    "EData": "pycf.cfl_util.EData",
    "Hamiltonian": "pycf.cfl.Hamiltonian",
    "SpinHamiltonian": "pycf.cfl.SpinHamiltonian",
    "Tensor": "pycf.cfl.Tensor",
    "EFit": "pycf.cfl.EFit",
    "MHFit": "pycf.cfl.MHFit",
    "ExData": "pycf.cfl.ExData",
    "Spectrum": "pycf.inten.Spectrum",
    "ImportSLJM": "pycf.import_sljm.ImportSLJM",
    "EFitRunner": "pycf.pyfit.EFitRunner",
    "MHFitRunner": "pycf.pyfit.MHFitRunner",
    "ESHFitRunner": "pycf.pyfit.ESHFitRunner",
    "MESHFitRunner": "pycf.pyfit.MESHFitRunner",
}
napoleon_attr_annotations = True

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

# -- Options for todo extension ------------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# Nitpick (sphinx-build -n) suppressions for conventional NumPy-doc types and
# external symbols that have no Python class to link to. The strict CI build
# uses -W but not -n, so these only matter for local nitpicky-mode runs;
# silencing them keeps that mode useful for spotting real broken xrefs.
nitpick_ignore_regex = [
    # Conventional NumPy-doc type strings (not actual classes)
    (r"py:class", r"optional"),
    (r"py:class", r"array_?[- ]?like"),
    (r"py:class", r"half[- ]?int(eger)?"),
    (r"py:class", r"list/tuple"),
    (r"py:class", r"list with .*"),
    (r"py:class", r"object with .*"),
    (r"py:class", r"^N$"),
    (r"py:class", r"^len$"),
    # Anything containing whitespace, parentheses, brackets, commas, pipes,
    # backticks, or quote marks is not a real class name — usually a stray
    # fragment from a compound type spec that Napoleon couldn't fully parse.
    (r"py:class", r".*[\s(){}\[\]|,`\"].*"),
    # External EMP tooling executables, not Python symbols
    (r"py:.*", r"^(cfit|vtrans|inten|spectrum)$"),
    # napoleon internal artifact when resolving type aliases
    (r"py:class", r"^TypeAliasForwardRef$"),
    # matplotlib private path; SpectrumAxes inherits these docstrings verbatim
    (r"py:class", r"matplotlib\.axes\._axes\.Axes"),
    # Internal short references that appear in :class:/:func:/:meth:/:attr:
    # roles inside docstrings (any reftype). These resolve correctly in the
    # generated HTML via autodoc's local scope but Sphinx can't validate them
    # without fully-qualified paths.
    (r"py:.*", r"^(EData|EFit|MHFit|Hamiltonian|SpinHamiltonian|Tensor|ExData|Spectrum)$"),
    (r"py:.*", r"^(EData\.(DTYPE|to_str)|EFit\.(get_edata|last_jacobian)|MHFit\.get_edata)$"),
    (r"py:.*", r"^(bgs_coeff_array|fd_jacobian|gen_edata_summary|dipole_str|lstsq|datetime)$"),
    (r"py:.*", r"^cfl_util\.gen_e_summary$"),
    (
        r"py:.*",
        r"^cfl\.(EFit|Tensor|Hamiltonian|SpinHamiltonian|SpinHamiltonian\.calc_param|CFLMin)$",
    ),
    (r"py:.*", r"^import_sljm\.ImportSLJM$"),
    (r"py:.*", r"^spinhamiltonian\.sh_lsq_func$"),
    (r"py:.*", r"^pycf\.cfl\._(temporary_x|x_to_coeff_dict)$"),
    (r"py:.*", r"^scipy\.optimize\.least_squares$"),
    (r"py:.*", r"^(last_result|2-tuple)$"),
    (
        r"py:.*",
        (
            r"^pycf\.(cfl_util\.EData|inten\.(dipole_str|group_transitions|Spectrum)"
            r"|pyfit\.(EFitRunner|MHFitRunner|ESHFitRunner|MESHFitRunner))$"
        ),
    ),
]


# Register matplotlib's custom :mpltype: role as a no-op so that inherited
# docstrings from matplotlib.axes.Axes (used via SpectrumAxes) don't emit
# "Unknown interpreted text role" warnings. The role is purely cosmetic in
# matplotlib's own docs; we render its content as plain text.
def setup(app):
    from docutils import nodes
    from docutils.parsers.rst import roles

    def mpltype_role(name, rawtext, text, lineno, inliner, options=None, content=None):
        return [nodes.Text(text)], []

    roles.register_local_role("mpltype", mpltype_role)
