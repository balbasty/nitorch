# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'NITorch'
# copyright = '2021, Graziella'
author = 'Yael Balbastre, Mikael Brudfors'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    # 'nitorch._C',
    'nitorch.cli',
    'nitorch.core',
    'nitorch.io',
    'nitorch.mesh',
    'nitorch.nn',
    'nitorch.plot',
    'nitorch.spatial',
    'nitorch.tests',
    'nitorch.tools',
    'nitorch.vb'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'