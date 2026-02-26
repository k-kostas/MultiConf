import os
import sys

project = 'MultiConf'
copyright = '2026, Kostas Katsios'
author = 'Kostas Katsios'
release = '0.1.0'

extensions = [  'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx_copybutton',
                'myst_parser',
                'sphinx.ext.viewcode'

]

templates_path = ['_templates']
exclude_patterns = []

# html_theme = 'sphinx_rtd_theme'
html_theme = "pydata_sphinx_theme"
# html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('../../src'))

autodoc_member_order = 'bysource'

copybutton_prompt_text = r">>> ?|\.\.\. ?"
copybutton_prompt_is_regexp = True

html_title = 'MultiConf v. 0.1.0'

html_sidebars = {
    "**": []
}