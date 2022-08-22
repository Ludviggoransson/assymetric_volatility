from setuptools import setup, find_packages

base_packages = [
    'python-dotenv~=0.17',
    'python-dotenv~=0.17',
    'pydantic',
    'pandas',
    'matplotlib', 
    'seaborn',
]

test_packages = base_packages + [
    'pytest~=7.0',
    'pytest-asyncio~=0.18',
    'pytest-icdiff~=0.5',
]

lint_packages = [
    'darglint~=1.5',
    'flake8~=3.8',
    'flake8-annotations~=2.5',
    'flake8-bandit>=3.0.0',
    'flake8-broken-line~=0.2',
    'flake8-bugbear~=19.8',
    'flake8-comprehensions~=3.3',
    'flake8-debugger~=3.2',
    'flake8-docstrings~=1.5',
    'flake8-eradicate~=0.3',
    'flake8-functions~=0.0',
    'flake8-print~=4.0',
    'flake8_quotes~=2.1',
    'flake8-string-format~=0.2',
    'mypy_extensions~=0.4',
    'naming~=0.6',
    'pydocstyle~=5.0',
]

formatter_packages = [
    'black>=22.3.0',
]

docs_packages = base_packages + [
    'mkdocs-autorefs~=0.1.0',
    'mkdocs-awesome-pages-plugin~=2.5',
    'mkdocs_gen_files~=0.3',
    'mkdocs-material~=6.1',
    'mkdocs-minify-plugin~=0.4',
    'mkdocstrings~=0.15',
]

dev_packages = (
    test_packages
    + lint_packages
    + formatter_packages
    + docs_packages
    + [
        'jupyterlab~=3.0',
        'jupytext~=1.10',
    ]
)

setup(
    name='assymetric_volatility',
    version='0.1',
    package_dir = {"": "src"},
    python_requires='~=3.9',
    extras_require={
        'base': base_packages,
        'test': test_packages,
        'lint': lint_packages,
        'formatter': formatter_packages,
        'docs': docs_packages,
        'dev': dev_packages,
    },
)
