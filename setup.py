from setuptools import setup, find_packages

setup(
    name='stash-api-tools',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'dash',
        'plotly'
    ],
)
