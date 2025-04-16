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
        'plotly',
        'scikit-learn',
        'configparser'
    ],
    entry_points={
        'console_scripts': [
            'stash-tools=main:main',
        ],
    },
    author='Your Name',
    description='Tools for analyzing and managing Stash data',
    long_description=open('README.md').read() if open('README.md').read() else '',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/stash-api-tools',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    default_command='stats'
)
