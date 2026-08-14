from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='project_team',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    url='https://github.com/gkuling/project_team',
    license='MIT License',
    author='Grey Kuling',
    author_email='gkuling@gmail.com',
    description='A lightweight personal research harness for organizing, '
                'training, and persisting PyTorch and statistical ML '
                'experiments.',
    long_description=Path(__file__).with_name('README.md').read_text(),
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',
        'torchvision',
        'pandas',
        'numpy',
        'tqdm',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'Pillow',
        # The config system subclasses PretrainedConfig; 5.x removed
        # internals this package's save/load path was written against.
        'transformers>=4.30,<5',
        'matplotlib',
    ],
    extras_require={
        'dev': ['pytest', 'pytest-cov'],
    },
    python_requires='>=3.10',
)
