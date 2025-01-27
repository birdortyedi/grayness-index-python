from setuptools import setup, find_packages

from grayness_index import __version__ as version

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="grayness-index-python",
    version=version,
    author="Furkan Kınlı",
    author_email="furkan.kinli@ozyegin.edu.tr",
    description="Python package for Grayness Index",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/birdortyedi/grayness-index-python",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "torch>=1.8.0",
        "kornia>=0.6.0",
        "scipy>=1.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    keywords=[
        "computational photography",
        "color constancy",
        "illuminant estimation",
        "computer vision"
    ]
)