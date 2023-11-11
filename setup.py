import setuptools
from grayness_index import __version__ as version


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grayness-index-python",
    version=version,
    author="Furkan Kınlı",
    author_email="furkan.kinli@ozyegin.edu.tr",
    description="Python package for Grayness Index",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/birdortyedi/grayness-index-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'glog'],
)