import os
import sys
import os.path

from setuptools.command.install import install as _install
from setuptools import setup, find_packages

root = os.path.abspath(os.path.dirname(__file__))
package_name = "tfdsio"
packages = find_packages(
    include=[package_name, "{}.*".format(package_name)]
)

__version_info__ = (0, 0, 7)
version = ".".join(map(str, __version_info__))
binary_names = [package_name]

with open(os.path.join(root, 'README.md'), 'rb') as readme:
    long_description = readme.read().decode('utf-8')

setup(
    name=package_name,
    version=version,
    description="tfdsio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tri Songz',
    author_email='ts@growthengineai.com',
    url='http://github.com/trisongz/tfdsio',
    python_requires='>=3.6',
    install_requires=[
        "tensorflow>=1.15.0",
        "tensorflow_datasets>=4.2.0",
        "file-io>=0.1.0",
        "seqio",
        "t5",
        "sentencepiece",
    ],
    packages=packages,
    extras_require={
        'transformers': ['transformers', 'sentencepiece'],
        't5': ['t5', 'seqio'],
    },
    entry_points={},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
    ],
    data_files=[],
    include_package_data=True
)