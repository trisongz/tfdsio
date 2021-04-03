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

__version_info__ = (0, 0, 1)
version = ".".join(map(str, __version_info__))
binary_names = [package_name]

with open(os.path.join(root, 'README.md'), 'rb') as readme:
    long_description = readme.read().decode('utf-8')


def _post_install():
    import subprocess
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'git+https://github.com/trisongz/PyFunctional'], stdout=subprocess.DEVNULL)

class install(_install):
    def run(self):
        _install.run(self)
        self.execute(_post_install, (self.install_lib,), msg="Installating Requirements")

setup(
    name=package_name,
    version=version,
    description="tfdsio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Tri Songz',
    author_email='ts@growthengineai.com',
    url='http://github.com/trisongz/tfdsio',
    python_requires='>3.6',
    install_requires=[
        "tensorflow>=2.3.0",
        "tensorflow_datasets>=4.2.0"
    ],
    packages=packages,
    extras_require={
        'transformers': ['transformers', 'sentencepiece'],
        't5': ['t5'],
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