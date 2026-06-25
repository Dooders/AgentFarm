from setuptools import find_packages, setup

from farm._version import __version__

setup(
    name="agentfarm",
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
)
