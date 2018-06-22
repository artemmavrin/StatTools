from setuptools import setup, find_packages
from os import path

# Load long description from README.md
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="stattools",
    version="0.0.2",
    description="Statistical learning and inference library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artemmavrin/StatTools",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib"]
)
