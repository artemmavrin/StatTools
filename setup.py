from os import path

from setuptools import setup, find_packages

# Load long description from README.md
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="stattools",
    version="0.0.3",
    description="Statistical learning and inference library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artemmavrin/StatTools",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    license="MIT",
    packages=sorted(find_packages(exclude=("*.test",))),
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib"]
)
