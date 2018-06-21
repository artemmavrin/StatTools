from setuptools import setup

setup(
    name="stattools",
    version="0.0.2",
    license="MIT",
    packages=["stattools", "stattools.glm", "stattools.sde", "stattools.utils",
              "stattools.generic", "stattools.datasets", "stattools.ensemble",
              "stattools.survival", "stattools.smoothing",
              "stattools.resampling", "stattools.optimization",
              "stattools.preprocessing", "stattools.visualization",
              "stattools.regularization"],
    url="https://github.com/artemmavrin/StatTools",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    description="Statistical learning and inference library",
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib"]
)