from setuptools import setup

setup(
    name="cosne",
    version="0.0.1",
    py_modules=["main", "htsne_impl"],
    packages=["hyptorch", "pvae"],
    install_requires=[
        "numpy", "pillow", "scikit-learn", "scipy",
        "seaborn", "torch", "torchvision",
        "geoopt @ git+https://github.com/geoopt/geoopt.git",
    ],
    python_requires=">=3.8",
)