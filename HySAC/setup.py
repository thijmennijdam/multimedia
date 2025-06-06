from setuptools import setup, find_packages

# Read the long description from your README, if desired
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Load requirements
try:
    with open("requirements.txt") as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    install_requires = []

setup(
    name="hysac",
    version="0.0.1",
    description="HySAC: Hyperbolic Safety-Aware Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimagelab/HySAC",
    author="Tobia Poppi, Tejaswi Kasarla, Pascal Mettes, Lorenzo Baraldi, Rita Cucchiara",
    author_email="aimagelab@ing.unimore.it",  # adjust if needed
    license="CC-BY-NC-4.0",
    packages=find_packages(exclude=("tests", "docs",)),
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points={
        # Uncomment and configure if you provide CLI entry points
        # "console_scripts": [
        #     "hysac=hysac.cli:main",
        # ],
    },
)
