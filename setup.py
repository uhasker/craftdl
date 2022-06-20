import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="craftdl",
    version="0.1.0",
    description="CraftDL is a library for quickly solving simple & common deep learning tasks",
    packages=["craftdl"],
    install_requires=["torch==1.11.*", "matplotlib==3.5.*", "tqdm==4.64.*"],
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/uhasker/craftdl",
    author="uhasker",
    author_email="uhasker@protonmail.com",
    license="MIT",
)
