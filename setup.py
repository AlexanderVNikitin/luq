import os
from setuptools import setup
from setuptools import find_packages


# Function to read version from __version__.py
def get_version():
    with open(os.path.join(os.path.dirname(__file__), "luq/version.py")) as f:
        exec(f.read())
    return locals()["__version__"]


name = "luq"
version = get_version()

keywords = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "uncertainty quantification",
    "large language models",
]

author = "Alexander Nikitin"
url = "https://github.com/AlexanderVNikitin/luq"

license = "MIT"

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
]

install_requires = [
    "requests>=2.25.1",
    "numpy>=1.21.0",
]


def read_file(filename: str) -> str:
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


readme_text = read_file("README.md")


setup(
    name="luq",
    version=version,
    description="Framework for uncertainty quantification of LLMs",
    author=author,
    author_email="",
    maintainer=author,
    maintainer_email="",
    url=url,
    download_url="",
    keywords=keywords,
    long_description=readme_text,
    long_description_content_type="text/markdown",
    license=license,
    install_requires=["huggingface", "loguru", "aisuite[all]", "scipy", "tqdm", "datasets"],
    package_data={"luq": ["README.md"]},
    packages=find_packages(),
)
