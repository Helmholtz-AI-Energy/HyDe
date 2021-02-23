from setuptools import setup, find_packages
import codecs


with codecs.open("README.md", "r", "utf-8") as handle:
    long_description = handle.read()

__version__ = None  # appeases flake, assignment in exec() below
with open("./hyde/version.py") as handle:
   exec(handle.read())

setup(
    name="HyDe",
    packages=find_packages(exclude=(,)),
    data_files=["README.md", "LICENSE"],
    version=__version__,
    description="Hyperspectral Denoising algorithm toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Helmholtz AI Local Energy",
    author_email="consultant-helmholtz.ai@kit.edu",
    url="https://github.com/Helmholtz-AI-Energy/HyperspectralDenoising",
    keywords=["hyperspectral", "denoising", "remote sensing", "gpu"],
    python_requires="~=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD-3 License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=["numpy>=1.15.0", "torch>=1.7.0"]
)
