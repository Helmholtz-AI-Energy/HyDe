"""
    Setup file for hyde.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.0.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import setuptools

if __name__ == "__main__":
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    try:
        # setup(use_scm_version={"version_scheme": "no-guess-dev"})
        setuptools.setup(
            name="hyde-images",
            description="Hyperspectral Denoising algorithm toolbox in Python for GPUs",
            version="0.2.0",
            author="Helmholtz-AI-Energy",
            author_email="consultant-helmholtz.ai@kit.edu",
            license="BSD-3-Clause",
            long_description=long_description,
            url="https://github.com/Helmholtz-AI-Energy/HyDe",
            long_description_content_type="text/markdown",
            project_urls={
                "Source": "https://github.com/Helmholtz-AI-Energy/HyDe",
                "Changelog": "https://github.com/Helmholtz-AI-Energy/HyDe/blob/main/CHANGELOG.md",
            },
            classifiers=[
                "Development Status :: 4 - Beta",
                "License :: OSI Approved :: BSD License",
                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.9",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.6",
                "Intended Audience :: Science/Research",
                "Topic :: Scientific/Engineering",
            ],
            package_dir={"": "src"},
            packages=setuptools.find_packages(where="src"),
            python_requires=">=3.6",
            install_requires=[
                "numpy>=1.13.0",
                "torch>=1.8.0",
                # "pytorch-wavelets>=1.3.0",
                "PyWavelets>=1.1.1",
            ],
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise
