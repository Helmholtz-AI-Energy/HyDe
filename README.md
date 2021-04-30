# HyDe - Hyperspectral Denoising

Hyperspectral Denoising algorithm toolbox in Python

Project Status
--------------

[![license: BSD-3](https://img.shields.io/badge/License-BSD3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Description
-----------

Image denoising is the task of recovering the true unknown image from a degraded observed image. It plays an important role in a variety of applications, for example in remote sensing imaging systems in lithological mapping. Hyperspectral Denoising is a Python toolbox aiming to provide, as the name suggests, denoising algorithms for hyperspectral image data. In particular, we provide:

* A wide variety of hyperspectral denoising algorithms (see Features for details)
* GPU acceleration for all algorithms
* An inuitive pythonic API design
* PyTorch compatibility

Features
--------

* Automatic Hyperspectral Image Restoration Using Sparse and Low-Rank Modeling ([source](https://ieeexplore.ieee.org/document/8098642))
* Hyperspectral Mixed Gaussian and Sparse Noise Reduction ([source](https://ieeexplore.ieee.org/document/8760540))

Requirements
------------

Hyperspectral denoises makes heavy use of PyTorch

License
-------

Hyperspectral Denoising is distributed under the BSD-3 license, see our [LICENSE](LICENSE) file.

Acknowledgements
----------------

*This work is supported by the [Helmholtz Association Initiative and
Networking Fund](https://www.helmholtz.de/en/about_us/the_association/initiating_and_networking/)
under the Helmholtz AI platform grant.*

---

<div align="center">
    <a href="https://www.helmholtz.ai/"><img src="logos/helmholtzai_logo.jpg" height="45px" hspace="3%" vspace="20px"></a><a href="http://www.kit.edu/english/index.php"><img src="logos/kit_logo.svg" height="45px" hspace="3%" vspace="20px"></a><a href="https://www.hzdr.de/db/Cms?pOid=32948&pNid=2423"><img src="logos/hif_logo.png" height="45px" hspace="3%" vspace="20px"></a><a href="https://www.helmholtz.de/en/"><img src="logos/helmholtz_logo.svg" height="45px" hspace="3%" vspace="20px"></a>
</div>

