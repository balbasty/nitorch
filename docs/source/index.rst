.. NITorch documentation master file, created by
   sphinx-quickstart on Fri Feb  4 14:53:43 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/balbasty/nitorch


Welcome to NITorch's documentation!
===================================

.. image:: ../images/nitorch_logo_v0.1.png
   :width: 400

NITorch is a library written in `PyTorch
<http://pytorch.org/>`_ aimed at medical image processing and analysis, with a focus on neuroimaging.

It is a versatile package that implements low-level tools and high-level algorithms 
for deep learning and optimization-based algorithms. It implements low level differentiable functions, 
layers, backbones, optimizers, but also high level algorithms for registration and inverse problems 
as well as a set of command line utilities for manipulating medical images.

Much of the current implementation targets image registration tasks, 
and many differentiable transformation models are implemented 
(classic and Lie-encoded affine matrices, B-spline-encoded deformation fields, 
dense deformation fields, stationary velocity fields, geodesic shooting). 
All of these models can be used as layers in neural networks, 
and we showcase them by reimplementing popular registration networks such as VoxelMorph. 
We also provide generic augmentation layers and easy-to-use and 
highly-parameterized backbone models (U-Nets, ResNets, etc.).

We also provide optimization-based registration tools that can be 
easily applied from the command line, as well as algorithms and 
utilities for solving inverse problems in Magnetic Resonance Imaging.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 4
   :caption: API Documentation

   nitorch.cli
   nitorch.core
   nitorch.io
   nitorch.mesh
   nitorch.nn
   nitorch.plot
   nitorch.spatial
   nitorch.tests
   nitorch.tools
   nitorch.vb

**Authors**

NITorch has been mostly written by Yael Balbastre and Mikael Brudfors, while post-docs 
in John Ashburner's group at the FIL (or *Wellcome Centre for Human Neuroimaging* as it 
is officially known). It is therefore conceptually related to SPM.

All contributions are welcome 
(though no nicely drafted guidelines exist, we'll try to get better).

**License**

NITorch is released under the MIT license.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
