Welcome to Galaxy Hackers' documentation!
==========================================

.. image:: _static/logo.png
   :alt: Galaxy Hackers Logo
   :width: 200px
   :align: center

The **GalaxyHackers** project focuses on training and evaluating deep learning models for cluster detection in astronomical datasets. It supports multiple model architectures, optimizers, and experiment tracking with **Comet ML**.

Features
--------

- Supports a variety of **deep learning models**: ResNet18, EfficientNet, DenseNet, ViTL16, and more.
- Integrates with **Comet ML** for tracking training experiments.
- Includes **automatic segmentation plot generation** after model training.
- Provides flexibility in choosing optimizers, learning rate schedulers, and hyperparameters.

Introduction
============

Galaxy Hackers is a project focused on detecting and analyzing galaxies and clusters.
It integrates data from various catalogs such as ACT-MCMF and DR5, processes FITS files into images, and uses YOLO for model training and detection.

Key Features:
- Fetch and process astronomical data.
- Annotate and visualize galaxy images dynamically.
- Train and evaluate YOLO models for galaxy and cluster detection.


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   topics/installation
   topics/usage
   topics/changelog
   guides/development
   modules/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Links
-----

- **GitHub Repository**: https://github.com/pelancha/galaxyHackers
