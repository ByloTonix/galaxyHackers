Welcome to Galaxy Hackers' documentation!
-----------------------------------------

.. image:: _static/logo.png
   :alt: Galaxy Hackers Logo
   :width: 200px
   :align: center

Galaxy Hackers is a project focused on detecting and analyzing galaxies and clusters.
It integrates data from various catalogs such as ACT-MCMF and DR5, processes FITS files into images, and uses YOLO for model training and detection.

Key Features:
-------------

- Fetch and process astronomical data.
- Annotate and visualize galaxy images dynamically.
- Train and evaluate YOLO models for galaxy and cluster detection.

Features
--------

- Supports a variety of **deep learning models**: ResNet18, EfficientNet, DenseNet, ViTL16, and more.
- Integrates with **Comet ML** for tracking training experiments.
- Includes **automatic segmentation plot generation** after model training.
- Provides flexibility in choosing optimizers, learning rate schedulers, and hyperparameters.


.. toctree::
   :maxdepth: 2
   :caption: galaxyHackers Overview

   topics/overview
   topics/license

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   topics/installation
   topics/changelog
   topics/usage
   guides/development

.. toctree::
   :maxdepth: 2
   :caption: Extra

   modules/index

GitHub Repository: https://github.com/pelancha/galaxyHackers
