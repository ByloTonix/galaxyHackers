Overview
========

Introduction
------------

**GalaxyHackers** is a state-of-the-art framework designed for astronomical data analysis, with a primary focus on galaxy cluster detection. The project provides tools for building, training, and evaluating machine learning models specifically tailored to the unique challenges of astrophysical datasets.

Key features include:

- **Model Variety**: Supports popular architectures like ResNet, EfficientNet, DenseNet, and custom implementations such as SpinalNet and ViTL16.
- **Automatic Segmentation Visualization**: Easily generate visual outputs for cluster detection results.
- **Comet ML Integration**: Monitor and analyze training experiments seamlessly.
- **Extensibility**: Easily add new datasets, models, or workflows with minimal setup.

.. image:: ../_static/scheme.png
   :alt: GalaxyHackers Overview Image
   :align: center
   :width: 100%


.. Key Features
.. ------------

.. The project provides a robust framework that includes the following features:

.. 1. **Predefined Models**:
..    - Pre-implemented architectures:
..      - **ResNet18**
..      - **EfficientNet**
..      - **DenseNet**
..      - **ViTL16**

.. 2. **Comprehensive Dataset Management**:
..    - Load, preprocess, and split datasets using built-in utilities.
..    - Direct integration with common astronomical formats (e.g., FITS).

.. 3. **Experiment Tracking with Comet ML**:
..    - Automatically log metrics, hyperparameters, and training results.

.. 4. **Dynamic Segmentation Visualizations**:
..    - Create plots to visualize predictions and compare them with ground truth.

.. 5. **Flexible Workflow**:
..    - Easily extend the framework with new datasets, models, or experiments.


Architecture Overview
----------------------

The framework follows a modular architecture, making it easy to integrate new components while keeping the codebase maintainable.

Modules include:

- **Data**:
  - Handles dataset preparation and augmentation.
- **Models**:
  - Predefined and custom machine learning models.
- **Train**:
  - Utilities for model training and validation.
- **Metrics**:
  - Compute evaluation metrics such as accuracy, precision, recall, and segmentation IoU.

---

Next Steps
----------

- Dive into the :doc:`usage` section to learn how to train and evaluate models.
