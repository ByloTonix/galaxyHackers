Usage
=====

The `main.py` script provides a flexible way to train models, track experiments, and visualize results.

Preparing the Dataset
---------------------

Unpack the dataset into the `./storage/` directory:

.. code-block:: bash

   unzip data.zip -d ./storage/

Training Models
---------------

Overview of `main.py`
~~~~~~~~~~~~~~~~~~~~~

The `main.py` script allows you to:

1. Train selected models or all supported architectures.
2. Choose optimizers, learning rates, and momentum values.
3. Track training and validation metrics using **Comet ML**.
4. Generate segmentation plots for visualization.

Script Arguments
~~~~~~~~~~~~~~~~

+-------------------+-----------------------------------------------------------+-------------------------+
| Argument          | Description                                               | Default Value           |
+===================+===========================================================+=========================+
| ``--models``      | List of models to train (e.g., `ResNet18`, `EfficientNet`) | All models              |
+-------------------+-----------------------------------------------------------+-------------------------+
| ``--epochs``      | Number of epochs to train                                 | ``5``                   |
+-------------------+-----------------------------------------------------------+-------------------------+
| ``--lr``          | Learning rate for the optimizer                           | ``0.0001``              |
+-------------------+-----------------------------------------------------------+-------------------------+
| ``--mm``          | Momentum for optimizers like ``SGD``                      | ``0.9``                 |
+-------------------+-----------------------------------------------------------+-------------------------+
| ``--optimizer``   | Optimizer to use (e.g., ``Adam``, ``SGD``)                | ``Adam``                |
+-------------------+-----------------------------------------------------------+-------------------------+

Supported Models and Optimizers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Models**:

- ``Baseline``
- ``ResNet18``
- ``EfficientNet``
- ``DenseNet``
- ``SpinalNet_ResNet``
- ``SpinalNet_VGG``
- ``ViTL16``
- ``AlexNet_VGG``

**Optimizers**:

- ``SGD``, ``Adam``, ``NAdam``, ``RAdam``, ``AdamW``, ``RMSprop``, ``DiffGrad``

Example Commands
~~~~~~~~~~~~~~~~

**Train All Models (Default):**

.. code-block:: bash

   python3 main.py

**Train Specific Models:**

.. code-block:: bash

   python3 main.py --models ResNet18 EfficientNet --epochs 10 --lr 0.001

**Train with Custom Optimizer:**

.. code-block:: bash

   python3 main.py --models DenseNet --epochs 20 --lr 0.0005 --optimizer SGD --mm 0.85

Output and Logs
---------------

- **Metrics**: Training and validation metrics are logged to Comet ML and saved as CSV files (``*_train_metrics.csv``, ``*_val_metrics.csv``).
- **Segmentation Plots**: Generated for each model and stored locally.
- **Model Performance**: Combines metrics across models for analysis.
