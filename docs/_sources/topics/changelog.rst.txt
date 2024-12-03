Changelog
=========

v0.1 (May 7, 2024)
------------------

Implemented Features
********************

**main.py**

- **Dataloader Creation**: Utilizes ``data.py`` to create a dataloader and splits it into train and validation loaders.
- **Model Selection**: Offers a list of easily modifiable models for training.
- **Optimizer Selection**: Provides a choice of optimizers from a predefined list.
- **Training Script**: Executes training with pre-selected models, number of epochs, optimizer, and its settings such as learning rate (``lr``) and momentum (``mm``).
- **Training Progress**: Displays training progress and statistics during execution.
- **Results Saving**: Saves the results of training for further analysis.

**train.py**

- **train()**: Displays progress and statistics during training. Saves best model weights, checkpoints, and results.
- **validate()**: Validates the model and provides statistics.
- **continue_training()**: Allows resuming training from checkpoints.

**data.py**

- Queries GAIA Star Catalog asynchronously (``read_gaia``).
- Reads data from **ACT_DR5** and **MaDCoWS** catalogs, pre-downloading if necessary.
- Creates positive and negative samples for training:
  - ``createNegativeClassDR5``, ``create_data_dr5``.
- Includes data transformations: resizing, rotation, reflection, and normalization.

**segmentation.py**

- Implements image dataset class and segmentation map creation:
  - ``create_samples``, ``formSegmentationMaps``, ``printSegMaps``, ``printBigSegMap``.
- Predicts probabilities for images using ``predict_folder`` and ``predict_tests``.

**legacy_for_img.py**

- Downloads image cutouts using multithreading (``grab_cutouts``).
- Supports **VLASS** and **unWISE** image downloads.

**metrics.py**

- Includes plotting functions:
  - ROC Curve (``plot_roc_curve``), Precision-Recall Curve (``plot_precision_recall``).
- ``modelPerformance``: Calculates and displays metrics such as accuracy, precision, recall, and F1-score.

Known Issues
************

- Not all optimizers are functional.
- Incorrect loss and accuracy calculations with Adam optimizer.
- **segmentation.py**: Paths for loaded weights need fixing.
- **data.py**: Ensure data folder existence.
- **main.py**:
  - Models are loaded simultaneously; optimize memory usage.
  - Some models need fixes (e.g., ``ViTL16``).

Directory Structure
*******************

.. code-block:: yaml

    .
    ├── data
    │   ├── DATA
    │   │   ├── test_dr5
    │   │   ├── test_madcows
    │   │   ├── train
    │   │   └── val
    ├── models
    ├── notebooks
    ├── results
    ├── screenshots
    ├── state_dict
    └── trained_models

Performance
***********

- **Initial startup**: ~1h 9m 43s (Internet speed-dependent).
- **Subsequent runs**: ~6m 55s.

Usage
*****

Run the script with the following command:

.. code-block:: bash

    python3 main.py --model MODEL_NAME --epoch NUM_EPOCH --optimizer OPTIMIZER --lr LR --mm MOMENTUM

Supported models:

- ``ResNet18``, ``AlexNet_VGG``, ``SpinalNet_VGG``, ``SpinalNet_ResNet``.

Supported optimizers:

- ``SGD``, ``Adam``, ``RMSprop``, ``AdamW``, ``Adadelta``, ``DiffGrad``.

Example output for AlexNet training with SGD optimizer:

.. code-block:: text

    Epoch 1/10. Training AlexNet with SGD optimizer: acc=0.683, loss=0.598
    Validation Loss: 0.0075, Validation Accuracy: 0.7926
    ...
    Epoch 10/10. Training AlexNet with SGD optimizer: acc=0.884, loss=0.286
    Validation Loss: 0.0057, Validation Accuracy: 0.8445

Bugs
****

- Not all optimizers work.
- Loss and accuracy calculations incorrect with Adam optimizer.
- Additional feedback from users is welcome.
