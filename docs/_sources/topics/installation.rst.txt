Installation
------------

Follow these steps to set up the project:

**Prerequisites**

Ensure the following tools are installed on your system:

- **Python**: Version 3.10 or higher
- **CUDA** (optional): For GPU acceleration
- **Poetry**: For dependency management (recommended). See `Poetry's website <https://python-poetry.org/>`_.

Cloning the Repository
======================

Clone the private repository using:

.. code-block:: bash

   git clone https://github.com/pelancha/galaxyHackers.git

Setting Up the Environment
==========================

**Using Poetry (Recommended)**

1. Navigate to the project directory:

   .. code-block:: bash

      cd galaxyHackers

2. Activate the Poetry environment:

   .. code-block:: bash

      poetry shell

3. Install dependencies:

   .. code-block:: bash

      poetry install

**Fix for MacOS Sonoma: `pixell` Installation**

If you encounter an error during `pixell` installation, manually install it using:

.. code-block:: bash

   pip install pixell

Then re-run:

.. code-block:: bash

   poetry install

**Using pip (Alternative)**

1. Create and activate a virtual environment:

   .. code-block:: bash

      python3.10 -m venv venv
      source ./venv/bin/activate

2. Install dependencies:

   .. code-block:: bash

      pip install torch torchvision timm torch_optimizer tqdm numpy pandas matplotlib scikit-learn Pillow astropy astroquery pixell dynaconf wget comet_ml
