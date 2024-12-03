Development
===========

Contribution Guidelines
------------------------

To ensure a smooth and collaborative development process, follow these guidelines:

1. **Branching Strategy**:
   - Use a descriptive branch name for new features or fixes:
     - Feature branches: ``feature/<feature-name>``
     - Bugfix branches: ``bugfix/<issue-description>``
   - Always base your branch off the ``develop`` branch:

     .. code-block:: bash

        git checkout develop
        git checkout -b feature/<feature-name>

2. **Pull Requests**:
   - Submit pull requests (PRs) to the ``develop`` branch.
   - Provide a clear description of your changes, including:

     - The purpose of the changes.
     - Any relevant issue numbers.
     - Instructions for testing your feature.

3. **Code Reviews**:
   - All PRs must be reviewed and approved by at least one other contributor before merging.
   - Address all review comments before requesting a re-review.

Code Style
----------

Maintain a consistent and clean codebase by adhering to the following standards:

1. **PEP 8 Guidelines**:
   - Follow the Python style guide: `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.

2. **Formatting**:
   - Use ``black`` for automatic code formatting:

     .. code-block:: bash

        black .

3. **Linting**:
   - Run ``flake8`` to check for linting issues:

     .. code-block:: bash

        flake8 .

4. **Docstrings**:
   - Use `Google-style docstrings <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ to document functions, classes, and modules.

   Example:

   .. code-block:: python

      def example_function(param1: int, param2: str) -> str:
          """
          Concatenate an integer and a string.

          Args:
              param1 (int): An integer to convert to a string.
              param2 (str): A string to concatenate.

          Returns:
              str: The concatenated result.
          """
          return f"{param1}{param2}"

Workflow for Adding Features
-----------------------------

1. **Create a Branch**:

   .. code-block:: bash

      git checkout -b feature/<feature-name>

2. **Develop Your Feature**:
   - Write modular, reusable, and well-documented code.

3. **Test Your Changes**:
   - Run your changes locally or in an environment like Colab.

4. **Submit a Pull Request**:
   - Push your branch:

     .. code-block:: bash

        git push origin feature/<feature-name>

   - Open a pull request on GitHub to merge into the ``develop`` branch.

Best Practices
--------------

1. **Commit Messages**:
   - Use meaningful and concise commit messages:

     - Example: ``Add support for ViTL16 model``
     - Example: ``Fix bug in DenseNet training loop``

2. **Modularity**:
   - Write code that is modular and reusable across different parts of the project.

3. **Documentation**:
   - Update or add documentation for all new features in the ``README.md`` or ``docs/`` directory.

4. **Performance**:
   - Profile new features to ensure they do not degrade performance.
