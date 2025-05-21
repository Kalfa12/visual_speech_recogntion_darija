CharMap Class
=============

The ``CharMap`` class is responsible for mapping characters to integer indices and vice-versa. This is essential for converting text to a model-understandable format and back for Connectionist Temporal Classification (CTC) loss.

Key Features
------------

* Handles the BLANK token required for CTC Loss.
* Initializes with an alphabet string (e.g., the Arabic alphabet used in the notebook).
* Provides methods like ``text_to_indices`` and ``indices_to_text``.

.. automodule:: your_project_module.CharMap  # If CharMap is in a .py file
   :members:
