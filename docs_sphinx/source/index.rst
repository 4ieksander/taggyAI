.. Taggy documentation master file, created by
   sphinx-quickstart on Sat Jan 18 17:56:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Taggy documentation
===================

Dependencies
============

Your project requires the following Python packages to function:

- **torch**: Used for loading models from `torch.hub`.
- **torchvision**: Provides pre-trained models.
- **click**: Command-line interface support.
- **rich**: For enhanced logging.
- **numpy** and **pillow**: For loading and handling images.
- **opencv-python**: For simple image analysis, such as sharpness or face detection.

Additional tools for documentation and translations:

- **Sphinx** and its extensions: `sphinx-click`, `sphinx-intl`, `myst-parser`.
- **Furo**: Modern documentation theme.
- **pypandoc**: For exporting documentation to DOCX.

Special models and utilities:

- **CLIP**: Main model for image analysis (from OpenAI).
- Optional: `tensorflow` and `toml` for additional model support.


.. toctree::
   :maxdepth: 5
   :caption: Spis treści:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`