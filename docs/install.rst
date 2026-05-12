Installation
============

How to install
--------------

.. tab-set::

    .. tab-item:: Conda

        You can install or upgrade Files Collection using the `conda install
        <https://docs.conda.io/projects/conda/en/latest/commands/install.html>`_
        command:

        .. code:: bash

            conda install files_collections -c conda-forge

        This will install a minimal set of dependencies required to run
        Files Collections similar to ``python -m pip install files-collections``

        In order to activate the optional features (see. :ref:`optional_dependencies`)
        and get a full installation, additional packages are needed:

        .. code:: bash

            conda install files_collections shapely geopandas 'pyinterp>=2026.4.0' dask numba -c conda-forge

        This will install a set of dependencies similar to
        ``python -m pip install files-collections[geo]``


    .. tab-item:: Pip

        You can install or upgrade Files Collection using the `pip install
        <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_
        command:

        .. code:: bash

            pip install files-collections

        This will install a minimal set of dependencies required to run
        Files Collections similar to ``conda install files_collections -c conda_forge``

        In order to activate the optional features (see. :ref:`optional_dependencies`)
        and get a full installation, ``pyinterp``
        `installation requirements <https://cnes.github.io/pangeo-pyinterp/setup/pip.html>`_
        for building from source must be fullfilled. Then, the full installation
        can be triggered with:

        .. code:: bash

            python -m pip install files-collections[geo]

        ``pyinterp`` installation requirements can be skipped if the package
        has already been `installed via conda <https://cnes.github.io/pangeo-pyinterp/setup/conda.html>`_

.. _optional_dependencies:

Optional dependencies
---------------------

Specific functionality in Files Collections may require optional dependencies.
These optional dependencies and their associated functionalities are listed
below.


+--------------+-----------------------------------------------------------------------------------------+-------------+
| Dependency   | Description                                                                             |   Version   |
+==============+=========================================================================================+=============+
| `dask`_      | Activate ``map()`` method in collections, Enable parallel file reading                  |             |
+--------------+-----------------------------------------------------------------------------------------+-------------+
| `geopandas`_ | Enable geometry intersection, introduce the ``bbox`` argument in the ``query()`` method |             |
+--------------+-----------------------------------------------------------------------------------------+-------------+
| `numba`_     | Enable geometry intersection, introduce the ``bbox`` argument in the ``query()`` method |             |
+--------------+-----------------------------------------------------------------------------------------+-------------+
| `pyinterp`_  | Enable geometry intersection, introduce the ``bbox`` argument in the ``query()`` method | >= 2026.4.0 |
+--------------+-----------------------------------------------------------------------------------------+-------------+
| `shapely`_   | Enable geometry intersection, introduce the ``bbox`` argument in the ``query()`` method |             |
+--------------+-----------------------------------------------------------------------------------------+-------------+

.. _dask: https://www.dask.org/
.. _geopandas: https://geopandas.org/en/stable/
.. _numba: https://numba.pydata.org/
.. _pyinterp: https://cnes.github.io/pangeo-pyinterp/index.html
.. _shapely: https://shapely.readthedocs.io/en/stable/
