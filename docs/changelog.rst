Release notes
=============

2.0.0 (2026-05-XX)
------------------

Important
.........

The project has been moved to the `CNES <https://github.com/CNES/fcollections>`_
Github repository.

A new method ``filter_values`` is available for all ``NetcdfFilesDatabase``
implementations. This method helps extracting the possible values of a query's
filter. It is fast when scanning the folders (layouts must be enabled), but
slower when scanning the files is necessary (a ``PerformanceWarning`` is issued
in that case).

.. code:: python

  >> from fcollections.implementations import NetcdfFilesDatabaseGriddedSLA
  >> fc = NetcdfFilesDatabaseGriddedSLA(...)
  >> fc.filter_values('version')
  {'0_3', '1_0', '2_0_1', '3_0'}

The following dependencies have new constraints:
  - Pandas: ``>=3``
  - Pyinterp: ``>=2026.4.0``

Folder-specific filters can now be given to filter a query. This new behavior
breaks the assumption that all layouts share the same filters, which has the
following consequences:

- Setting a folder-specific filter that is present in one layout, with an
  underconstrained query will raise  a ``LayoutMismatchError``. Adding more
  filters to constrain the scan to the part of the file system matching the
  layout will fix the query.

  .. code:: python

    from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3
    fc = NetcdfFilesDatabaseSwotLRL3(...)

    # Will raise an error, the scan will explore v1 and v2 which have no concept
    # of 'temporality'
    fc.query(temporality='REPROC')

    # Add constraint to fix the query
    fc.query(temporality='REPROC', version='3.0')

- Folder-specific filters with the layouts disabled will be ignored, and a
  ``UserWarning`` will be emitted

Breaking Changes
................

The ``IPredicate`` interface has been refactored into the more explicit
``IFilterBuilder`` interface. This interface handles both complex predicates and
filters' converter through the ``build_predicate`` and ``build_filter`` methods.

Previously, if a file did not match the file name convention, it was ignored. It
now raises a ``LayoutMismatchError``.

``PeriodMixin`` now expects to work on a single homogeneous subset of data. In
case the filters given to the methods are not sufficient to extract an
homogeneous dataset, an error will be raised.

Details
.......

- feat: allow folder-specific filters `PR#9 <https://github.com/CNES/fcollections/pull/9>`_
- chore: migration to pyinterp 2026.4.0 `PR#8 <https://github.com/CNES/fcollections/pull/8>`_
- Half orbit mixin `PR#7 <https://github.com/CNES/fcollections/pull/7>`_
- perf: subset unmixing prior to listing `PR#6 <https://github.com/CNES/fcollections/pull/6>`_
- feat!: Add phase filter argument `PR#5 <https://github.com/CNES/fcollections/pull/5>`_
- feat!: filter values from Layout folders `PR#4 <https://github.com/CNES/fcollections/pull/4>`_
- refactor: adapt code to work with pandas `PR#1 <https://github.com/CNES/fcollections/pull/1>`_

Contributors
............

- Robin Chevrier
- Anne-Sophie Tonneau

1.0.0 (2026-02-04)
------------------

A bit of refactoring has been done in this version to improve the maintainability
and versatility of the code. The following interfaces are impacted:

- ``fcollections.core.FileDiscoverer`` and ``fcollections.core.FileSystemIterable`` have been merged into
  ``fcollections.core.FileSystemMetadataCollector``
- ``fcollections.core.FilesDatabase`` is configured using multiple layouts. Layouts must now declare both folder and filename conventions.
  This effectively makes ``fcollections.core.CompositeLayout`` obsolete (it has been removed)
- A new parameter ``enable_layouts`` has been introduced in ``fcollections.core.FilesDatabase`` to simplify layout feature disabling in
  cast of a mismatch
- A new parameter ``follow_symlinks`` has been introduced in ``fcollections.core.FilesDatabase`` to enable symlinks. (The feature is
  disabled by default)

Details
.......

- fix: keep dataset sorted for bbox selection on unsmoothed dataset `#20 <https://github.com/robin-cls/fcollections/pull/20>`_
- refactor!: Use one or multiple Layouts instead of FileNameConvention in FilesDatabase `#14 <https://github.com/robin-cls/fcollections/pull/14>`_
- feat!: refactor and add layouts for CMEMS implementations `#19 <https://github.com/robin-cls/fcollections/pull/19>`_
- feat!: allow listing with/without layouts `#17 <https://github.com/robin-cls/fcollections/pull/17>`_
- feat!: use INode for FileSystemMetadataCollector `#21 <https://github.com/robin-cls/fcollections/pull/21>`_
- feat: follow symbolic links for posix-compliant local file systems `#18 <https://github.com/robin-cls/fcollections/pull/18>`_
- chore!: remove obsolete code from core API `#16 <https://github.com/robin-cls/fcollections/pull/16>`_
- chore: switch license file name to US spelling `#22 <https://github.com/robin-cls/fcollections/pull/22>`_
- doc: create README and documentation landing page `#15 <https://github.com/robin-cls/fcollections/pull/15>`_

Contributors
............

- Robin Chevrier
- Anne-Sophie Tonneau

0.1.3 (2025-12-15)
------------------

Important
.........

KaRIn geometries URL are now up. Geographical selection should work as intended
for SWOT implementations

Details
.......

- fix: relax area selector longitudes convention `#8 <https://github.com/robin-cls/fcollections/pull/8>`_
- fix: use AVISO Karin Geometries `#4 <https://github.com/robin-cls/fcollections/pull/4>`_
- refactor: redispatch implementations code per product `#6 <https://github.com/robin-cls/fcollections/pull/6>`_
- test: process warnings and enforce full coverage `#11 <https://github.com/robin-cls/fcollections/pull/11>`_
- doc: getting started section conversion to myst_nb `#10 <https://github.com/robin-cls/fcollections/pull/10>`_
- doc: switch implementations documentation to myst_nb `#9 <https://github.com/robin-cls/fcollections/pull/9>`_
- doc: fix signatures and docstrings for sphinx documentation `#7 <https://github.com/robin-cls/fcollections/pull/7>`_
- doc: switch Parameters section to Attributes docstrings in dataclasses `#5 <https://github.com/robin-cls/fcollections/pull/5>`_
- doc: installation procedure and release note `#3 <https://github.com/robin-cls/fcollections/pull/3>`_


0.1.2 (2025-12-09)
------------------

First release. The KaRIn geometries URL are not set up and are expected to break
the geographical selection: avoid using the ``bbox`` argument in the
``query()`` method.
