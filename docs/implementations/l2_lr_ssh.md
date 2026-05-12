---
kernelspec:
  language: python
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---
# Swot L2_LR_SSH

This chapter will present the functionalities specific to the Level 2 SWOT Low
Rate products.

```{code-cell}
from fcollections.implementations import (
    # Handler
    NetcdfFilesDatabaseSwotLRL2,
    # Version
    L2Version, Timeliness)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
```

## Data samples

We will illustrate the functionalities using a data sample from [AVISO](https://www.aviso.altimetry.fr/).
You can use the [altimetry-downloader-aviso](https://robin-cls.github.io/aviso/api.html)
tool to run the following script.

```{literalinclude} scripts/pull_data_l2_lr_ssh.py
:language: python
```

## Query overview

Detailed information on the filters and reading arguments can be found in the
``query`` API description

{meth}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL2.query`

The following examples can be used to build complex queries

::::{tab-set}
:::{tab-item} Half Orbits
  - A unique half orbit
    ```python
    fc.query(cycle_number=1, pass_number=1)
    ```
  - One half orbit repeating over all cycles
    ```python
    fc.query(pass_number=1)
    ```
  - A list of half orbits, over multiple cycles
    ```python
    fc.query(cycle_number=slice(1, 4), pass_number=[1, 3])
    ```
  - A specific orbit of the SWOT mission
    ```python
    fc.query(phase='CALVAL')
    ```
:::
:::{tab-item} Periods
  - A time stamp
    ```python
    fc.query(time='2024-01-01')
    ```
  - A period
    ```python
    fc.query(time=('2024-01-01', '2024-03-31'))
    ```
:::
:::{tab-item} Variables subset
  - A subset of variables
    ```python
    fc.query(selected_variables=['time', 'longitude', 'latitude'])
    ```

    :::{note}
      Available variables can explored using {meth}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL2.variables_info`
    :::
:::
:::{tab-item} Area selection
  - Zoom over an area selection
    ```python
    fc.query(bbox=(-10, 5, 35, 40))
    ```
:::
:::{tab-item} Reading options
  - Left swath (Unsmoothed only)
    ```python
    fc.query(left_swath=True, right_swath=False)
    ```
  - Right swath (Unsmoothed only)
    ```python
    fc.query(left_swath=False, right_swath=True)
    ```
  - Stacking over cycles (Basic, Expert, Windwave only)
    ```python
    fc.query(stack='CYCLES')
    ```
  - Stacking over both cycles and passes (Basic, Expert, Windwave only)
    ```python
    fc.query(stack='CYCLES_PASSES')
    ```
:::
:::{tab-item} Subset definitions
  - Use baseline C versions
    ```python
    fc.query(version='P?C?')
    ```
  - Use a specific baseline, and only reprocessed data
    ```python
    fc.query(version='PGD?')
    ```
  - Complete version specification
    ```python
    fc.query(version='PGD0_02')
    ```
  - Choose one dataset
    ```python
    fc.query(subset='Expert')
    ```
:::
::::

## Generic information

Generic information about the files set can be extracted. This available
information is specific to the [mixins](#mixins) used to build
{class}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL2`

::::{tab-set}
:::{tab-item} PeriodMixin
  - Time coverage
    ```python
    fc.time_coverage(subset='Expert', version='P?D?', phase='SCIENCE')
    ```
  - Time holes
    ```python
    fc.time_holes(subset='Expert', version='P?D?', phase='SCIENCE')
    ```
:::
:::{tab-item} HalfOrbitMixin
  - Half orbit range
    ```python
    fc.half_orbit_range(subset='Expert', version='P?D?', phase='SCIENCE')
    ```
  - Cycle range
    ```python
    fc.cycle_range(subset='Expert', version='P?D?', phase='SCIENCE')
    ```
:::
::::


## Stack for temporal analysis

The most prominent functionality is the ability to stack the half orbits when
the grid is fixed (Basic, Expert and WindWave subsets). This allows
to work along the ``cycle_number`` dimension and compute temporal analysis
(mean, standard deviation, ...).

There are currently three modes for stacking the half orbits

- ``NOSTACK``: do not stack the half orbits
- ``CYCLES``: concatenate the half orbits of one cycle along the ``num_lines``
  dimension, and stack the cycles along a new ``cycle_number`` dimension
- ``CYCLES_PASSES``: stack the half orbits along the ``cycle_number`` and
  ``pass_number`` dimensions. Useful for regional analysis where the half orbits
  are cropped and we need an additional dimension to reflect the spatial jump

```{code-cell}
fc = NetcdfFilesDatabaseSwotLRL2("data_l2_lr_ssh")
ds = fc.query(stack='CYCLES', cycle_number=[9, 10, 11], pass_number=10, subset='Basic')
ds.ssha_karin_2.data
```

```{code-cell}
ds = fc.query(stack='CYCLES_PASSES', cycle_number=[9, 10, 11], pass_number=[10, 11], subset='Basic')
ds.ssha_karin_2.data
```

```{note}
Incomplete cycles are completed with invalids
```

## Filter Level-2 version

The Level-2 version is a complex tag composed of a temporality (forward I or
reprocessed G), a baseline (major version A, B, C, D, ...), a minor version
(0, 1, 2, ..) and a product counter (01, 02, ...). The
{class}`fcollections.implementations.L2Version` class can handle the tag
information and filter out non-desired versions. It can be partially initialized
in order to control the granularity of the filter.

```{code-cell}
version = L2Version(temporality=Timeliness.I)
version
```

```{code-cell}
fc.list_files(cycle_number=10, pass_number=10, version=version)
```

```{code-cell}
version = L2Version(baseline='C')
version
```

```{code-cell}
fc.list_files(cycle_number=9, pass_number=10, version=version)
```

## Area selection

It is possible to select data crossing a specific region by providing ``bbox`` parameter to ``query`` or ``list_files`` method.

The bounding box is represented by a tuple of 4 float numbers, such as : (longitude_min, latitude_min, longitude_max, latitude_max).
Its longitude must follow one of the known conventions: [0, 360[ or [-180, 180[.

If bbox's longitude crosses -180/180, data around the crossing and matching the bbox will be selected.
(e.g. for an interval [170, -170] -> both [170, 180[ and [-180, -170] intervals will be used to list/subset data).

To list files corresponding to half orbits crossing the bounding box:

```{code-cell}
bbox = -126, 32, -120, 40
fc.list_files(
    version='PIC?',
    subset='Basic',
    bbox=bbox)
```

To query a subset of Swot LR L2 data crossing the bounding box:

```{note}
Lines of the swath crossing the bounding box will be entirely selected.
```

```{code-cell}
bbox = -126, 32, -120, 40
ds_area = fc.query(subset="Basic", version='P?C?', cycle_number=9, pass_number=11, bbox=bbox)

# Figure
localbox_cartopy = bbox[0] - 1, bbox[2] + 1, bbox[1] - 1, bbox[3] + 1
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent(localbox_cartopy)

plot_kwargs = dict(
    x="longitude",
    y="latitude",
    cmap="Spectral_r",
    vmin=-1.5,
    vmax=1.5,
    cbar_kwargs={"shrink": 0.3},)

# SWOT KaRIn SLA plots
ds_area.ssha_karin_2.plot.pcolormesh(ax=ax, **plot_kwargs)
ax.set_title("SLA KaRIn (uncalibrated) and selection box (in red)")
ax.coastlines()
ax.gridlines(draw_labels=['left', 'bottom'])


# Add the patch to the Axes
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor='r', facecolor='none')
ax.add_patch(rect)
```

## Swath sides in Level-2 Unsmoothed subset

The L2_LR_SSH Unsmoothed dataset files are using netcdf groups to separate the
swath sides. This means we can open one of the two sides. The following figure
illustrates how the ``left_swath`` and ``right_swath`` parameters can be used to
retrieve one or the other side.

```{code-cell}
ds_left = fc.query(subset='Unsmoothed', cycle_number=9, pass_number=10, left_swath=True, right_swath=False,
                  selected_variables=['longitude', 'latitude', 'sig0_karin_2']).compute()
ds_right = fc.query(subset='Unsmoothed', cycle_number=9, pass_number=10, left_swath=False, right_swath=True,
                  selected_variables=['longitude', 'latitude', 'sig0_karin_2']).compute()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

plot_kwargs = dict(
    cmap="Greys_r",
    cbar_kwargs={"shrink": 0.3},
    vmin=5, vmax=60)

# SWOT KaRIn SLA plots
s = slice(45000, 50000, 3)
ds_left.isel(num_lines=s).sig0_karin_2.plot.imshow(ax=ax1, **plot_kwargs)
ds_right.isel(num_lines=s).sig0_karin_2.plot.imshow(ax=ax2, **plot_kwargs)

ax1.set_title("Sigma0 KaRIn Left Swath")
ax2.set_title("Sigma0 KaRIn Right Swath")
fig.tight_layout()
```

```{note}
Combination of both sides is not yet possible. Moreover, keep in mind that
position coordinates may include invalids which can break geo-plots.
```
