---
kernelspec:
  language: python
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---
# Swot L3_LR_SSH

This chapter will present the functionalities specific to the Level 3 SWOT Low
Rate products.

```{code-cell}
from fcollections.implementations import NetcdfFilesDatabaseSwotLRL3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cartopy.crs as ccrs
```

## Data samples

We will illustrate the functionalities using a data sample from [AVISO](https://www.aviso.altimetry.fr/).
You can use the [altimetry-downloader-aviso](https://robin-cls.github.io/aviso/api.html)
tool to run the following script.

```{literalinclude} scripts/pull_data_l3_lr_ssh.py
:language: python
```

## Query overview

Detailed information on the filters and reading arguments can be found in the
``query`` API description

{meth}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL3.query`

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
      Available variables can explored using {meth}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL3.variables_info`
    :::
:::
:::{tab-item} Area selection
  - Zoom over an area selection
    ```python
    fc.query(bbox=(-10, 5, 35, 40))
    ```
:::
:::{tab-item} Reading options
  - Nadir and KaRIn data (Basic, Expert only)
    ```python
    fc.query(nadir=True)
    ```
  - Nadir data only (Basic, Expert only)
    ```python
    fc.query(swath=False, nadir=True)
    ```
  - Stacking over cycles (Basic, Expert, Technical only)
    ```python
    fc.query(stack='CYCLES')
    ```
  - Stacking over both cycles and passes (Basic, Expert, Technical only)
    ```python
    fc.query(stack='CYCLES_PASSES')
    ```
:::
:::{tab-item} Subset definitions
  - Fix a version
    ```python
    fc.query(version='2.0.1')
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
{class}`fcollections.implementations.NetcdfFilesDatabaseSwotLRL3`

::::{tab-set}
:::{tab-item} PeriodMixin
  - Time coverage
    ```python
    fc.time_coverage(subset='Expert')
    ```
  - Time holes
    ```python
    fc.time_holes(subset='Expert')
    ```
:::
:::{tab-item} HalfOrbitMixin
  - Half orbit range
    ```python
    fc.half_orbit_range(subset='Expert', phase='SCIENCE')
    ```
  - Cycle range
    ```python
    fc.cycle_range(subset='Expert', phase='SCIENCE')
    ```
:::
::::

## Stack for temporal analysis

The most prominent functionality is the ability to stack the half orbits when
the grid is fixed (Basic, Expert and Technical subsets). This allows
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
fc = NetcdfFilesDatabaseSwotLRL3("data_l3_lr_ssh")
ds = fc.query(stack='CYCLES', version='2.0.1', cycle_number=[1, 2, 3], pass_number=10, subset='Basic')
ds.ssha_filtered.data
```

```{code-cell}
ds = fc.query(stack='CYCLES_PASSES', version='2.0.1', cycle_number=[1, 2, 3], pass_number=[10, 11], subset='Basic')
ds.ssha_filtered.data
```

```{note}
Incomplete cycles are completed with invalids
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
    version='2.0.1',
    subset='Basic',
    bbox=bbox)
```

To query a subset of Swot LR L3 data crossing the bounding box:

```{note}
Lines of the swath crossing the bounding box will be entirely selected.
```

```{code-cell}
bbox = -126, 32, -120, 40
ds_area = fc.query(version='2.0.1', subset="Basic", cycle_number=2, pass_number=11, bbox=bbox)

# Figure
localbox_cartopy = bbox[0] - 1, bbox[2] + 1, bbox[1] - 1, bbox[3] + 1
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent(localbox_cartopy)

plot_kwargs = dict(
    x="longitude",
    y="latitude",
    cmap="Spectral_r",
    vmin=-0.2,
    vmax=0.2,
    cbar_kwargs={"shrink": 0.3},)

# SWOT KaRIn SLA plots
ds_area.ssha_filtered.plot.pcolormesh(ax=ax, **plot_kwargs)
ax.set_title("SLA KaRIn and selection box (in red)")
ax.coastlines()
ax.gridlines(draw_labels=['left', 'bottom'])


# Add the patch to the Axes
rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1.5, edgecolor='r', facecolor='none')
ax.add_patch(rect)
```

## Handling nadir clipped data in Level-3 Basic and Expert subsets

The L3_LR_SSH Basic and Expert subsets have the Nadir instrument data clipped in
the Sea Level Anomaly fields. The indexes where the nadir data has been
introduced are stored along ``num_nadir`` dimension. The SWOT implementation
offers various choices for handling this clipped data:

- ``nadir=False`` and ``swath=True``: remove the nadir data clipped. This is the
  default behavior
- ``nadir=True`` and ``swath=True``: do nothing and keep both the KaRIn and
  Nadir instruments data
- ``nadir=True`` and ``swath=False``: extract the Nadir instrument data only.
  This will give a dataset indexed along the ``num_nadir`` dimension. Because
  it returns the nadir data only, we lose the possibility of stacking multiple
  half orbits

```{code-cell}
ds_full = fc.query(version='2.0.1', subset="Basic", cycle_number=2, pass_number=11, nadir=True)
ds_swath = fc.query(version='2.0.1', subset="Basic", cycle_number=2, pass_number=11, nadir=False)

# set figures
localbox = 224.5, 228.5, -27, -23
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax1.set_extent(localbox)
ax2.set_extent(localbox)

plot_kwargs = dict(
    x="longitude",
    y="latitude",
    cmap="Spectral_r",
    vmin=-0.2,
    vmax=0.2,
    cbar_kwargs={"shrink": 0.3},)

# SWOT KaRIn SLA plots
ds_full.ssha_filtered.plot.pcolormesh(ax=ax1, **plot_kwargs)
ds_swath.ssha_filtered.plot.pcolormesh(ax=ax2, **plot_kwargs)

ax1.set_title("SLA KaRIn + Nadir")
ax1.coastlines()
ax1.gridlines(draw_labels=['left', 'bottom'])
ax2.set_title("SLA KaRIn only")
ax2.coastlines()
ax2.gridlines(draw_labels=['left', 'bottom'])
```

```{code-cell}
ds_nadir = fc.query(version='2.0.1', subset="Basic", cycle_number=2, pass_number=11, nadir=True, swath=False)

plt.plot(ds_nadir.latitude.values, ds_nadir.ssha_filtered.values)
plt.ylabel(f'{ds_nadir.ssha_filtered.attrs["standard_name"]} [{ds_nadir.ssha_filtered.attrs["units"]}]')
plt.xlabel(f'{ds_nadir.latitude.attrs["standard_name"]} [{ds_nadir.latitude.attrs["units"]}]')
plt.title("SLA Nadir")
```
