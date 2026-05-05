---
kernelspec:
  language: python
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
---
# Getting started

``fcollections`` is a library that aims at reading a collections of files. Its
primary goal is to combine the selection, reading and concatenation of files
within a common model.

Let's set up a minimal case with stub data for the SWOT altimetry mission.

```{code-cell}

import tempfile
import numpy as np
import xarray as xr

# Create stub data
path = tempfile.mkdtemp()
ds = xr.Dataset(data_vars={
    "ssha": (('num_lines', 'num_pixels'), np.random.random((9860, 69))),
    "swh": (('num_lines', 'num_pixels'), np.random.random((9860, 69))),})
ds.to_netcdf(f'{path}/SWOT_L2_LR_SSH_Expert_001_011_20240101T000000_20240101T030000_PGC0_01.nc')
ds.to_netcdf(f'{path}/SWOT_L2_LR_SSH_Expert_001_012_20240101T030000_20240101T060000_PGC0_01.nc')
```

## Implementations

When confronted to a files collection, the first step is to try and find if
an implementation matching the data already exists. Such implementation may be
found in the [catalog](implementations/catalog)

From the catalog, we can see that {class}`NetcdfFilesDatabaseSwotLRL2 <fcollections.implementations.NetcdfFilesDatabaseSwotLRL2>`
matches our file names. In case no implementation is available, the user can
build its own following [creation procedure](custom).

## Listing files

An implementation can be used by simply giving the path to the data. An
important endpoint for the implementation is the ability to list files matching
given criterias

```{code-cell}
from fcollections.implementations import NetcdfFilesDatabaseSwotLRL2

fc = NetcdfFilesDatabaseSwotLRL2(path)
fc.list_files(cycle_number=1)
```

Listing files using filters is the first step toward subsetting the files set.


## Query data

Another important endpoint is the ability to read the file contents using the
``query`` method.


```{code-cell}
fc.query()
```

The method returns a {class}`xarray.Dataset` containing the combined data for
all files matching the regex specified by the implementation.

It is possible to load only a subset of the data by applying filters in the
query. For example, giving the ``cycle_number`` and ``pass_number`` argument
will select one half orbit of our altimetry mission.

```{code-cell}
fc.query(cycle_number=1, pass_number=11)
```

Variable selection is also available to return only part of the data

```{code-cell}
ds = fc.query(selected_variables=['ssha'])
list(ds.variables)
```

### Filter types

Each implementation has its own filters. By order of availability, the user
should consult:

- The ``Query overview`` section of the implementation's ``Documentation`` (see the [catalog](implementations/catalog))
- The API documentation of the implementation's method (see the [catalog](implementations/catalog))
- The prompted help displayed in a jupyter notebook or Python interpreter

```{code-cell}
:tags: [hide-output]
fc.query?
```

### Filter values

Possible values for a given filter can be displayed

```{code-cell}
fc.filter_values('version')
```

Only filters whose information are contained in the intermediated folders can be
scanned in a quick way, other will trigger a full scan. As such, to ensure
optimal performance, this method should be called with the layouts enabled, with
files organized with folders (see the [advanced section](#disable-layouts)), and
on filters whose information is encoded in the folders.

## Access metadata

The database can display information about the variables and attributes
contained in the files' collection using the ``variables_info`` method

```{code-cell}
# Use the enumeration name for filtering a specific subset
fc.variables_info()
```

It will offer a simple collapsible tree view with multiple levels of nesting
depending on the data you manipulate

## Subsets

### Errors on mixed subsets

In order to return consistent results, most methods must work on an homogeneous
subset of data. In case multiple subsets are mixed (for example Expert and
Unsmoothed datasets), proper filters matching the partitioning keys must be
given. If these filters are missing, an error with the possible choices will be
raised.

```{code-cell}
:tags: [raises-exception]
# Create Unsmoothed file, this will mix Expert and Unsmoothed dataset
ds.to_netcdf(f'{path}/SWOT_L2_LR_SSH_Unsmoothed_001_012_20240101T030000_20240101T060000_PGC0_01.nc')

# This will not work because we don't know if we need to display Expert or
# Unsmoothed metadata
fc.variables_info()
```

### Compatibility matrix

The following table summarizes which methods can work on mixed data. Most
methods need homogeneous data and will require filtering the subset.

| Method             | Works on mixed data ? |
|--------------------|-----------------------|
| ``list_files``     | Yes                   |
| ``variables_info`` | No                    |
| ``filter_values``  | No                    |
| ``query``          | No                    |
| ``map``            | No                    |

### Listing subsets

Subsets that are on the file system can be listed using the
{meth}`subsets <fcollections.core.FilesDatabase.subsets>` property.

```{code-cell}
fc.subsets
```

One of the returned choices must be selected and used as a filter to work on an
homogeneous dataset.

```{code-cell}
# Use the enumeration name for filtering a specific subset
fc.variables_info(subset='Expert')
```
