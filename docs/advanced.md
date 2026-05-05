# Advanced usage

(remote)=

## Remote file systems

It is possible to access a file set from a remote location. Fcollections is based on
the powerful ``fsspec`` abstraction. As a consequence, files collections might
accept any file system.

```{warning}
In case the reader does not support a specific file system, an error will be
triggered. The solution is to implement its own reader following
{doc}`custom`
```

The following code shows how to access data directly from the AVISO public FTP
server using both FTP and SFTP protocols. You will need credentials to
authentify (see the [aviso website](https://www.aviso.altimetry.fr/))

::::{tab-set}
:::{tab-item} FTP
```python
from fsspec.implementations.ftp import FTPFileSystem
from fcollections.implementations import NetcdfFilesDatabaseSwotLRL2

fs = FTPFileSystem('ftp-access.aviso.altimetry.fr', 21, username='...', password='...')
db = NetcdfFilesDatabaseSwotLRL2('/swot_products/l2_karin/l2_lr_ssh/PIC2/Expert/cycle_031', fs=fs)
ds = db.list_files(pass_number=1)
```
:::
:::{tab-item} SFTP
```python
from fsspec.implementations.sftp import SFTPFileSystem
from fcollections.implementations import NetcdfFilesDatabaseSwotLRL2

fs = SFTPFileSystem(host='ftp-access.aviso.altimetry.fr', port=2221, username='...', password='...')
db = NetcdfFilesDatabaseSwotLRL2('/swot_products/l2_karin/l2_lr_ssh/PIC2/Expert/cycle_031', fs=fs)
ds = db.list_files(pass_number=1)
```
```{note}
[paramiko](https://www.paramiko.org/) must be installed to use the SFTP
implementation of ``fsspec``
:::
::::

Remote file system listing can be quite long. Implementations are usually
shipped with layouts for an improved listing speed. See the
{ref}`Layout <layout>` introduction if listing performance becomes an issue.

(disable-layouts)=

## Disable layouts

Files Collections implementations can define up to two classes of pre-configured
layouts:

- Flat layouts: the files are in a single folder without any nesting
- Official layouts: files' organization mirrors public data providers such as
  AVISO, Copernicus Marine, etc...

There is no easy way to ensure that the pre-configured layouts perfectly matches
your target. The current strategy is to raise a {class}`LayoutMismatchError <fcollections.core.LayoutMismatchError>`
if a folder is not recognized. This behavior can be changed setting the
``enable_layouts`` parameter:

```python
from fcollections.implementations import NetcdfFilesDatabaseSwotLRL2

db = NetcdfFilesDatabaseSwotLRL2('/mypath/with/custom_nesting', enable_layouts=False)
```

This will disable the branch exploration pruning and slow down the files
listing. To avoid losing performance, it is possible to modify an existing
implementation by defining an additionnal {ref}`layout <layout>`.

```python
from fcollections.core import Layout

layout = Layout(...)
NetcdfFilesDatabaseSwotLRL2.layouts.append(layout)

# Branch pruning will be enabled again
db = NetcdfFilesDatabaseSwotLRL2('/mypath/with/custom_nesting', enable_layouts=True)
```

## Follow symlinks

By default, symbolic links on a file system will not be resolved. This means
queries might return empty results if your target collection contains symbolic
links.

Enabling the ``follow_symlinks`` parameter causes symbolic links to be resolved.
Note that, although rare, they may form cyclic graphs, which are not handled
correctly by the current implementation.

```python
db = NetcdfFilesDatabaseSwotLRL2('/mypath/with/symlinks', follow_symlinks=True)
```

:::{note}
The functionality has been tested for POSIX-compliant
{class}`LocalFileSystem <fsspec.implementations.local.LocalFileSystem>` only
:::
