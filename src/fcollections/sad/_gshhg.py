from __future__ import annotations

import io
import itertools
import logging
import tarfile
import typing as tp

import requests

from ._interface import IAuxiliaryDataFetcher

if tp.TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

logger = logging.getLogger(__name__)


class GSHHG(IAuxiliaryDataFetcher):
    """Data from GSHHG data base.

    A Global Self-consistent, Hierarchical, High-resolution Geography Database.

    It contains the following geometries' types: GSHHS (aka. coastlines), river
    and border. It also comes in 5 resolutions: f, h, i, l, c (high resolution
    to crude resolution). The key for getting an asset is a composition of the
    geometry type and the resolution, for example: 'GSHHS_c', 'border_i',
    'river_l'

    Parameters
    ----------
    preferred_target_folder
        The folder where data will be downloaded if it is missing. Default to
        the user home (~/.config/sad)
    """

    HTTP_URL = "http://www.soest.hawaii.edu/pwessel"
    FILE = "gshhg/gshhg-gmt-2.3.7.tar.gz"

    @property
    def keys(self) -> set[str]:
        resolutions = {"c", "l", "i", "h", "f"}
        subset = {"border", "GSHHS", "river"}
        return {f"{s}_{r}" for s, r in itertools.product(subset, resolutions)}

    def _download(self, remote_file: str, target_folder: Path):
        fetch_http_file(self.HTTP_URL, self.FILE, target_folder)
        return target_folder / remote_file

    def _file_name(self, key: str):
        return f"binned_{key}.nc"


def fetch_http_file(url: str, filename: str, target_folder: Path):

    full_url = url + "/" + filename

    logger.info("Downloading %s...", full_url)
    tar_data = io.BytesIO()
    response = requests.get(full_url, timeout=60)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download file from {full_url}") from e

    # Filter out non-netcdf and flatten the tar gz structure
    def tar_info_filter(tar_info: tarfile.TarInfo, _) -> tarfile.TarInfo | None:
        if ".nc" not in tar_info.name:
            logger.debug("Not an netcdf, skipping extraction")
            return None

        logger.debug("Extracting %s", tar_info.name)
        tar_info.path = tar_info.path.split("/")[-1]
        return tar_info

    # Extract in-memory buffer
    tar_data.write(response.content)
    tar_data.seek(0)
    with tarfile.open(fileobj=tar_data, mode="r") as tar:
        for member in tar.getmembers():
            tar.extract(member, path=target_folder, filter=tar_info_filter)

    logger.info("Downloading %s... Done", full_url)
