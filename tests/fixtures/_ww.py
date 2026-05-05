from __future__ import annotations

import dataclasses as dc
import pickle
import typing as tp
from pathlib import Path

import netCDF4 as nc4
import numpy as np
import pytest
import xarray as xr

from ._generation import (
    HalfOrbitTrackCoordinatesGenerator,
    group_metadata_to_netcdf,
    group_metadata_to_xarray,
    pluck,
)

if tp.TYPE_CHECKING:
    import numpy.typing as np_t

    from fcollections.core import GroupMetadata


@dc.dataclass
class LightExtendedSubsetGenerator(HalfOrbitTrackCoordinatesGenerator):

    offset: float = 0

    def H18(self, shape: tuple[int, ...]) -> np_t.NDArray[np.float64]:
        h18 = np.arange(self.offset, self.offset + shape[0])
        self.offset1 = self.offset + shape[0]
        return h18

    def __next__(self):
        super().__next__()
        self.offset = self.offset1


def _decimate_dimension(group: GroupMetadata):
    try:
        # Don't use the full shapes, smaller datasets are enough for unit
        # testing
        group.dimensions["n_box"] = 100
    except KeyError:
        # Nothing to do
        pass


@pytest.fixture(scope="session")
def _light_metadata(resources_directory: Path) -> GroupMetadata:
    path = resources_directory / "L3_LR_WIND_WAVE_Light_2.0"
    with path.open(mode="rb") as f:
        metadata = pickle.load(f)

    metadata.apply(_decimate_dimension)
    return metadata


@pytest.fixture(scope="session")
def l3_lr_ww_light_files(
    tmp_path_factory: pytest.TempPathFactory, _light_metadata: GroupMetadata
) -> Path:
    tmp_path = tmp_path_factory.mktemp("l3_karin_wind_wave")

    light_file_names = [
        "SWOT_L3_LR_WIND_WAVE_002_001_20230811T021853_20230811T031018_v2.0.nc",
        "SWOT_L3_LR_WIND_WAVE_002_002_20230811T031018_20230811T040607_v2.0.nc",
    ]
    light_files = [tmp_path / x for x in light_file_names]

    generator = LightExtendedSubsetGenerator()
    for light_file in light_files:
        with nc4.Dataset(light_file, mode="w") as nds:
            group_metadata_to_netcdf(nds, _light_metadata, generator)
            next(generator)
    return light_files


@pytest.fixture(scope="session")
def l3_lr_ww_light_dataset(_light_metadata: GroupMetadata) -> xr.Dataset:
    ds = group_metadata_to_xarray(_light_metadata, LightExtendedSubsetGenerator())
    return ds.set_coords(["longitude", "latitude"])


@pytest.fixture(scope="session")
def _extended_metadata(resources_directory: Path) -> GroupMetadata:
    path = resources_directory / "L3_LR_WIND_WAVE_Extended_2.0"
    with path.open(mode="rb") as f:
        metadata = pickle.load(f)
    metadata.apply(_decimate_dimension)
    return metadata


@pytest.fixture(scope="session")
def l3_lr_ww_extended_files(
    tmp_path_factory: pytest.TempPathFactory, _extended_metadata: GroupMetadata
) -> Path:
    tmp_path = tmp_path_factory.mktemp("l3_karin_wind_wave")

    light_file_names = [
        "SWOT_L3_LR_WIND_WAVE_Extended_002_001_20230811T021853_20230811T031018_v2.0.nc",
        "SWOT_L3_LR_WIND_WAVE_Extended_002_002_20230811T031018_20230811T040607_v2.0.nc",
    ]
    files = [tmp_path / x for x in light_file_names]

    generator = LightExtendedSubsetGenerator()
    for extended_file in files:
        with nc4.Dataset(extended_file, mode="w") as nds:
            group_metadata_to_netcdf(nds, _extended_metadata, generator)
            next(generator)
    return files


@pytest.fixture(scope="session")
def l3_lr_ww_extended_dataset(_extended_metadata: GroupMetadata) -> xr.Dataset:
    ds = group_metadata_to_xarray(
        pluck(_extended_metadata, "/tile_10km/box_40km"), LightExtendedSubsetGenerator()
    )
    return ds.set_coords(["longitude", "latitude"])


@pytest.fixture(scope="session")
def l3_lr_ww_files() -> list[str]:
    # The files that we will use to test the listing
    return [
        "v2_0/Light/cycle_482/SWOT_L3_LR_WIND_WAVE_482_011_20230406T051804_20230406T060909_v2.0.nc",
        "v2_0/Light/cycle_482/SWOT_L3_LR_WIND_WAVE_482_012_20230406T060910_20230406T070014_v2.0.nc",
        "v3_0/Extended/cycle_010/SWOT_L3_LR_WIND_WAVE_Extended_010_010_20240125T025352_20240125T034438_v3.0.nc",
    ]


@pytest.fixture(scope="session")
def l3_lr_ww_dir_layout(
    tmp_path_factory: pytest.TempPathFactory, l3_lr_ww_files: list[str]
) -> Path:
    root = tmp_path_factory.mktemp("l3_lr_ww")

    # create test folder
    for filepath in l3_lr_ww_files:
        filepath = root / filepath

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    return root
