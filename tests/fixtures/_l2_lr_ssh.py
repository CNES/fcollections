from __future__ import annotations

import copy
import typing as tp
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fcollections.core import GroupMetadata, VariableMetadata
from fcollections.geometry import query_geometries

from ._generation import (
    HalfOrbitTrackCoordinatesGenerator,
    group_metadata_to_xarray,
)

if tp.TYPE_CHECKING:
    import numpy.typing as np_t


@pytest.fixture(scope="session")
def _l2_lr_ssh_basic_metadata() -> GroupMetadata:
    return GroupMetadata(
        name="/",
        variables=[
            VariableMetadata("time", np.dtype("M8[ns]"), ("num_lines",), {}),
            VariableMetadata("longitude", float, ("num_lines", "num_pixels"), {}),
            VariableMetadata("latitude", float, ("num_lines", "num_pixels"), {}),
            VariableMetadata("ssha_karin_2", float, ("num_lines", "num_pixels"), {}),
        ],
        dimensions={"num_lines": 9860, "num_pixels": 69},
        attributes={},
        subgroups=[],
    )


class BasicGenerator(HalfOrbitTrackCoordinatesGenerator):

    def ssha_karin_2(self, shape: tuple[int, ...]) -> np_t.NDArray[np.float64]:
        return np.random.random(shape)


def _swath_properties(
    pass_number: int, phase: str = "calval"
) -> tuple[float, float, float]:
    polygon = query_geometries(pass_number, phase).geometry.values[0]
    xx, yy = polygon.exterior.coords.xy
    xx, yy = np.array(xx.tolist()), np.array(yy.tolist())

    return xx[0], np.max(xx - xx[0]), yy[0]


@pytest.fixture(scope="session")
def l2_lr_ssh_basic_dataset(_l2_lr_ssh_basic_metadata: GroupMetadata) -> xr.Dataset:
    generator = BasicGenerator(
        *_swath_properties(25),
        t0=np.datetime64("2024-01-01T11:39"),
        dt=np.timedelta64(100, "ms"),
    )

    ds = group_metadata_to_xarray(_l2_lr_ssh_basic_metadata, generator)

    return ds.set_coords(["time", "longitude", "latitude"])


@pytest.fixture(scope="session")
def _l2_lr_ssh_expert_metadata(
    _l2_lr_ssh_basic_metadata: GroupMetadata,
) -> GroupMetadata:
    metadata = copy.deepcopy(_l2_lr_ssh_basic_metadata)
    metadata.variables.extend(
        [
            VariableMetadata(
                "cross_track_distance", float, ("num_lines", "num_pixels"), {}
            )
        ]
    )
    return metadata


class ExpertGenerator(BasicGenerator):

    def cross_track_distance(self, shape: tuple[int, int]) -> np_t.NDArray[np.float64]:
        num_lines, num_pixels = shape
        return (
            np.repeat(
                np.arange(-(num_pixels - 1) * 1000, num_pixels * 1000 + 1, 2000),
                num_lines,
            )
            .reshape(num_pixels, num_lines)
            .T
        )


@pytest.fixture(scope="session")
def l2_lr_ssh_expert_dataset(_l2_lr_ssh_expert_metadata: GroupMetadata) -> xr.Dataset:
    generator = ExpertGenerator(
        *_swath_properties(25),
        t0=np.datetime64("2024-01-01T11:39"),
        dt=np.timedelta64(100, "ms"),
    )

    ds = group_metadata_to_xarray(_l2_lr_ssh_expert_metadata, generator)

    return ds.set_coords(["time", "longitude", "latitude"])


@pytest.fixture(scope="session")
def l2_lr_ssh_unsmoothed_dataset(l2_lr_ssh_expert_dataset: xr.Dataset) -> xr.Dataset:
    return l2_lr_ssh_expert_dataset.copy(deep=True)


@pytest.fixture(scope="session")
def l2_lr_ssh_files() -> list[str]:
    # The files that we will use to test the listing
    return [
        "PIA2/Expert/cycle_577/SWOT_L2_LR_SSH_Expert_577_018_20230709T202541_20230709T211246_PIA2_02.nc",
        "PIB0/Basic/cycle_546/SWOT_L2_LR_SSH_Basic_546_011_20230608T191826_20230608T200933_PIB0_01.nc",
        "PIB0/Expert/cycle_546/SWOT_L2_LR_SSH_Expert_546_018_20230609T011606_20230609T020311_PIB0_01.nc",
        "PIB0/Expert/cycle_577/SWOT_L2_LR_SSH_Expert_577_011_20230709T142801_20230709T151908_PIB0_01.nc",
        "PIB0/Expert/cycle_577/SWOT_L2_LR_SSH_Expert_577_018_20230709T202541_20230709T211246_PIB0_01.nc",
        "PIB0/Expert/cycle_006/SWOT_L2_LR_SSH_Expert_006_011_20231102T215435_20231102T224354_PIB0_01.nc",
        "PIB0/Expert/cycle_006/SWOT_L2_LR_SSH_Expert_006_532_20231121T120000_20231121T130000_PIB0_01.nc",
        "PIB0/Expert/cycle_006/SWOT_L2_LR_SSH_Expert_006_533_20231121T130000_20231121T140000_PIB0_01.nc",
        "PIB0/Expert/cycle_007/SWOT_L2_LR_SSH_Expert_007_532_20231210T120000_20231210T130000_PIB0_01.nc",
        "PIB0/Expert/cycle_007/SWOT_L2_LR_SSH_Expert_007_533_20231210T130000_20231210T140000_PIB0_01.nc",
        "PIC2/Basic/cycle_482/SWOT_L2_LR_SSH_Basic_482_011_20230406T051804_20230406T060909_PIC2_01.nc",
        "PIC2/Basic/cycle_482/SWOT_L2_LR_SSH_Basic_482_012_20230406T060910_20230406T070014_PIC2_02.nc",
        "PIC2/Basic/cycle_482/SWOT_L2_LR_SSH_Basic_482_025_20230406T100000_20230406T110000_PIC2_02.nc",
        "PIC2/Basic/cycle_482/SWOT_L2_LR_SSH_Basic_482_026_20230406T110000_20230406T120000_PIC2_02.nc",
        "PIC2/Basic/cycle_483/SWOT_L2_LR_SSH_Basic_483_025_20230427T100000_20230427T110000_PIC2_02.nc",
        "PIC2/Basic/cycle_483/SWOT_L2_LR_SSH_Basic_483_026_20230427T110000_20230427T120000_PIC2_02.nc",
        "PGC0/Expert/cycle_546/SWOT_L2_LR_SSH_Expert_546_018_20230609T011606_20230609T020311_PGC0_02.nc",
        "PGC0/Expert/cycle_546/SWOT_L2_LR_SSH_Expert_546_018_20230609T011606_20230609T020311_PGC0_01.nc",
        "PID0/Unsmoothed/cycle_010/SWOT_L2_LR_SSH_Unsmoothed_010_004_20240125T025352_20240125T034438_PID0_01.nc",
    ]


@pytest.fixture(scope="session")
def l2_lr_ssh_dir_empty_files(
    tmp_path_factory: pytest.TempPathFactory, l2_lr_ssh_files: list[str]
) -> Path:
    root = tmp_path_factory.mktemp("l2_lr_ssh")

    # create test folder
    for filepath in l2_lr_ssh_files:
        filepath = root / Path(filepath).name

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    return root


@pytest.fixture(scope="session")
def l2_lr_ssh_dir_empty_files_layout(
    tmp_path_factory: pytest.TempPathFactory, l2_lr_ssh_files: list[str]
) -> Path:
    root = tmp_path_factory.mktemp("l2_lr_ssh")

    # create test folder
    for filepath in l2_lr_ssh_files:
        filepath = root / Path(filepath)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    return root


@pytest.fixture(scope="session")
def l2_lr_ssh_dir(
    tmpdir_factory: pytest.TempdirFactory,
    l2_lr_ssh_files: list[str],
    l2_lr_ssh_basic_dataset: xr.Dataset,
    l2_lr_ssh_expert_dataset: xr.Dataset,
    l2_lr_ssh_unsmoothed_dataset: xr.Dataset,
):

    # Create fake data for the netcdf files, do not just use touch() to simulate
    # the file tree
    root_dir = Path(tmpdir_factory.mktemp("l2_lr_ssh"))
    for file in l2_lr_ssh_files:
        relative_path = Path(file)
        if "Basic" in relative_path.name:
            path = root_dir / "basic" / relative_path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            l2_lr_ssh_basic_dataset.to_netcdf(path)
        elif "Expert" in relative_path.name:
            path = root_dir / "expert" / relative_path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            l2_lr_ssh_expert_dataset.to_netcdf(path)
        else:
            path = root_dir / "unsmoothed" / relative_path.name
            path.parent.mkdir(parents=True, exist_ok=True)
            l2_lr_ssh_unsmoothed_dataset.to_netcdf(path, group="left")
            l2_lr_ssh_unsmoothed_dataset.to_netcdf(path, group="right", mode="a")

    return root_dir


@pytest.fixture(scope="session")
def l2_lr_ssh_basic_dir(l2_lr_ssh_dir: Path) -> Path:
    return l2_lr_ssh_dir / "basic"


@pytest.fixture(scope="session")
def l2_lr_ssh_expert_dir(l2_lr_ssh_dir: Path) -> Path:
    return l2_lr_ssh_dir / "expert"


@pytest.fixture(scope="session")
def l2_lr_ssh_unsmoothed_dir(l2_lr_ssh_dir: Path) -> Path:
    return l2_lr_ssh_dir / "unsmoothed"


@pytest.fixture(scope="session")
def l2_lr_ssh_basic_files(l2_lr_ssh_basic_dir: Path) -> list[Path]:
    return sorted(l2_lr_ssh_basic_dir.iterdir())


@pytest.fixture(scope="session")
def l2_lr_ssh_expert_files(l2_lr_ssh_expert_dir: Path) -> list[Path]:
    return sorted(l2_lr_ssh_expert_dir.iterdir())


@pytest.fixture(scope="session")
def l2_lr_ssh_unsmoothed_files(l2_lr_ssh_unsmoothed_dir: Path) -> list[Path]:
    return sorted(l2_lr_ssh_unsmoothed_dir.iterdir())


@pytest.fixture(scope="session")
def l2_lr_ssh_basic_files_unique(l2_lr_ssh_basic_files: list[Path]) -> list[Path]:
    return l2_lr_ssh_basic_files


@pytest.fixture(scope="session")
def l2_lr_ssh_expert_files_unique(l2_lr_ssh_expert_files: list[Path]) -> list[Path]:
    return l2_lr_ssh_expert_files[:6]
