from __future__ import annotations

import typing as tp
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

if tp.TYPE_CHECKING:
    import numpy.typing as np_t


def _build_grid(
    lon: np_t.NDArray[np.int64 | np.float64],
    lat: np_t.NDArray[np.int64 | np.float64],
    lon_name: str = "longitude",
    lat_name: str = "latitude",
) -> xr.Dataset:
    np.random.seed(0)
    time = np.arange("2024-01-01T12", "2024-01-01T18", dtype="M8[h]")

    sla = (np.arange(lon.size * lat.size) / (lon.size * lat.size)).reshape(
        lon.size, lat.size
    )
    sla = np.repeat(sla, time.size).reshape(time.size, *sla.shape)
    ds = xr.Dataset(
        data_vars=dict(
            time=("time", time.astype("M8[ns]")),
            sla=(("time", lon_name, lat_name), sla),
        ),
        coords={lon_name: ((lon_name), lon), lat_name: ((lat_name), lat)},
    )
    return ds


@pytest.fixture(scope="session")
def l4_ssha_dataset_0_360() -> xr.Dataset:
    lon, lat = np.array([0, 1, 2, 358, 359]), np.array([1, 2])
    return _build_grid(lon, lat, "lon", "lat")


@pytest.fixture(scope="session")
def l4_ssha_dataset_180_180() -> xr.Dataset:
    lon, lat = np.array([-2, -1, 0, 1, 2]), np.array([-2, -1, 0, 1, 2])
    return _build_grid(lon, lat)


@pytest.fixture(scope="session")
def l4_ssha_dataset_reversed_lat() -> xr.Dataset:
    lon, lat = np.array([-2, -1, 0, 1, 2]), np.array([2, 1, 0, -1, -2])
    return _build_grid(lon, lat)


@pytest.fixture(scope="session")
def l4_ssha_files() -> list[str]:
    # The files that we will use to test the listing
    return [
        "aviso/v1.0/miost/science/dt_global_allsat_phy_l4_20230728_20240912.nc",
        "aviso/v1.0/miost/science/dt_global_allsat_phy_l4_20230729_20240912.nc",
        "aviso/v3.0/miost/dt_global_allsat_phy_l4_20230728_20240912.nc",
        "aviso/v3.0/miost/dt_global_allsat_phy_l4_20230729_20240912.nc",
        "aviso/v0.3/4dvarnet/calval/dt_global_allsat_phy_l4_20230729_20240913.nc",
        "aviso/v0.3/4dvarqg/calval/dt_global_allsat_phy_l4_20230729_20240912.nc",
        "aviso/v0.3/4dvarqg/calval/nrt_global_allsat_phy_l4_20230729_20240912.nc",
        "flat/nrt_europe_allsat_phy_l4_20231016_20231103.nc",
        "flat/nrt_global_allsat_phy_l4_20231016_20231103.nc",
        "flat/dt_global_allsat_phy_l4_20231017_20231106.nc",
        "cmems/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_202211/2023/10/dt_global_allsat_phy_l4_20230728_20240912.nc",
        "cmems/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_PT12H_202211/2023/10/nrt_global_allsat_phy_l4_20230729_20240912.nc",
        "cmems/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.5deg_P1D_202211/2023/10/nrt_global_allsat_phy_l4_20230729_20240913.nc",
        "cmems/cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202411/2023/10/nrt_global_allsat_phy_l4_20230728_20251212.nc",
    ]


@pytest.fixture(scope="session")
def l4_ssha_dir(
    tmp_path_factory: pytest.TempPathFactory, l4_ssha_files: list[str]
) -> Path:
    root = tmp_path_factory.mktemp("l4_karin_nadir")

    # create test folder
    for filepath in l4_ssha_files:
        filepath = root / filepath

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
    return root


@pytest.fixture(scope="session")
def l4_ssha_dir_no_layout(l4_ssha_dir: Path) -> Path:
    return l4_ssha_dir / "flat"


@pytest.fixture(scope="session")
def l4_ssha_dir_layout_aviso(l4_ssha_dir: Path) -> Path:
    return l4_ssha_dir / "aviso"


@pytest.fixture(scope="session")
def l4_ssha_dir_layout_cmems(l4_ssha_dir: Path) -> Path:
    return l4_ssha_dir / "cmems"
