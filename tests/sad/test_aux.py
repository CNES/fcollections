import io
import tarfile
from unittest.mock import Mock, patch

import pytest
import requests

from fcollections.sad import GSHHG, KarinFootprints


def test_karin_footprint(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("sad")
    mock_resp = Mock()
    mock_resp.content = b"Hello world"
    mock_resp.raise_for_status.return_value = None

    aux = KarinFootprints()
    with (
        patch("requests.get", return_value=mock_resp) as get,
        patch("fcollections.sad.KarinFootprints.lookup_folders", return_value=[path]),
    ):
        fetched_file = aux["calval"]

    with open(fetched_file, "rb") as f:
        actual = f.read()

    assert actual == b"Hello world"
    get.assert_called_once_with(
        aux.HTTP_URL + "/KaRIn_2kms_calval_geometries.geojson.zip", timeout=60
    )


def test_karin_footprint_http_error(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("sad")
    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

    aux = KarinFootprints()
    with (
        patch("fcollections.sad._karin.requests.get", return_value=mock_resp),
        patch("fcollections.sad.KarinFootprints.lookup_folders", return_value=[path]),
    ):
        with pytest.raises(RuntimeError):
            aux["calval"]


@pytest.fixture
def gshhg_tar_gz() -> bytes:
    files = {
        "binned_border_i.nc": b"hello",
        "binned_GSHHS_h.nc": b"world",
        "README": b"foo",
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = io.BytesIO(content)
            info = tarfile.TarInfo(name=f"gshhg-gmt-2.3.7/{name}")
            info.size = len(content)

            tar.addfile(info, fileobj=data)

    return buf.getvalue()


def test_gshhg(gshhg_tar_gz: bytes, tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("sad")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"Hello World"
        info = tarfile.TarInfo(name="binned_border_i.nc")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    mock_resp = Mock()
    mock_resp.content = buf.getvalue()
    mock_resp.raise_for_status.return_value = None

    aux = GSHHG()
    with (
        patch("requests.get", return_value=mock_resp) as get,
        patch("fcollections.sad.GSHHG.lookup_folders", return_value=[path]),
    ):
        fetched_file = aux["border_i"]

    with open(fetched_file) as f:
        assert f.read() == "Hello World"

    get.assert_called_once_with(
        aux.HTTP_URL + "/gshhg/gshhg-gmt-2.3.7.tar.gz", timeout=60
    )


def test_gshhg_http_error(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("sad")
    mock_resp = Mock()
    mock_resp.raise_for_status.side_effect = requests.HTTPError("404")

    aux = GSHHG()
    with (
        patch("fcollections.sad._gshhg.requests.get", return_value=mock_resp),
        patch("fcollections.sad.GSHHG.lookup_folders", return_value=[path]),
    ):
        with pytest.raises(RuntimeError):
            aux["border_i"]
