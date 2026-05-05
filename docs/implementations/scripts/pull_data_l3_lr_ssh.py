import logging
from pathlib import Path

from altimetry_downloader_aviso import get

logging.basicConfig()
logging.getLogger("altimetry_downloader_aviso").setLevel("INFO")

DATA_DIR = Path(__file__).resolve().parent.parent / "data_l3_lr_ssh"
DATA_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":

    get(
        "SWOT_L3_LR_SSH_Basic",
        output_dir=DATA_DIR,
        version="2.0.1",
        cycle_number=[1, 2, 3],
        pass_number=[10, 11],
    )
