import logging
from pathlib import Path

from altimetry_downloader_aviso import get

logging.basicConfig()
logging.getLogger("altimetry_downloader_aviso").setLevel("INFO")

DATA_DIR = Path(__file__).resolve().parent.parent / "data_l2_lr_ssh"
DATA_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":

    get(
        "SWOT_L2_LR_SSH_Basic",
        output_dir=DATA_DIR,
        cycle_number=[9, 10, 11],
        pass_number=[10, 11],
        version="P?C?",
    )

    get(
        "SWOT_L2_LR_SSH_Unsmoothed",
        output_dir=DATA_DIR,
        cycle_number=9,
        pass_number=10,
        version="PGC?",
    )
