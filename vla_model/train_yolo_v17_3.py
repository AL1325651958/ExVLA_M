"""Train STVTA V17.3 with within-modality mask diversity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vla_model.train_yolo_v17_1 import main as train_v17_1_main


def main():
    return train_v17_1_main(training_version="v17.3")


if __name__ == "__main__":
    main()
