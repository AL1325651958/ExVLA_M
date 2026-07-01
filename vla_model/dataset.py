"""HDF5 excavation trajectory dataset with sliding window.

All episodes are pre-loaded into memory at init time for fast GPU utilization.
"""

import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ExcavatorDataset(Dataset):
    """Sliding-window dataset over HDF5 excavation episodes.

    Pre-loads ALL images into memory at init time. Each __getitem__ is a
    fast memory slice — no disk I/O during training.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 8,
        action_chunk: int = 1,
        img_size: int = 224,
        split: str = "train",
        train_split: float = 0.9,
        sample_ratio: float = 1.0,  # 1.0=all, 0.2=20% for fast training
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.action_chunk = action_chunk
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.sample_ratio = sample_ratio

        # Find all H5/HDF5 files
        self.file_list = sorted(
            glob.glob(str(self.data_dir / "**" / "*.h5"), recursive=True) +
            glob.glob(str(self.data_dir / "**" / "*.hdf5"), recursive=True)
        )
        if len(self.file_list) == 0:
            candidates = list(self.data_dir.glob("**/*.h5")) + list(self.data_dir.glob("**/*.hdf5"))
            self.file_list = [str(c) for c in candidates]
        print(f"[{split}] Found {len(self.file_list)} episode(s)")

        # Determine which file indices belong to this split FIRST
        n_files = len(self.file_list)
        n_train_files = max(1, int(n_files * train_split))
        self.train_files = set(range(n_train_files))
        self.val_files   = set(range(n_train_files, n_files))

        if split == "train":
            my_files = self.train_files
        elif split == "val":
            my_files = self.val_files
        else:
            my_files = set(range(n_files))

        # ---- Pre-load only OUR split's episodes into memory ----
        self._episodes = []

        for fidx, fpath in enumerate(self.file_list):
            if fidx not in my_files:
                continue
            # Map original fidx -> local _episodes index
            ep_idx = len(self._episodes)
            print(f"[{split}] Loading episode {ep_idx+1}/{len(my_files)}: {Path(fpath).name} ...", flush=True)
            ep = self._load_and_preprocess(fpath)
            if ep is None:
                print(f"  -> SKIPPED (corrupted)", flush=True)
                continue
            self._episodes.append(ep)
            print(f"  -> {len(ep['qpos'])} frames, "
                  f"mem: {sum(a.nbytes if hasattr(a,'nbytes') else 0 for a in ep.values()) / 1024**2:.0f} MB + raw imgs", flush=True)

        # Build samples from loaded episodes (all local)
        self.samples = []
        for fidx, ep in enumerate(self._episodes):
            n_frames = len(ep["qpos"])
            total_window = seq_len + action_chunk
            for start in range(0, n_frames - total_window + 1):
                self.samples.append((fidx, start))

        # Sub-sample if ratio < 1.0 (for fast training)
        if sample_ratio < 1.0:
            step = max(1, int(1.0 / sample_ratio))
            self.samples = self.samples[::step]  # take every N-th
            print(f"[{split}] Subsampled to {len(self.samples)} samples ({sample_ratio*100:.0f}%)")

        print(f"[{split}] Loaded {len(self._episodes)} episodes, {len(self.samples)} samples")

    def _parse_excavator_id(self, fpath: str) -> int:
        """Parse excavator model ID from file path. Returns 0=75, 1=306, 2=490, 3=unknown."""
        path_lower = fpath.lower()
        if '/75/' in path_lower or '\\75\\' in path_lower:
            return 0
        elif '/306/' in path_lower or '\\306\\' in path_lower:
            return 1
        elif '/490/' in path_lower or '\\490\\' in path_lower:
            return 2
        return 3

    def _load_and_preprocess(self, fpath: str) -> dict:
        """Load raw uint8 data from one HDF5 episode (no preprocessing).
        Preprocessing happens on-the-fly in __getitem__.
        Returns None if file is corrupted."""
        try:
            f = h5py.File(fpath, 'r')
            # Keep file handle open for lazy access
            mains_raw = f['observations/images/main']      # [N, H, W, 3] uint8 BGR
            elevations_raw = f['observations/images/elevation']  # [N, 200, 200, 3]
            qpos = f['observations/qpos'][:].astype(np.float32)
            if 'action' in f:
                action = f['action'][:].astype(np.float32)
            else:
                action = np.zeros_like(qpos)
                action[:-1] = qpos[1:]
                action[-1] = qpos[-1]
        except OSError as e:
            print(f"  SKIP corrupted file: {fpath}: {e}")
            return None
        except KeyError as e:
            print(f"  SKIP missing key: {fpath}: {e}")
            return None

        excv_id = self._parse_excavator_id(fpath)
        return {
            "_h5": f,                          # keep file handle alive
            "mains_raw": mains_raw,            # h5py Dataset, lazy access
            "elevations_raw": elevations_raw,  # h5py Dataset, lazy access
            "qpos": qpos,
            "action": action,
            "excavator_id": excv_id,           # int: 0=75, 1=306, 2=490
        }

    def _preprocess_image(self, img_bgr: np.ndarray, augment: bool = False) -> np.ndarray:
        """Resize, BGR→RGB, normalize. Returns [3, H, W] float32."""
        img = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Data augmentation (only for training)
        if augment:
            # Random brightness/contrast jitter (mild, ~10%)
            if np.random.random() < 0.5:
                alpha = 1.0 + np.random.uniform(-0.1, 0.1)  # contrast
                beta = np.random.uniform(-10, 10)            # brightness
                img = (alpha * img.astype(np.float32) + beta).clip(0, 255).astype(np.uint8)
            # Random horizontal flip (excavator can face either direction)
            if np.random.random() < 0.3:
                img = img[:, ::-1].copy()

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img.transpose(2, 0, 1)  # CHW

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fidx, start = self.samples[idx]
        ep = self._episodes[fidx]
        T = self.seq_len
        K = self.action_chunk

        end = start + T

        # Preprocess on-the-fly from raw uint8 h5py datasets (memory efficient)
        rgb_seq = np.zeros((T, 3, self.img_size, self.img_size), dtype=np.float32)
        elev_seq = np.zeros((T, 3, self.img_size, self.img_size), dtype=np.float32)

        augment = (self.split == "train")
        for t in range(T):
            i = start + t
            rgb_seq[t] = self._preprocess_image(ep["mains_raw"][i], augment)
            elev_seq[t] = self._preprocess_image(ep["elevations_raw"][i], augment)

        qpos_seq = ep["qpos"][start:end]         # [T, 4]
        # Target: absolute next qpos (not delta)
        next_qpos = ep["action"][start + T - 1]   # [4] -- absolute joint angles
        excv_id = ep["excavator_id"]

        return {
            "rgb": torch.from_numpy(rgb_seq),
            "elevation": torch.from_numpy(elev_seq),
            "qpos": torch.from_numpy(qpos_seq.copy()),
            "action": torch.from_numpy(next_qpos.reshape(1, 4).copy()),
            "excavator_id": torch.tensor(excv_id, dtype=torch.long),
        }
