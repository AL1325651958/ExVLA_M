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
        train_split: float = 0.857,
        sample_ratio: float = 1.0,
        force_excv_id: int = None,  # override all excavator_ids (single-machine training)
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.action_chunk = action_chunk
        self.img_size = img_size
        self.split = split
        self.train_split = train_split
        self.sample_ratio = sample_ratio
        self.force_excv_id = force_excv_id

        # Find all H5/HDF5 files
        self.file_list = sorted(
            glob.glob(str(self.data_dir / "**" / "*.h5"), recursive=True) +
            glob.glob(str(self.data_dir / "**" / "*.hdf5"), recursive=True)
        )
        if len(self.file_list) == 0:
            candidates = list(self.data_dir.glob("**/*.h5")) + list(self.data_dir.glob("**/*.hdf5"))
            self.file_list = [str(c) for c in candidates]
        print(f"[{split}] Found {len(self.file_list)} episode(s)")

        # ── Stratified train/val split by excavator type ──
        # Group files by excavator, shuffle each group, then split proportionally.
        # This ensures every excavator type appears in both train and val.
        import random
        rng = random.Random(42)

        groups = {}  # excv_id -> list of (global_idx, filepath)
        for fidx, fpath in enumerate(self.file_list):
            eid = self._parse_excavator_id(fpath)
            groups.setdefault(eid, []).append((fidx, fpath))

        train_indices, val_indices = [], []
        for eid, items in groups.items():
            rng.shuffle(items)
            n_train = max(1, int(len(items) * train_split))
            train_indices.extend([x[0] for x in items[:n_train]])
            val_indices.extend([x[0] for x in items[n_train:]])

        self.train_files = set(train_indices)
        self.val_files   = set(val_indices)

        if split == "train":
            my_files = self.train_files
        elif split == "val":
            my_files = self.val_files
        else:
            my_files = set(range(len(self.file_list)))

        # Report split composition
        train_by_excv = {}
        val_by_excv = {}
        for fidx in self.train_files:
            eid = self._parse_excavator_id(self.file_list[fidx])
            train_by_excv[eid] = train_by_excv.get(eid, 0) + 1
        for fidx in self.val_files:
            eid = self._parse_excavator_id(self.file_list[fidx])
            val_by_excv[eid] = val_by_excv.get(eid, 0) + 1
        all_eids = sorted(set(list(train_by_excv.keys()) + list(val_by_excv.keys())))
        print(f"[{split}] Split: train={{{', '.join(f'{k}:{train_by_excv.get(k,0)}' for k in all_eids)}}}, "
              f"val={{{', '.join(f'{k}:{val_by_excv.get(k,0)}' for k in all_eids)}}}")

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
            # Target = NEXT frame's absolute joint angles (always use qpos[1:])
            # Different data sources have inconsistent 'action' keys, so we unify.
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
        if self.force_excv_id is not None:
            excv_id = self.force_excv_id
        return {
            "_h5": f,                          # keep file handle alive
            "mains_raw": mains_raw,            # h5py Dataset, lazy access
            "elevations_raw": elevations_raw,  # h5py Dataset, lazy access
            "qpos": qpos,
            "action": action,
            "excavator_id": excv_id,           # int: 0=75, 1=306, 2=490
        }

    def _sample_night_aug(self):
        """Sample clip-consistent nighttime augmentation params (RGB only).

        Returns a dict of augmentation params, or None if no aug applied.
        All frames in the same clip share the same params.
        """
        rng = np.random
        aug = {}

        # Gamma perturbation: 0.3→1.0 simulates varied lighting
        if rng.random() < 0.6:
            aug["gamma"] = rng.uniform(0.3, 1.0)

        # Brightness/contrast jitter
        if rng.random() < 0.5:
            aug["alpha"] = rng.uniform(0.5, 1.5)       # contrast
            aug["beta"] = rng.uniform(-40, 40)           # brightness shift

        # Gaussian blur (simulates motion blur / defocus)
        if rng.random() < 0.3:
            aug["blur_ksize"] = rng.choice([3, 5])
            aug["blur_sigma"] = rng.uniform(0.5, 2.0)

        # Gaussian noise (sensor noise in low light)
        if rng.random() < 0.4:
            aug["noise_std"] = rng.uniform(2, 15)

        # Simulated low-light: darken + add slight noise
        if rng.random() < 0.3:
            aug["lowlight_factor"] = rng.uniform(0.15, 0.5)

        return aug if aug else None

    def _apply_night_aug(self, img_uint8: np.ndarray, aug: dict) -> np.ndarray:
        """Apply nighttime augmentations to a uint8 BGR image."""
        img = img_uint8.astype(np.float32)

        if "lowlight_factor" in aug:
            img = img * aug["lowlight_factor"]

        if "gamma" in aug:
            img = 255.0 * ((img / 255.0) ** aug["gamma"])
            img = img.clip(0, 255)

        if "alpha" in aug:
            img = aug["alpha"] * img + aug["beta"]
            img = img.clip(0, 255)

        if "blur_ksize" in aug:
            # Apply only if ksize > 1
            k = aug["blur_ksize"]
            img = cv2.GaussianBlur(img, (k, k), aug["blur_sigma"])

        if "noise_std" in aug:
            noise = np.random.randn(*img.shape).astype(np.float32) * aug["noise_std"]
            img = img + noise
            img = img.clip(0, 255)

        return img.astype(np.uint8)

    def _preprocess_image(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img.transpose(2, 0, 1)

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

        # Night augmentation (RGB only, clip-consistent, training only)
        night_aug = self._sample_night_aug() if self.split == "train" else None

        for t in range(T):
            i = start + t
            rgb_frame = ep["mains_raw"][i]
            if night_aug is not None:
                rgb_frame = self._apply_night_aug(rgb_frame, night_aug)
            rgb_seq[t] = self._preprocess_image(rgb_frame)
            elev_seq[t] = self._preprocess_image(ep["elevations_raw"][i])

        qpos_seq = ep["qpos"][start:end]         # [T, 4]
        # Target: K future absolute qpos values (action_chunk steps)
        #   action[i] ≈ qpos[i+1], so action[start+T-1] = first post-window qpos
        tgt_start = start + T - 1
        targets = ep["action"][tgt_start : tgt_start + K]  # [K, 4]
        excv_id = ep["excavator_id"]

        return {
            "rgb": torch.from_numpy(rgb_seq),
            "elevation": torch.from_numpy(elev_seq),
            "qpos": torch.from_numpy(qpos_seq.copy()),
            "action": torch.from_numpy(targets.copy()),     # [K, 4]
            "excavator_id": torch.tensor(excv_id, dtype=torch.long),
            # Local episode index is stable for this dataset split and collates
            # as a tensor without changing the existing batch fields.
            "episode_id": torch.tensor(fidx, dtype=torch.long),
        }
