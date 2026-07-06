from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
 #   ConcatDataset,
 #   DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
    random_split,
)


class ARDAEDataset(Dataset):
    """
    ARDAE 학습용 데이터셋.

    MLP ARDAE는 [N, input_dim], U-Net ARDAE는 [N, C, H, W] 형태를 사용한다.
    """

    def __init__(
        self,
        data,
        input_dim=None,
        normalize=None,
        flatten=True,
        image_shape=None,
        dtype=torch.float32,
    ):
        from .dataset import preprocess_ardae_data

        self.x = preprocess_ardae_data(
            data=data,
            input_dim=input_dim,
            normalize=normalize,
            flatten=flatten,
            image_shape=image_shape,
            dtype=dtype,
        )

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index]


class ImagePatchDataset(Dataset):
    """
    이미지 파일에서 patch를 lazy하게 읽는 데이터셋.

    모든 patch를 미리 numpy array로 만들지 않고, __getitem__ 때 해당 이미지 하나만
    열어서 필요한 patch만 잘라낸다.
    """

    def __init__(
        self,
        image_paths,
        patch_size,
        stride,
        channels=1,
        max_patches_per_image=None,
        seed=0,
        flatten=False,
        dtype=torch.float32,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.channels = int(channels)
        self.max_patches_per_image = max_patches_per_image
        self.seed = int(seed)
        self.flatten = bool(flatten)
        self.dtype = dtype

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3.")
        if len(self.image_paths) == 0:
            raise ValueError("No image files found.")

        self.index = []
        for image_idx, path in enumerate(self.image_paths):
            width, height = self._read_image_size(path)
            coords = self._make_patch_coords(
                width=width,
                height=height,
                image_idx=image_idx,
            )
            self.index.extend((image_idx, y, x) for y, x in coords)

        if len(self.index) == 0:
            raise ValueError(
                "No patches could be extracted. Check patch_size/stride and image sizes."
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        image_idx, y, x = self.index[index]
        image = self._load_image(self.image_paths[image_idx])
        patch = image[
            y : y + self.patch_size,
            x : x + self.patch_size,
        ]

        if self.channels == 1:
            tensor = torch.as_tensor(patch, dtype=self.dtype).unsqueeze(0)
        else:
            tensor = torch.as_tensor(patch, dtype=self.dtype).permute(2, 0, 1)

        tensor = tensor / 255.0
        if self.flatten:
            tensor = tensor.reshape(-1)
        return tensor.contiguous()

    @staticmethod
    def _open_image(path):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "ImagePatchDataset requires Pillow. Install it with `pip install pillow`."
            ) from exc

        return Image.open(path)

    def _read_image_size(self, path):
        with self._open_image(path) as image:
            return image.size

    def _load_image(self, path):
        mode = "L" if self.channels == 1 else "RGB"
        with self._open_image(path) as image:
            return np.asarray(image.convert(mode)).copy()

    def _make_patch_coords(self, width, height, image_idx):
        if width < self.patch_size or height < self.patch_size:
            return []

        ys = range(0, height - self.patch_size + 1, self.stride)
        xs = range(0, width - self.patch_size + 1, self.stride)
        coords = [(y, x) for y in ys for x in xs]

        if (
            self.max_patches_per_image is not None
            and len(coords) > self.max_patches_per_image
        ):
            rng = np.random.default_rng(self.seed + image_idx)
            selected = rng.choice(
                len(coords),
                size=int(self.max_patches_per_image),
                replace=False,
            )
            coords = [coords[i] for i in np.sort(selected)]

        return coords


class StreamingImagePatchDataset(IterableDataset):
    """
    이미지 단위로 순회하며 patch를 yield하는 빠른 데이터셋.

    기존 인터페이스는 유지하되, image_paths는 이제 이미지 파일 경로가 아니라
    이미지 하나당 하나씩 저장된 .npy 파일 경로를 받는다.

    기대하는 npy shape:
        channels=1: [H, W] 또는 [H, W, 1] 또는 [1, H, W]
        channels=3: [H, W, 3] 또는 [3, H, W]

    권장 dtype:
        uint8, range [0, 255]

    float npy를 넣는 경우:
        이미 [0, 1]로 normalize되어 있다고 가정한다.
    """

    def __init__(
        self,
        image_paths,
        patch_size,
        stride,
        channels=1,
        max_patches_per_image=None,
        seed=0,
        flatten=False,
        dtype=torch.float32,
        shuffle_images=True,
        shuffle_patches=True,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.channels = int(channels)
        self.max_patches_per_image = max_patches_per_image
        self.seed = int(seed)
        self.flatten = bool(flatten)
        self.dtype = dtype
        self.shuffle_images = bool(shuffle_images)
        self.shuffle_patches = bool(shuffle_patches)

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3.")
        if len(self.image_paths) == 0:
            raise ValueError("No npy files found.")

        self._length = 0
        for image_idx, path in enumerate(self.image_paths):
            height, width = self._read_npy_hw(path)
            coords = self._make_patch_coords(
                width=width,
                height=height,
                image_idx=image_idx,
            )
            self._length += len(coords)

        if self._length == 0:
            raise ValueError(
                "No patches could be extracted. Check patch_size/stride and npy image sizes."
            )

    def __len__(self):
        return self._length

    def __iter__(self):
        worker = get_worker_info()

        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        rng = np.random.default_rng(self.seed + worker_id)

        image_indices = np.arange(len(self.image_paths))

        if self.shuffle_images:
            rng.shuffle(image_indices)

        # 각 worker가 서로 다른 이미지 subset을 담당
        image_indices = image_indices[worker_id::num_workers]

        for image_idx in image_indices:
            image_idx = int(image_idx)
            path = self.image_paths[image_idx]

            image = self._load_npy_image(path)
            height, width = image.shape[:2]

            coords = self._make_patch_coords(
                width=width,
                height=height,
                image_idx=image_idx,
            )

            if self.shuffle_patches:
                rng.shuffle(coords)

            for y, x in coords:
                patch = image[
                    y : y + self.patch_size,
                    x : x + self.patch_size,
                ]

                tensor = self._patch_to_tensor(patch)

                if self.flatten:
                    tensor = tensor.reshape(-1)

                yield tensor.contiguous()

    def _read_npy_hw(self, path):
        arr = np.load(path, mmap_mode="r")
        shape = arr.shape

        if self.channels == 1:
            if len(shape) == 2:
                height, width = shape
                return int(height), int(width)

            if len(shape) == 3:
                # [H, W, 1]
                if shape[-1] == 1:
                    height, width = shape[:2]
                    return int(height), int(width)

                # [1, H, W]
                if shape[0] == 1:
                    height, width = shape[1:]
                    return int(height), int(width)

            raise ValueError(
                f"Expected grayscale npy shape [H, W], [H, W, 1], or [1, H, W], "
                f"but got {shape} from {path}"
            )

        if self.channels == 3:
            if len(shape) == 3:
                # [H, W, 3]
                if shape[-1] == 3:
                    height, width = shape[:2]
                    return int(height), int(width)

                # [3, H, W]
                if shape[0] == 3:
                    height, width = shape[1:]
                    return int(height), int(width)

            raise ValueError(
                f"Expected RGB npy shape [H, W, 3] or [3, H, W], "
                f"but got {shape} from {path}"
            )

        raise ValueError(f"Unsupported channels={self.channels}")

    def _load_npy_image(self, path):
        image = np.load(path)

        if self.channels == 1:
            if image.ndim == 2:
                # [H, W]
                return image

            if image.ndim == 3:
                # [H, W, 1] -> [H, W]
                if image.shape[-1] == 1:
                    return image[..., 0]

                # [1, H, W] -> [H, W]
                if image.shape[0] == 1:
                    return image[0]

            raise ValueError(
                f"Expected grayscale npy shape [H, W], [H, W, 1], or [1, H, W], "
                f"but got {image.shape} from {path}"
            )

        if self.channels == 3:
            if image.ndim == 3:
                # [H, W, 3]
                if image.shape[-1] == 3:
                    return image

                # [3, H, W] -> [H, W, 3]
                if image.shape[0] == 3:
                    return np.transpose(image, (1, 2, 0)).copy()

            raise ValueError(
                f"Expected RGB npy shape [H, W, 3] or [3, H, W], "
                f"but got {image.shape} from {path}"
            )

        raise ValueError(f"Unsupported channels={self.channels}")

    def _patch_to_tensor(self, patch):
        # npy가 mmap 또는 non-contiguous slice일 수 있으므로 copy()로 안전하게 tensor화
        if self.channels == 1:
            tensor = torch.as_tensor(patch.copy(), dtype=self.dtype).unsqueeze(0)
        else:
            tensor = torch.as_tensor(patch.copy(), dtype=self.dtype)
            tensor = tensor.permute(2, 0, 1)

        # uint8/int 계열이면 [0, 255]라고 보고 normalize
        # float 계열이면 이미 [0, 1]이라고 가정
        if np.issubdtype(patch.dtype, np.integer):
            tensor = tensor / 255.0

        return tensor

    def _make_patch_coords(self, width, height, image_idx):
        if width < self.patch_size or height < self.patch_size:
            return []

        ys = range(0, height - self.patch_size + 1, self.stride)
        xs = range(0, width - self.patch_size + 1, self.stride)
        coords = [(y, x) for y in ys for x in xs]

        if (
            self.max_patches_per_image is not None
            and len(coords) > self.max_patches_per_image
        ):
            rng = np.random.default_rng(self.seed + image_idx)
            selected = rng.choice(
                len(coords),
                size=int(self.max_patches_per_image),
                replace=False,
            )
            coords = [coords[i] for i in np.sort(selected)]

        return coords
    
"""
class StreamingImagePatchDataset(IterableDataset):
"""
"""
    이미지 단위로 순회하며 patch를 yield하는 빠른 데이터셋.

    map-style ImagePatchDataset보다 shuffle 자유도는 낮지만, 이미지 하나를 한 번만
    decode한 뒤 여러 patch를 뽑기 때문에 큰 PNG/JPEG 폴더 학습에서 훨씬 빠르다.
    """
"""
    def __init__(
        self,
        image_paths,
        patch_size,
        stride,
        channels=1,
        max_patches_per_image=None,
        seed=0,
        flatten=False,
        dtype=torch.float32,
        shuffle_images=True,
        shuffle_patches=True,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.channels = int(channels)
        self.max_patches_per_image = max_patches_per_image
        self.seed = int(seed)
        self.flatten = bool(flatten)
        self.dtype = dtype
        self.shuffle_images = bool(shuffle_images)
        self.shuffle_patches = bool(shuffle_patches)

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.channels not in (1, 3):
            raise ValueError("channels must be 1 or 3.")
        if len(self.image_paths) == 0:
            raise ValueError("No image files found.")

        probe = ImagePatchDataset(
            image_paths=self.image_paths,
            patch_size=self.patch_size,
            stride=self.stride,
            channels=self.channels,
            max_patches_per_image=self.max_patches_per_image,
            seed=self.seed,
            flatten=self.flatten,
            dtype=self.dtype,
        )
        self._length = len(probe)

    def __len__(self):
        return self._length

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        rng = np.random.default_rng(self.seed + worker_id)
        image_indices = np.arange(len(self.image_paths))
        if self.shuffle_images:
            rng.shuffle(image_indices)
        image_indices = image_indices[worker_id::num_workers]

        for image_idx in image_indices:
            path = self.image_paths[int(image_idx)]
            image = self._load_image(path)
            height, width = image.shape[:2]
            coords = self._make_patch_coords(
                width=width,
                height=height,
                image_idx=int(image_idx),
            )
            if self.shuffle_patches:
                rng.shuffle(coords)

            for y, x in coords:
                patch = image[
                    y : y + self.patch_size,
                    x : x + self.patch_size,
                ]
                if self.channels == 1:
                    tensor = torch.as_tensor(patch, dtype=self.dtype).unsqueeze(0)
                else:
                    tensor = torch.as_tensor(patch, dtype=self.dtype).permute(2, 0, 1)

                tensor = tensor / 255.0
                if self.flatten:
                    tensor = tensor.reshape(-1)
                yield tensor.contiguous()

    @staticmethod
    def _open_image(path):
        return ImagePatchDataset._open_image(path)

    def _load_image(self, path):
        mode = "L" if self.channels == 1 else "RGB"
        with self._open_image(path) as image:
            return np.asarray(image.convert(mode)).copy()

    def _make_patch_coords(self, width, height, image_idx):
        if width < self.patch_size or height < self.patch_size:
            return []

        ys = range(0, height - self.patch_size + 1, self.stride)
        xs = range(0, width - self.patch_size + 1, self.stride)
        coords = [(y, x) for y in ys for x in xs]

        if (
            self.max_patches_per_image is not None
            and len(coords) > self.max_patches_per_image
        ):
            rng = np.random.default_rng(self.seed + image_idx)
            selected = rng.choice(
                len(coords),
                size=int(self.max_patches_per_image),
                replace=False,
            )
            coords = [coords[i] for i in np.sort(selected)]

        return coords
"""
