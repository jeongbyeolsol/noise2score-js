from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    IterableDataset,
    get_worker_info,
    random_split,
)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


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

    map-style ImagePatchDataset보다 shuffle 자유도는 낮지만, 이미지 하나를 한 번만
    decode한 뒤 여러 patch를 뽑기 때문에 큰 PNG/JPEG 폴더 학습에서 훨씬 빠르다.
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


def load_array(path, key=None):
    """
    csv/txt/tsv, npy/npz, pt/pth 파일을 torch.Tensor로 읽는다.

    npz 또는 dict 형태의 pt/pth 파일은 key가 필요할 수 있다.
    key가 없고 항목이 하나뿐이면 그 항목을 자동으로 사용한다.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
        return torch.as_tensor(array)

    if suffix == ".npy":
        return torch.as_tensor(np.load(path))

    if suffix == ".npz":
        archive = np.load(path)
        if key is None:
            if len(archive.files) != 1:
                raise ValueError(f"key must be given for {path}; found keys: {archive.files}")
            key = archive.files[0]
        return torch.as_tensor(archive[key])

    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            if key is None:
                if len(obj) != 1:
                    raise ValueError(f"key must be given for {path}; found keys: {list(obj.keys())}")
                key = next(iter(obj))
            return torch.as_tensor(obj[key])
        return torch.as_tensor(obj)

    raise ValueError(f"Unsupported data file type: {suffix}")


def find_image_paths(path, recursive=False):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [path]
        with path.open("r", encoding="utf-8") as f:
            base_dir = path.parent
            image_paths = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                item = Path(line)
                if not item.is_absolute():
                    item = base_dir / item
                image_paths.append(item)
            return image_paths

    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            p for p in path.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"Image path not found: {path}")


def _normalize_image_shape(image_shape):
    if image_shape is None:
        return None
    image_shape = tuple(int(v) for v in image_shape)
    if len(image_shape) == 2:
        image_shape = (1, *image_shape)
    if len(image_shape) != 3:
        raise ValueError("image_shape must be [H, W] or [C, H, W]")
    return image_shape


def _reshape_to_image(x, image_shape):
    image_shape = _normalize_image_shape(image_shape)
    if image_shape is None:
        return x

    c, h, w = image_shape
    flat_dim = c * h * w

    if x.ndim == 2:
        if x.size(1) != flat_dim:
            raise ValueError(f"Flat data dim {x.size(1)} != image_shape product {flat_dim}")
        return x.view(x.size(0), c, h, w)

    if x.ndim == 3:
        # [N, H, W] -> [N, 1, H, W]
        if c == 1 and tuple(x.shape[1:]) == (h, w):
            return x.unsqueeze(1)
        # [N, C, L] 같은 애매한 형태는 명시적으로 거부
        raise ValueError(f"3D data shape {tuple(x.shape)} cannot be reshaped to [N, {c}, {h}, {w}]")

    if x.ndim == 4:
        if tuple(x.shape[1:]) == image_shape:
            return x
        # NHWC -> NCHW도 자주 나오므로 지원
        if tuple(x.shape[1:]) == (h, w, c):
            return x.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"4D data shape {tuple(x.shape)} does not match [N, {c}, {h}, {w}]")

    raise ValueError(f"Unsupported image data shape: {tuple(x.shape)}")


def preprocess_ardae_data(
    data,
    input_dim=None,
    normalize=None,
    flatten=True,
    image_shape=None,
    dtype=torch.float32,
):
    """
    원본 데이터를 ARDAE 입력 형태로 변환한다.

    Args:
        data: tensor, numpy array, list, 또는 파일에서 읽은 배열.
        input_dim: MLP 입력 feature 차원.
        normalize: None, "standard", "minmax", 또는 "zero_one".
        flatten: True이면 첫 번째 차원만 sample 차원으로 남기고 나머지를 펼친다.
        image_shape: U-Net용 [C, H, W] 또는 [H, W]. 주어지면 [N, C, H, W]로 변환한다.
        dtype: 반환 tensor dtype.
    """
    x = torch.as_tensor(data, dtype=dtype)

    if x.ndim == 0:
        raise ValueError("data must have at least one sample dimension")

    if image_shape is not None and not flatten:
        x = _reshape_to_image(x, image_shape)
    else:
        if flatten:
            x = x.view(x.size(0), -1)
        elif x.ndim == 1:
            x = x.view(-1, 1)

        if input_dim is not None:
            x = x.view(-1, int(input_dim))
        elif x.ndim != 2:
            x = x.view(x.size(0), -1)

    if normalize is not None:
        x = normalize_tensor(x, method=normalize)

    if not torch.isfinite(x).all():
        raise ValueError("data contains NaN or Inf values")

    return x.contiguous()


def normalize_tensor(x, method="standard", eps=1e-8):
    # MLP [N, D]뿐 아니라 이미지 [N, C, H, W]도 sample 차원 기준으로 정규화한다.
    reduce_dims = (0,)

    if method == "standard":
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True).clamp_min(eps)
        return (x - mean) / std

    if method in {"minmax", "zero_one"}:
        x_min = x.amin(dim=reduce_dims, keepdim=True)
        x_max = x.amax(dim=reduce_dims, keepdim=True)
        return (x - x_min) / (x_max - x_min).clamp_min(eps)

    raise ValueError(f"Unknown normalize method: {method}")


def make_ardae_dataset(
    data,
    input_dim=None,
    normalize=None,
    flatten=True,
    image_shape=None,
    dtype=torch.float32,
):
    return ARDAEDataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        image_shape=image_shape,
        dtype=dtype,
    )


def make_ardae_dataloader(
    data,
    input_dim=None,
    batch_size=128,
    shuffle=True,
    normalize=None,
    flatten=True,
    image_shape=None,
    dtype=torch.float32,
    **loader_kwargs,
):
    dataset = make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        image_shape=image_shape,
        dtype=dtype,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)


def make_ardae_dataloaders(
    data,
    input_dim=None,
    batch_size=128,
    val_ratio=0.1,
    normalize=None,
    flatten=True,
    image_shape=None,
    dtype=torch.float32,
    seed=0,
    **loader_kwargs,
):
    dataset = make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        image_shape=image_shape,
        dtype=dtype,
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def make_image_patch_dataloaders(
    image_paths,
    patch_size,
    stride,
    channels=1,
    max_patches_per_image=None,
    input_dim=None,
    batch_size=128,
    val_ratio=0.1,
    seed=0,
    flatten=False,
    dtype=torch.float32,
    split_by_image=True,
    **loader_kwargs,
):
    if input_dim is not None:
        expected_dim = channels * patch_size * patch_size
        if int(input_dim) != expected_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match "
                f"channels*patch_size^2={expected_dim}."
            )

    image_paths = list(image_paths)

    if len(image_paths) == 0:
        raise ValueError("image_paths is empty.")

    if split_by_image:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(image_paths))
        rng.shuffle(indices)

        val_image_count = int(len(image_paths) * val_ratio)

        if val_ratio > 0:
            val_image_count = max(1, val_image_count)

        train_image_count = len(image_paths) - val_image_count

        if train_image_count <= 0:
            raise ValueError(
                "Train image split is empty. Reduce val_ratio or provide more images."
            )

        val_indices = indices[:val_image_count]
        train_indices = indices[val_image_count:]

        train_paths = [image_paths[i] for i in train_indices]
        val_paths = [image_paths[i] for i in val_indices]

        train_dataset = ImagePatchDataset(
            image_paths=train_paths,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            max_patches_per_image=max_patches_per_image,
            seed=seed,
            flatten=flatten,
            dtype=dtype,
        )

        val_dataset = ImagePatchDataset(
            image_paths=val_paths,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            max_patches_per_image=max_patches_per_image,
            seed=seed + 1,
            flatten=flatten,
            dtype=dtype,
        )

        full_dataset = ConcatDataset([train_dataset, val_dataset])

    else:
        full_dataset = ImagePatchDataset(
            image_paths=image_paths,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            max_patches_per_image=max_patches_per_image,
            seed=seed,
            flatten=flatten,
            dtype=dtype,
        )

        n = len(full_dataset)
        val_size = int(n * val_ratio)

        if val_ratio > 0:
            val_size = max(1, val_size)

        train_size = n - val_size

        if train_size <= 0:
            raise ValueError(
                "Train dataset is empty. Reduce val_ratio or provide more data."
            )

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, full_dataset


def split_image_paths(image_paths, val_ratio=0.1, seed=0):
    image_paths = list(image_paths)
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)

    val_count = int(len(image_paths) * val_ratio)
    if val_ratio > 0:
        val_count = max(1, val_count)

    train_count = len(image_paths) - val_count
    if train_count <= 0:
        raise ValueError("Train image split is empty. Reduce val_ratio.")

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_paths = [image_paths[int(i)] for i in train_indices]
    val_paths = [image_paths[int(i)] for i in val_indices]
    return train_paths, val_paths


def make_streaming_image_patch_dataloaders(
    image_paths,
    patch_size,
    stride,
    channels=1,
    max_patches_per_image=None,
    input_dim=None,
    batch_size=128,
    val_ratio=0.1,
    seed=0,
    flatten=False,
    dtype=torch.float32,
    **loader_kwargs,
):
    if input_dim is not None:
        expected_dim = channels * patch_size * patch_size
        if int(input_dim) != expected_dim:
            raise ValueError(
                f"input_dim={input_dim} does not match "
                f"channels*patch_size^2={expected_dim}."
            )

    train_paths, val_paths = split_image_paths(
        image_paths=image_paths,
        val_ratio=val_ratio,
        seed=seed,
    )

    train_dataset = StreamingImagePatchDataset(
        image_paths=train_paths,
        patch_size=patch_size,
        stride=stride,
        channels=channels,
        max_patches_per_image=max_patches_per_image,
        seed=seed,
        flatten=flatten,
        dtype=dtype,
        shuffle_images=True,
        shuffle_patches=True,
    )

    if len(val_paths) > 0:
        val_dataset = StreamingImagePatchDataset(
            image_paths=val_paths,
            patch_size=patch_size,
            stride=stride,
            channels=channels,
            max_patches_per_image=max_patches_per_image,
            seed=seed + 100000,
            flatten=flatten,
            dtype=dtype,
            shuffle_images=False,
            shuffle_patches=False,
        )
    else:
        val_dataset = []

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def load_ardae_dataset(
    path,
    key=None,
    input_dim=None,
    normalize=None,
    flatten=True,
    image_shape=None,
    dtype=torch.float32,
):
    data = load_array(path, key=key)
    return make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        image_shape=image_shape,
        dtype=dtype,
    )
