from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
#    Dataset,
#    IterableDataset,
#    get_worker_info,
    random_split,
)
from .classes import (
    ARDAEDataset,
    ImagePatchDataset,
    StreamingImagePatchDataset,
)
from utils import load_array, normalize_tensor


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}



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
