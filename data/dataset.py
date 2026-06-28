from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ARDAEDataset(Dataset):
    """
    ARDAE 학습용 데이터셋.

    모델 입력은 [B, input_dim] float tensor여야 하므로, 들어온 데이터를
    [N, input_dim] 형태로 정리해서 보관한다.
    """

    def __init__(
        self,
        data,
        input_dim=None,
        normalize=None,
        flatten=True,
        dtype=torch.float32,
    ):
        self.x = preprocess_ardae_data(
            data=data,
            input_dim=input_dim,
            normalize=normalize,
            flatten=flatten,
            dtype=dtype,
        )

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        return self.x[index]


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


def preprocess_ardae_data(
    data,
    input_dim=None,
    normalize=None,
    flatten=True,
    dtype=torch.float32,
):
    """
    원본 데이터를 ARDAE 입력 형태인 [N, input_dim] float tensor로 변환한다.

    Args:
        data: tensor, numpy array, list, 또는 파일에서 읽은 배열.
        input_dim: feature 차원. None이면 마지막 차원을 사용한다.
        normalize: None, "standard", "minmax", 또는 "zero_one".
        flatten: True이면 첫 번째 차원만 sample 차원으로 남기고 나머지를 펼친다.
        dtype: 반환 tensor dtype.
    """
    x = torch.as_tensor(data, dtype=dtype)

    if x.ndim == 0:
        raise ValueError("data must have at least one sample dimension")

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
    if method == "standard":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True).clamp_min(eps)
        return (x - mean) / std

    if method in {"minmax", "zero_one"}:
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        return (x - x_min) / (x_max - x_min).clamp_min(eps)

    raise ValueError(f"Unknown normalize method: {method}")


def make_ardae_dataset(
    data,
    input_dim=None,
    normalize=None,
    flatten=True,
    dtype=torch.float32,
):
    return ARDAEDataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        dtype=dtype,
    )


def make_ardae_dataloader(
    data,
    input_dim=None,
    batch_size=128,
    shuffle=True,
    normalize=None,
    flatten=True,
    dtype=torch.float32,
    **loader_kwargs,
):
    dataset = make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
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
    dtype=torch.float32,
    seed=0,
    **loader_kwargs,
):
    dataset = make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
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


def load_ardae_dataset(
    path,
    key=None,
    input_dim=None,
    normalize=None,
    flatten=True,
    dtype=torch.float32,
):
    data = load_array(path, key=key)
    return make_ardae_dataset(
        data=data,
        input_dim=input_dim,
        normalize=normalize,
        flatten=flatten,
        dtype=dtype,
    )
