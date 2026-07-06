import torch

from models.ardae import ARDAE
from utils.image_shape import as_tuple, infer_image_shape_from_data, normalize_image_shape_arg


def load_ardae_from_checkpoint(path, input_dim, device, clean=None, image_shape_override=None):
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    backbone = ckpt_args.get("backbone", "mlp")

    image_shape = None
    if backbone == "unet":
        if clean is None:
            image_shape = normalize_image_shape_arg(ckpt_args.get("image_shape"))
        else:
            image_shape = infer_image_shape_from_data(
                clean,
                input_dim,
                ckpt_args,
                override=image_shape_override,
            )

    model = ARDAE(
        input_dim=input_dim,
        h_dim=ckpt_args.get("h_dim", 1000),
        noise_param=ckpt_args.get("noise_param", 0.1),
        num_hidden_layers=ckpt_args.get("num_hidden_layers", 1),
        nonlinearity=ckpt_args.get("nonlinearity", "tanh"),
        noise_type=ckpt_args.get("noise_type", "gaussian"),
        use_metric=False,
        backbone=backbone,
        image_shape=image_shape,
        base_channels=ckpt_args.get("base_channels", 64),
        channel_mults=as_tuple(ckpt_args.get("channel_mults"), default=(1, 2, 4, 8)),
        use_norm=not ckpt_args.get("no_norm", False),
        use_gaussian_smoothing=ckpt_args.get("use_gaussian_smoothing", False),
    ).to(device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    return model, backbone, image_shape
