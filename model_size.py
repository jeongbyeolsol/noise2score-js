import torch

ckpt_path = "./checkpoints/ardae_unet/poisson_lam001_005_smoothing/best_model.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

if "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

total = 0

for name, tensor in state_dict.items():
    if torch.is_tensor(tensor):
        n = tensor.numel()
        total += n
        print(f"{name:70s} {n:,} {tuple(tensor.shape)}")

print("-" * 90)
print(f"Total parameters in state_dict: {total:,}")

# Total parameters: 468,163