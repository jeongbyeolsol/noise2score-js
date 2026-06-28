import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_nonlinear_func, ConvBlock, UpBlock, DownBlock



class MLP(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_dim=8,
        output_dim=2,
        nonlinearity="silu",
        num_hidden_layers=1,
        use_nonlinearity_output=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.num_hidden_layers = num_hidden_layers
        self.use_nonlinearity_output = use_nonlinearity_output

        self.act = get_nonlinear_func(nonlinearity)

        layers = []
        for i in range(num_hidden_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)

        final_in_dim = input_dim if num_hidden_layers == 0 else hidden_dim
        self.fc = nn.Linear(final_in_dim, output_dim)

    def forward(self, input):
        batch_size = input.size(0)
        x = input.view(batch_size, self.input_dim)

        hidden = x

        for layer in self.layers:
            hidden = self.act(layer(hidden))

        output = self.fc(hidden)

        if self.use_nonlinearity_output:
            output = self.act(output)

        return output

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        nonlinearity="silu",
        use_norm=True,
        use_noise_level=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.nonlinearity = nonlinearity
        self.use_norm = use_norm
        self.use_noise_level = use_noise_level

        # ARDAE의 MLP(input_dim + 1)처럼,
        # 이미지 입력에 noise level channel 하나를 추가로 붙일 수 있게 한다.
        actual_in_channels = in_channels + 1 if use_noise_level else in_channels

        channels = [base_channels * m for m in channel_mults]

        self.inc = ConvBlock(
            actual_in_channels,
            channels[0],
            nonlinearity=nonlinearity,
            use_norm=use_norm,
        )

        self.downs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(
                DownBlock(
                    channels[i],
                    channels[i + 1],
                    nonlinearity=nonlinearity,
                    use_norm=use_norm,
                )
            )

        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(channels))

        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            out_ch = reversed_channels[i + 1]

            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    nonlinearity=nonlinearity,
                    use_norm=use_norm,
                )
            )

        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def _append_noise_level_channel(self, x, noise_level):
        """
        x: [B, C, H, W]
        noise_level:
            float
            or scalar tensor
            or [B]
            or [B, 1]
            or [B, 1, 1, 1]

        return:
            [B, C + 1, H, W]
        """
        batch_size, _, height, width = x.shape

        if noise_level is None:
            raise ValueError("use_noise_level=True이면 forward(x, noise_level)이 필요합니다.")

        if not torch.is_tensor(noise_level):
            noise_level = x.new_tensor(noise_level)

        noise_level = noise_level.to(device=x.device, dtype=x.dtype)

        if noise_level.dim() == 0:
            noise_level = noise_level.view(1, 1, 1, 1).expand(batch_size, 1, height, width)
        elif noise_level.dim() == 1:
            noise_level = noise_level.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
        elif noise_level.dim() == 2:
            noise_level = noise_level.view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width)
        elif noise_level.dim() == 4:
            noise_level = noise_level.expand(batch_size, 1, height, width)
        else:
            raise ValueError(f"Unsupported noise_level shape: {tuple(noise_level.shape)}")

        return torch.cat([x, noise_level], dim=1)

    def forward(self, x, noise_level=None):
        """
        x: [B, C, H, W]
        noise_level: optional noise condition

        return:
            output: [B, out_channels, H, W]
        """
        if x.dim() != 4:
            raise ValueError(f"UNet expects x shape [B, C, H, W], but got {tuple(x.shape)}")

        if self.use_noise_level:
            x = self._append_noise_level_channel(x, noise_level)

        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # 마지막 feature는 bottleneck이므로 skip 연결에서 제외
        skips = skips[:-1]
        skips = list(reversed(skips))

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        x = self.outc(x)
        return x
    


if __name__ == "__main__":
   
    print('MLP test')
    model = MLP(input_dim=2, hidden_dim=8, output_dim=2, nonlinearity='relu', num_hidden_layers=1)
    x = torch.rand(1, 2)
    y = model(x)
    print(y.shape)
    print('MLP test END')
    
    print('\n============================\n')
    
    print('UNet test')
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        use_noise_level=True,
    )

    x = torch.randn(4, 1, 64, 64)
    sigma = torch.full((4, 1), 0.1)

    y = model(x, sigma)
    print(y.shape)
    print('UNet test END')