import torch

'''
miscellaneous functions: learning
'''

def expand_tensor(input, sample_size, do_unsqueeze):
    """
    설명:
        배치 텐서의 각 샘플을 sample_size번 복제한 뒤,
        복제 차원을 배치 차원으로 합친 flattened tensor를 함께 반환한다.

        do_unsqueeze=True인 경우:
            입력 텐서가 [B, ...] 형태라고 가정한다.
            먼저 dim=1 위치에 sample dimension을 추가하여 [B, 1, ...]로 만든 뒤,
            이를 [B, sample_size, ...]로 확장한다.
            이후 [B * sample_size, ...] 형태로 flatten한다.

        do_unsqueeze=False인 경우:
            입력 텐서가 이미 [B, 1, ...] 형태라고 가정한다.
            dim=1의 singleton dimension을 sample_size로 확장하여
            [B, sample_size, ...]로 만든 뒤,
            이를 [B * sample_size, ...] 형태로 flatten한다.

    Args:
        input (torch.Tensor):
            확장할 입력 텐서.
            do_unsqueeze=True이면 [B, ...] 형태,
            do_unsqueeze=False이면 [B, 1, ...] 형태여야 한다.

        sample_size (int):
            각 배치 샘플을 복제할 횟수.

        do_unsqueeze (bool):
            True이면 dim=1에 새 sample dimension을 추가한 뒤 확장한다.
            False이면 input의 dim=1이 이미 크기 1이라고 가정하고 확장한다.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            input_expanded:
                sample dimension이 유지된 확장 텐서.
                do_unsqueeze=True: [B, sample_size, ...]
                do_unsqueeze=False: [B, sample_size, ...]

            input_expanded_flattened:
                sample dimension을 batch dimension으로 합친 텐서.
                shape: [B * sample_size, ...]

    예시:
        input.shape = [B, C, H, W], do_unsqueeze=True인 경우

            input_expanded.shape = [B, sample_size, C, H, W]
            input_expanded_flattened.shape = [B * sample_size, C, H, W]

        input.shape = [B, 1, C, H, W], do_unsqueeze=False인 경우

            input_expanded.shape = [B, sample_size, C, H, W]
            input_expanded_flattened.shape = [B * sample_size, C, H, W]
    """
    batch_size = input.size(0)
    if do_unsqueeze:
        sz_from = [-1]*(input.dim()+1)
        sz_from[1] = sample_size
        input_expanded = input.unsqueeze(1).expand(*sz_from).contiguous()

        sz_to = list(input.size())
        sz_to[0] = batch_size*sample_size
    else:
        assert input.size(1) == 1
        sz_from = [-1]*(input.dim())
        sz_from[1] = sample_size
        input_expanded = input.expand(*sz_from).contiguous()

        _sz_to = list(input.size())
        sz_to = _sz_to[0:1]+_sz_to[2:]
        sz_to[0] = batch_size*sample_size
    input_expanded_flattened = input_expanded.view(*sz_to)
    return input_expanded, input_expanded_flattened
