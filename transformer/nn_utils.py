from collections.abc import Iterable
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def softmax(
        in_features: Float[Tensor, " ..."], dim: int = -1
) -> Float[Tensor, " ..."]:
    
    x = in_features
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x-x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def CrossEntropy(
        inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
)-> Float[Tensor, ""]:
    
    x = inputs
    x_target = x[range(x.size(0)), targets].unsqueeze(1)
    x_max, _ = torch.max(x, dim=-1,keepdim=True)
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim = -1, keepdim=True)
    result = x_target - x_max - torch.log(x_sum)
    return  - torch.mean(result)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # 1. 过滤出有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if len(params_with_grad) == 0:
        return
    
    # 2. 计算所有参数梯度的总L2范数
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 
        2
    )
    
    # 3. 如果总范数超过阈值，进行缩放
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)  # 关键修正：使用 .data 而不是 .detach()
    
     