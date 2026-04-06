from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-   place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        """
        AdamW 优化器实现
        
        Args:
            params: 需要优化的参数
            lr: 学习率 (default: 1e-3)
            betas: 动量系数 (beta1, beta2) (default: (0.9, 0.999))
            eps: 数值稳定性常数 (default: 1e-8)
            weight_decay: 权重衰减系数 (default: 1e-2)
            amsgrad: 是否使用AMSGrad变体 (default: False)
        """
        assert lr > 0, f"Invalid learning rate: {lr}"
        assert 0.0 <= betas[0] < 1.0, f"Invalid beta parameter at index 0: {betas[0]}"
        assert 0.0 <= betas[1] < 1.0, f"Invalid beta parameter at index 1: {betas[1]}"
        assert eps > 0, f"Invalid epsilon value: {eps}"
        assert weight_decay >= 0, f"Invalid weight_decay value: {weight_decay}"
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad
        }
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单次优化步骤
        
        Args:
            closure: 一个重新计算损失的可调用函数 (default: None)
            
        Returns:
            loss: 如果提供了closure，返回损失值
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    
                    # 初始化状态
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    
                    # 更新步数
                    state['step'] += 1
                    state_steps.append(state['step'])
            
            # 对每个参数执行AdamW更新
            self._adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps']
            )
        
        return loss
    
    def _adamw_step(self, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, 
                   state_steps, amsgrad, beta1, beta2, lr, weight_decay, eps):
        """
        执行实际的AdamW参数更新
        """
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            
            # 更新一阶和二阶动量
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            if amsgrad:
                # 维护二阶动量的最大值
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = max_exp_avg_sqs[i].sqrt().add_(eps)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)
            
            # 偏差校正
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            
            # AdamW的核心：在参数更新时应用权重衰减，而不是在梯度中
            param.mul_(1 - lr * weight_decay)
            
            # Adam更新步骤
            param.addcdiv_(exp_avg, denom, value=-step_size * (math.sqrt(bias_correction2)))


class ConsinSchedule():
    def __init__(self,it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):

        self.it = it
        self.max_lr = max_learning_rate
        self.min_lr = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
    
    def __call__(self):
        # self.it = it
        if self.it < self.warmup_iters:
            return self.max_lr * (self.it / self. warmup_iters)
        elif self.it >= self.warmup_iters and self.it <= self.cosine_cycle_iters:
            v = math.cos((self.it - self.warmup_iters)/(self.cosine_cycle_iters- self.warmup_iters) * math.pi)
            return self.min_lr + 0.5 * (1 + v) * (self.max_lr - self.min_lr)
        else:
            return self.min_lr
    
    def get_lr(self):
        return self()
        


# # 使用示例
# if __name__ == "__main__":
#     # 创建测试模型
#     model = torch.nn.Linear(10, 1)
    
#     # 使用自定义的AdamW优化器
#     optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
#     # 模拟训练步骤
#     for epoch in range(10):
#         # 模拟前向传播和损失计算
#         inputs = torch.randn(32, 10)
#         targets = torch.randn(32, 1)
#         outputs = model(inputs)
#         loss = torch.nn.functional.mse_loss(outputs, targets)
        
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
