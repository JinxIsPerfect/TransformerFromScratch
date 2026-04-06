import torch


def save_checkpoint(model:torch.nn.Module, optimizer: torch.optim.Optimizer, iteration:int, out:str):
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    """
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                },out)


def load_checkpoint(src, model:torch.nn.Module, optimizer:torch.optim.Optimizer):
    
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]

    return iteration