import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

def train_step(model, data, target, optimizer, device):
    optimizer.zero_grad()
    output = model(data)
    loss = torch.nn.functional.nll_loss(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)
    return loss
