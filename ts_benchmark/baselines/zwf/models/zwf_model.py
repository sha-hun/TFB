import torch.nn as nn
import torch

class ZWFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.enc_in, config.pred_len)
    
    def forward(self, x_enc):
        # x_enc: [batch_size, seq_len, n_vars]
        output = self.linear(x_enc)
        loss_importance = torch.tensor(0.0, device=x_enc.device)
        return output, loss_importance
