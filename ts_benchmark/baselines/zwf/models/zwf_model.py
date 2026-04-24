import torch.nn as nn
import torch

class ZWFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # print("初始化 ZWF model with config:", config)
        # for key, value in vars(config).items():
        #     print(f"{key}: {value}")
        T = config['seq_len']
        C = config['enc_in']

        self.T = T
        self.C = C
        self.max_season_length = config['max_season_length']
        self.num_seasonal_components = config['num_seasonal_components']
        self.horizon = config['horizon']

        self.E_trend = nn.Parameter(torch.randn(config['trend_length'], C))
        self.E_seasonal = nn.Parameter(torch.randn(config['num_seasonal_components'], config['max_season_length'], C))
        # 每个周期的可学习长度参数（浮点数）
        self.raw_lengths = nn.Parameter(torch.ones(config['num_seasonal_components']) * (config['max_season_length']))

        self.Linear = nn.Linear( 2 * config['num_seasonal_components'] * C, config['data_dim'])
        
    
    def forward(self, x_enc, mask, time_dif, time_idx):
        # x_enc: [batch_size, seq_len, n_vars]
        # mask: [batch_size, seq_len, n_vars]
        # time_dif: [batch_size, seq_len, n_vars]
        # time_idx: [batch_size, seq_len, 1]

        # print(f"x_enc shape: {x_enc.shape}")
        # print(f"mask shape: {mask.shape}")
        # print(f"time_dif shape: {time_dif.shape}")
        # print(f"time_idx shape: {time_idx.shape}")
        
        batch, T, C = x_enc.shape
        trend_length = self.E_trend.shape[0]

        # 使用高级索引
        time_idx_mod = (time_idx.squeeze(-1) % trend_length).long()  # (batch, seq_len)
        trend_seq = self.E_trend[time_idx_mod]  # (batch, seq_len, C)
        x_detrended = x_enc - trend_seq

        lengths = torch.clamp(self.raw_lengths, 1, self.max_season_length)

        # -----------------------------
        # 多周期残差
        # -----------------------------
        X_Residual = []
        for i in range(self.num_seasonal_components):
            # round + STE 获取整数周期长度 Nsi
            length_float = torch.clamp(self.raw_lengths[i], 1.0, self.max_season_length)
            length_ste = (length_float.round() - length_float).detach() + length_float  # STE
            Nsi = length_ste.long()  # 用于索引
            # 输出Nsi的值
            print(f"Seasonal component {i}: length_float={length_float.item():.2f}, length_ste={length_ste.item():.2f}, Nsi={Nsi.item()}")
            # 循环取模时间对齐（高级索引）
            time_idx_mod = (time_idx.squeeze(-1) % Nsi).long()  # (batch, seq_len)

            # 高级索引，从 E_seasonal[i] 中取对应位置
            seasonal_aligned = self.E_seasonal[i][time_idx_mod]  # (batch, seq_len, C)

            # 残差
            residual_i = x_detrended - seasonal_aligned  # (batch, seq_len, C)

            # 时间维度复制
            residual_i_dup = residual_i.unsqueeze(2).repeat(1, 1, 2, 1)  # (batch, seq_len, 2, C)
            X_Residual.append(residual_i_dup)

        X_Residual = torch.cat(X_Residual, dim=2)  # (batch, seq_len, num_seasonal_components * 2, C)


        X_Residual_flat = X_Residual.reshape(batch, self.horizon, -1)  # (batch, seq_len , 2 * num_seasonal_components * C)



        output = self.Linear(X_Residual_flat)  # [batch_size, horizon, 22]
        loss_importance = torch.tensor(0.0, device=x_enc.device)
        # print(f"输出的形状 : {output.shape}")
        return output, loss_importance
