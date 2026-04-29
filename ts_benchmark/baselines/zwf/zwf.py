from ts_benchmark.baselines.zwf.models.zwf_model import ZWFModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch.nn as nn
from torch import optim

# model hyper params
MODEL_HYPER_PARAMS = {
    # 在 Informer / Transformer 类的时间序列模型 中，decoder 的输入通常是：label_len (历史真实值) + pred_len (预测步长)
    "freq": "h",
    "patience": 10,
}

class ZWF(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(ZWF, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        # self.config = **kwargs
        self.config_ = kwargs


    @property
    def model_name(self):
        return "ZWF"
    
    def _init_model(self):
        # print("初始化 ZWF model with config:", self.config)
        # for key, value in vars(self.config).items():
        #     print(f"{key}: {value}")
        
        return ZWFModel(self.config_)
    
    def _process(self, input, target, input_mark, target_mark):

        """
        对ETTh1_missing_4，input内容为
        'HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT',                                           [0-6]
        'HUFL_miss', 'HULL_miss', 'LUFL_miss', 'LULL_miss', 'MUFL_miss', 'MULL_miss', 'OT_miss',        [7-13]
        'mask_HUFL', 'mask_HULL', 'mask_LUFL', 'mask_LULL', 'mask_MUFL', 'mask_MULL', 'mask_OT',        [14-20]
        't_dif_HUFL', 't_dif_HULL', 't_dif_LUFL', 't_dif_LULL', 't_dif_MUFL', 't_dif_MULL', 't_dif_OT'  [21-27]
        't_idx',                                                                                        [28]
        """
        """
        本方法作为模板方法，定义了数据处理和建模的标准流程，以及计算附加损失的规范。应根据自身需求实现具体的处理和计算逻辑。

        参数:
        - input: 输入数据，具体形式和含义取决于子类实现
        - target: 目标数据，与输入数据配合用于处理和损失计算
        - input_mark: 输入数据的标记/元数据，辅助数据处理或模型训练
        - target_mark: 目标数据的标记/元数据，同样辅助数据处理或模型训练

        返回:
        - dict: 包含以下至少一个键的字典:
            - 'output' (必需): 模型输出张量
            - 'additional_loss' (可选): 存在的附加损失值

        异常:
        - NotImplementedError: 如果子类未实现本方法，调用时将抛出该异常

        注意DeepForecastingModelBase类中
        multi_forecasting_hyper_param_tune指定了label_len长度为seq_len(self.model_name == "MICN"时)或者seq_len//2
        single_forecasting_hyper_param_tune则是指定成self.config.horizon
        并且无法通过参数设置来修改label_len的值
        导致获取的target跟target_mark的长度是 label_len + horizon

        训练时target的前label_len步是历史真实值，后horizon步是未来预测目标值，也就是跟后horizon步的真实值进行对比来计算评估指标，因此需要提供未来horizon步的真实值来计算评估指标
        训练时有 teacher forcing，所以模型每次都用真实历史值，不滚动

        评估时为滚动预测
        arget[:, :label_len, :] → 上一轮模型输出（不是直接用真实值）
        target[:, label_len:, :] → 未来horizon步的真实值（评估指标的计算需要）

        故在模型初始化中，使用全局变量进行覆盖，保证label_len是所需的
        """

        # print("ZWF: processing===============================================================================")
        # print(f"Input shape: {input.shape}")
        # print(f"Target shape: {target.shape}")
        # print(f"Input mark shape: {input_mark.shape}")
        # print(f"Target mark shape: {target_mark.shape}")

        # 将input按列切分成四部分：原始值、mask、时间差分特征，以及时间索引
        input_values = input[:, :, :7:14]  # miss部分（HUFL 到 OT）
        input_masks = input[:, :, 14:21]  # mask部分
        input_time_difs = input[:, :, 21:28]  # 时间差分特征部分
        input_time_idx = input[:, :, 28:]  # 时间索引部分

        # print(f"input_values shape: {input_values.shape}")
        # print(f"input_masks shape: {input_masks.shape}")
        # print(f"input_time_difs shape: {input_time_difs.shape}")
        # print(f"input_time_idx shape: {input_time_idx.shape}")
        
        output, loss_importance = self.model(input_values, input_masks, input_time_difs, input_time_idx)
        # print(f"Output shape before padding: {output.shape}")
        batch_size,horizon,C = output.shape
        if output.shape[2] < input.shape[2]:  # self.config.data_dim
            data_dim = input.shape[2]
            # 补齐到 data_dim
            import torch
            padding = torch.zeros(batch_size, horizon, data_dim - output.shape[2], device=output.device)
            output = torch.cat([output, padding], dim=2)
        # print(f"Output shape after padding: {output.shape}")
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss
    

    def _init_criterion_and_optimizer(self):

        if self.config.loss == "MSE":
            criterion = nn.MSELoss()
        elif self.config.loss == "MAE":
            criterion = nn.L1Loss()
        else:
            criterion = nn.HuberLoss(delta=0.5)

        # optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        optimizer = optim.Adam([
            {'params': [self.model.residual_decomposition.raw_lengths], 'lr': self.config.lr * 100},  # raw_lengths lr 提高
            {'params': [p for n, p in self.model.named_parameters() 
                if n != 'residual_decomposition.raw_lengths'], 'lr': self.config.lr}  # 其他参数用默认 lr
            ])
        return criterion, optimizer

"""
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "ETTh1_missing_4_linear_filled" --strategy-args '{"horizon": 36, "target_channel": [[0, 7]]}' --model-name "zwf.ZWF" --model-hyper-params '{"batch_size": 128, "loss": "MAE", "lr": 0.001, "lradj": "type1", "norm": true, "num_epochs": 100, "patience": 5, "seq_len": 12, "horizon": 24}' --deterministic "full" --gpus 0 --num-workers 1 --timeout 60000 --save-path "ETTh1_missing_4_linear_filled\ZWF" 

"""