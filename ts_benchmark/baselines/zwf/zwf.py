from ts_benchmark.baselines.zwf.models.zwf_model import ZWFModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "patience": 10
}

class ZWF(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        super(ZWF, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "ZWF"
    
    def _init_model(self):
        return ZWFModel(self.config)
    
    def _process(self, input, target, input_mark, target_mark):

        """
        对ETTh1_missing_4，input内容为
        'HUFL', 'HULL', 'LUFL', 'LULL', 'MUFL', 'MULL', 'OT',                                           [0-6]
        'mask_HUFL', 'mask_HULL', 'mask_LUFL', 'mask_LULL', 'mask_MUFL', 'mask_MULL', 'mask_OT',        [7-13]
        't_dif_HUFL', 't_dif_HULL', 't_dif_LUFL', 't_dif_LULL', 't_dif_MUFL', 't_dif_MULL', 't_dif_OT'  [14-20]
        't_idx',                                                                                        [21]
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
        """

        print("ZWF: processing===============================================================================")
        print(f"Input shape: {input.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Input mark shape: {input_mark.shape}")
        print(f"Target mark shape: {target_mark.shape}")

        # 输出前几个输入数据的标记信息（如果有的话）    
        print("Input mark sample:", input_mark[:5].cpu().numpy() if input_mark is not None else "None")


        output, loss_importance = self.model(input)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss

