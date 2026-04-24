import sys
import json
from pathlib import Path
import runpy

def run():

    # ===== 1️⃣ 路径 =====
    ROOT = Path(__file__).resolve().parents[3]
    run_file = ROOT / "scripts" / "run_benchmark.py"

    # ===== 2️⃣ 参数 =====
    # 影响实验流程的参数设置，例如Evaluation Config等
    strategy_args = {"horizon": 12 , "target_channel": [[0,6],10], "tv_ratio": 0.75, "train_ratio_in_tv": 0.8}  # 预测未来36步，预测目标是第0到第6列（HUFL 到 OT）
    
    model_hyper_params = {  # 影响模型训练过程的超参数设置,会传入模型中，以config的形式提供给模型使用，例如enc_in、label_len等参数，模型可以根据这些参数进行相应的处理和计算
        "batch_size": 128,
        "c_out": 7,
        "loss": "MAE",
        "lr": 0.001,
        "lradj": "type1",
        "norm": True,
        "num_epochs": 100,
        "patience": 5,
        "period": [48, 90, 102, 360, 720], 
        "seq_len": 12,
        "horizon": 12,
        "enc_in":7,
        "data_dim": 22,  # 输入数据的维度，包含原始值、mask、时间差分特征和时间索引等信息
        "label_len": 0,# 在 Informer / Transformer 类的时间序列模型 中，decoder 的输入通常是：label_len (历史真实值) + pred_len (预测步长)
        "num_seasonal_components" : 4,
        "max_season_length" : 168, # 根据数据的周期性特征设置季节性组件的长度，例如168代表一周的小时级数据
        "trend_length" : 720,  # 趋势组件的长度，可以根据数据的长期趋势特征设置，例如720代表一个月的小时级数据
    }

    strategy_json = json.dumps(strategy_args)
    model_json = json.dumps(model_hyper_params)

    config_path = "rolling_forecast_config.json"
    data_name = "ETTh1_missing_4_linear_filled"
    # data_name = "ETTh1_10test_linear_filled"  # 用于测试的10条数据版本
    model_name = "ZWF"
    save_path = f"{data_name}\\{model_name}"
    deterministic = 'full'
    gpus = "0"
    num_workers = "1"
    timeout = "60000"


    # ===== 3️⃣ ① 输出命令（你要的 print）=====
    cmd = (
        f'python ./scripts/run_benchmark.py '
        f'--config-path "{config_path}" '
        f'--data-name-list "{data_name}" '
        f'--strategy-args \'{strategy_json}\' '
        f'--model-name "{model_name.lower()}.{model_name}" '
        f'--model-hyper-params \'{model_json}\' '
        f'--deterministic "{deterministic}" '
        f'--gpus {gpus} '
        f'--num-workers {num_workers} '
        f'--timeout {timeout} '
        f'--save-path "{save_path}" '
    )

    # print("\n================ COPY COMMAND ================\n")
    # print(cmd)
    # print("\n==============================================\n")

    # ===== 4️⃣ ② 直接运行 run_benchmark =====
    sys.argv = [
        str(run_file),
        "--config-path", config_path,
        "--data-name-list", f"{data_name}.csv",
        "--strategy-args", json.dumps(strategy_args),
        "--model-name", f"{model_name.lower()}.{model_name}",
        "--model-hyper-params", json.dumps(model_hyper_params),
        "--deterministic", deterministic,
        "--gpus", gpus,
        "--num-workers", num_workers,
        "--timeout", timeout,
        "--save-path", f"{data_name}\\{model_name}"
    ]

    # 加项目路径（避免 import 报错）
    sys.path.insert(0, str(ROOT))

    # ===== 5️⃣ 执行 benchmark =====
    runpy.run_path(str(run_file), run_name="__main__")


if __name__ == "__main__":
    run()