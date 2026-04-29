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
    strategy_args = {"horizon": 96 ,"seed":2026, "target_channel": [[0,7]], "tv_ratio": 0.75, "train_ratio_in_tv": 0.8}  # 预测未来36步，预测目标是第0到第6列（HUFL 到 OT）
    
    model_hyper_params = {"CI": 1, "batch_size": 32, "d_ff": 512, "d_model": 512, "dropout": 0.5, "e_layers": 1, 
                          "factor": 3, "fc_dropout": 0.1, "horizon": 96, "k": 1, "loss": "MAE", "lr": 0.0005, "lradj": "type1",
                            "n_heads": 1, "norm": True, "num_epochs": 100, "num_experts": 2, ""
                            "patch_len": 48, "patience": 5, "seq_len": 512, 'real_data_dim' : 7,
                            }

    strategy_json = json.dumps(strategy_args)
    model_json = json.dumps(model_hyper_params)

    config_path = "rolling_forecast_config.json"
    data_name = "ETTh1_missing_4_linear_filled"
    # data_name = "ETTh1_10test_linear_filled"  # 用于测试的10条数据版本
    model_name = "DUET"
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
