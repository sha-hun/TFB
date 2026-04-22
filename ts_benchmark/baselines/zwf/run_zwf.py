import sys
import json
from pathlib import Path
import runpy

def run():

    # ===== 1️⃣ 路径 =====
    ROOT = Path(__file__).resolve().parents[3]
    run_file = ROOT / "scripts" / "run_benchmark.py"

    # ===== 2️⃣ 参数 =====
    strategy_args = {"horizon": 96}

    model_hyper_params = {
        "batch_size": 128,
        "loss": "MAE",
        "lr": 0.001,
        "lradj": "type1",
        "norm": True,
        "num_epochs": 100,
        "patience": 5,
        "period": [48, 90, 102, 360, 720], 
        "seq_len": 96,
        "horizon": 96,
    }
    strategy_json = json.dumps(strategy_args)
    model_json = json.dumps(model_hyper_params)

    config_path = "rolling_forecast_config.json"
    data_name = "ETTm1"
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
        f'--gpus {gpus}'
        f'--num-workers {num_workers}'
        f'--timeout {timeout} '
        f'--save-path "{save_path}"'
    )

    print("\n================ COPY COMMAND ================\n")
    print(cmd)
    print("\n==============================================\n")

    # ===== 4️⃣ ② 直接运行 run_benchmark =====
    data_set = "ETTh1"
    model_name = "ZWF"
    sys.argv = [
        str(run_file),
        "--config-path", config_path,
        "--data-name-list", f"{data_set}.csv",
        "--strategy-args", json.dumps(strategy_args),
        "--model-name", f"{model_name.lower()}.{model_name}",
        "--model-hyper-params", json.dumps(model_hyper_params),
        "--deterministic", deterministic,
        "--gpus", gpus,
        "--num-workers", num_workers,
        "--timeout", timeout,
        "--save-path", f"{data_set}\\{model_name}"
    ]

    # 加项目路径（避免 import 报错）
    sys.path.insert(0, str(ROOT))

    # ===== 5️⃣ 执行 benchmark =====
    runpy.run_path(str(run_file), run_name="__main__")


if __name__ == "__main__":
    run()