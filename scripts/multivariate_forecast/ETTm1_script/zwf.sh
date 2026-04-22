HYPER='{"CI": 1, "batch_size": 8, "d_ff": 512, "d_model": 256, "dropout": 0.1}'

python ./scripts/run_benchmark.py \
--config-path "rolling_forecast_config.json" \
--data-name-list "ILI.csv" \
--strategy-args '{"horizon": 24}' \
--model-name "your_model.DUET" \
--model-hyper-params "$HYPER" \
--gpus 0