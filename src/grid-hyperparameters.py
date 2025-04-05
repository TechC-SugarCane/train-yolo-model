import sys

from ultralytics import YOLO
import yaml

# 以下のエラーがWindowsで起きるので、それの対策
# UnicodeEncodeError: 'cp932' codec can't encode character '\u274c' in position 3871: illegal multibyte sequence
sys.stdout.reconfigure(encoding='utf-8')

model = YOLO("../weights/yolov10/yolov10n.pt")

result_grid = model.tune(
    data="../../../data/sugarcane.yaml",  # ライブラリ内でdataを参照するROOTが.venv/Lib/site-packages/ultralyticsなので、../../../data/sugarcane.yamlとする必要がある
    use_ray=True,
    epochs=10,
    iterations=100,
    batch=8,
    imgsz=640,
    device=0,
    workers=0,  # Windowsではworkers=0にしないとエラーになる
    project="yolo10n_tune",
    name="sugarcane",
)

# Print the results
if result_grid:
    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

    best_result = result_grid.get_best_result(metric='metrics/mAP50(B)', mode='max')

    best_metrics = best_result.metrics
    best_hyperparameters = best_result.config

    print(f"Best trial metrics: {best_metrics}")
    print(f"Best hyperparameters: {best_hyperparameters}")

    with open("best_hyperparameters.yaml", "w") as f:
        yaml.dump(best_hyperparameters, f)

