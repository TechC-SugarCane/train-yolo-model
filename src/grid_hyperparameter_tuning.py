import sys
import os
from datetime import datetime
from typing import Optional

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics
from ray.tune import TuneConfig, RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import yaml


# 以下のエラーがWindowsで起きるので、それの対策
# UnicodeEncodeError: 'cp932' codec can't encode character '\u274c' in position 3871: illegal multibyte sequence
sys.stdout.reconfigure(encoding='utf-8')



# 参考: https://github.com/ultralytics/ultralytics/blob/675d3705923915cd5f67765f4721895b12a3f0be/ultralytics/cfg/__init__.py#L67
metric = 'metrics/mAP50-95(B)'

# 参考: https://github.com/ultralytics/ultralytics/blob/675d3705923915cd5f67765f4721895b12a3f0be/ultralytics/utils/tuner.py#L7-L157
def run_ray_tune(
    model: YOLO,
    space: dict = None,
    grace_period: int = 10,
    max_samples: int = 10,
    use_wandb: bool = False,
    stop_train_epoch: Optional[int] = None,
    **train_args,
):
    default_space = {
        # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
        "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
        "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
        "box": tune.uniform(0.02, 0.2),  # box loss gain
        "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "bgr": tune.uniform(0.0, 1.0),  # image channel BGR (probability)
        "mosaic": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
    }

    # Put the model in ray store
    model_in_store = ray.put(model)

    def _tune(config):
        """
        Trains the YOLO model with the specified hyperparameters and additional arguments.

        Args:
            config (dict): A dictionary of hyperparameters to use for training.

        Returns:
            None
        """
        model_to_train = ray.get(model_in_store)  # get the model from ray store for tuning
        model_to_train.reset_callbacks()
        config.update(train_args)
        results = model_to_train.train(**config)
        return results.results_dict

    # Get search space
    if not space:
        space = default_space
        print("WARNING ⚠️ search space not provided, using default search space.")

    data = train_args['data']
    space["data"] = data

    # Define the trainable function with allocated resources
    trainable_with_resources = tune.with_resources(_tune, resources={"cpu": os.cpu_count(), "gpu": 0})

    # Define the ASHA scheduler for hyperparameter search
    asha_scheduler = ASHAScheduler(
        time_attr="epoch",
        metric=metric,
        mode="max",
        max_t=max_samples,
        grace_period=grace_period,
        reduction_factor=3,
    )

    tune_config = TuneConfig(
        scheduler=asha_scheduler,
        num_samples=max_samples,
        # Windowsでのエラー回避 (path名260文字制限のため)
        # これしないと、「FileNotFoundError: [WinError 3] 指定されたパスが見つかりません。」 とエラーが出る
        trial_dirname_creator=lambda trial: "trial_" + str(trial.trial_id),
    )

    # Define the callbacks for the hyperparameter search
    tuner_callbacks = [WandbLoggerCallback(project="YOLOv10-tune")] if use_wandb else []

    run_config = RunConfig(
        name="sugarcane",
        stop={"training_iteration": stop_train_epoch} if stop_train_epoch else None,
        callbacks=tuner_callbacks,
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune_config,
        run_config=run_config,
    )

    # Run the hyperparameter search
    tuner.fit()

    # Get the results of the hyperparameter search
    results = tuner.get_results()

    # Shut down Ray to clean up workers
    ray.shutdown()

    return results


def main():
    now = datetime.now()

    model = YOLO("../weights/yolov10/yolov10n.pt", task="detect")

    train_args = {
        # ライブラリ内でdataを参照するROOTが.venv/Lib/site-packages/ultralyticsなので、../../../data/sugarcane.yamlとする必要がある
        "data": "../../../data/sugarcane.yaml",
        # これを設定しなくてもデフォルトで.venv\Lib\site-packages\ultralytics\cfg\default.yaml を参照してくれるが、ここがバージョンアップによって変わったら怖いので、固定している
        "cfg": "../../../cfg/yolov10/sugarcane.yaml",
        "epochs": 300,
        "batch": 8,
        "imgsz": 640,
        "device": 0,
    }

    result_grid = run_ray_tune(
        model=model,
        grace_period=10,
        max_samples=10,
        stop_train_epoch=10,
        **train_args
    )

    print("=" * 80)

    best_result = result_grid.get_best_result(metric=metric, mode='max')

    best_metrics = best_result.metrics
    best_hyperparameters = best_result.config

    print(f"\nBest trial metrics: {best_metrics}")
    print(f"Best hyperparameters: {best_hyperparameters}")

    with open("best_hyperparameters.yaml", "w") as f:
        yaml.dump(best_hyperparameters, f)

    end = datetime.now() - now
    h = end.seconds // 3600
    m = (end.seconds // 60) % 60
    s = end.seconds % 60
    print(f"\nElapsed time: {h} hours, {m} minutes, {s} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
