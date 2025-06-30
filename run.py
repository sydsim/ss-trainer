
from datetime import datetime, timedelta
import argparse
import pickle
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter, CategoricalParameter


session = sagemaker.Session()
image_uri = "101778506059.dkr.ecr.ap-southeast-1.amazonaws.com/ss-trainer:latest"


def launch_fold_tuning(
    trial_id,
    target_symbols,
    seed,
    fold_number,
    dataset_start_date,
    period_val_len,
    input_length,
    prediction_length,
    train_data_dir,
):
    period_start_date = dataset_start_date + timedelta(days=period_val_len * fold_number)
    period_start_date = period_start_date.date().isoformat()

    estimator = Estimator(
        image_uri=image_uri,
        entry_point="train.py",
        source_dir="src/",
        role="arn:aws:iam::101778506059:role/SageMakerExecutionRole",
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        output_path=f"s3://ss-data-train/model/{trial_id}/{seed}/{fold_number}/",
        hyperparameters={  # 고정값
            "seed": seed,
            "target-symbols": ",".join(target_symbols),
            "period-start-date": period_start_date,
            "period-val-len": period_val_len,
            "input-length": input_length,
            "prediction-length": prediction_length,
            "learning-rate": 1e-4,
            "dropout": 0.2,
        },
        base_job_name=f"ss-trainer-{trial_id}-{seed}-{fold_number}",
        input_mode="FastFile",
        use_spot_instances=True,
        max_wait=24 * 60 * 60,
    )

    hyperparameter_ranges = {
        "dropout": ContinuousParameter(1e-4, 1e-1),
        "period-train-len": CategoricalParameter([90, 365, 365 * 3]),
        "hidden-dims": IntegerParameter(16, 128),
        "num-layer": IntegerParameter(1, 4),
    }

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:loss",
        metric_definitions=[
            {
                "Name":  "validation:loss",
                "Regex": "validation:loss=([0-9\\.]+)",
            },
        ],
        objective_type="Minimize",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=100,
        max_parallel_jobs=8,
        base_tuning_job_name=f"ss-trainer-{trial_id}-{seed}-{fold_number}-tuner",
    )

    tuner.fit(
        inputs={
            "train": train_data_dir,
        },
        job_name=f"ss-trainer-{trial_id}-{seed}-{fold_number}",
        experiment_config={
            "ExperimentName": "ss-trainer",
            "TrialName": f"{trial_id}-{seed}-{fold_number}",
            "RunName": f"run-{trial_id}-{seed}-{fold_number}",
        },
    )
    return tuner


def main(
    train_data_dir,
    dataset_start_date,
    target_symbols,
    period_val_len,
    input_length,
    prediction_length,
    num_seed,
    num_fold,
):
    dataset_start_date = datetime.fromisoformat(dataset_start_date)
    target_symbols = target_symbols.strip().split(",")

    trial_id = datetime.now().strftime("%m%d%H%M%S")

    fold_tuners = []
    for seed in range(num_seed):
        for fold_number in range(num_fold):
            tuner = launch_fold_tuning(
                trial_id,
                target_symbols,
                seed,
                fold_number,
                dataset_start_date,
                period_val_len,
                input_length,
                prediction_length,
                train_data_dir,
            )
            fold_tuners.append((fold_number, tuner))

    # 모든 튜닝 잡이 끝난 뒤
    final_params = {}
    for fold_number, tuner in fold_tuners:
        best_hp = tuner.best_estimator().hyperparameters()
        best_metrics = tuner.best_training_job()  # job 명칭 얻어와서 메트릭 로딩 가능
        final_params[fold_number] = {
            "learning_rate": float(best_hp["learning_rate"]),
            "dropout": int(best_hp["dropout"]),
            "best_metrics": best_metrics,
        }
    print(final_params)
    with open("result.pkl", "wb") as f:
        pickle.dump(final_params, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    train_data_dir = "s3://ss-data-train/dataset/20250522"
    dataset_start_date = datetime(2023, 1, 1)
    target_symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
    period_val_len = 30
    input_length = 240
    prediction_length = 20

    parser.add_argument('--train-data-dir', type=str, default="s3://ss-data-train/dataset/20250522")
    parser.add_argument('--train-start-date', type=str, default="2024-01-01")

    parser.add_argument("--target-symbols", type=str, default="BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT,ADAUSDT")
    parser.add_argument("--period-val-len", type=int, default=30)
    parser.add_argument("--input-length", type=int, default=240)
    parser.add_argument("--prediction-length", type=int, default=20)

    parser.add_argument("--num-seed", type=int, default=10)
    parser.add_argument("--num-fold", type=int, default=24)

    args = parser.parse_args()

    main(
        args.train_data_dir,
        args.train_start_date,

        args.target_symbols,
        args.period_val_len,
        args.input_length,
        args.prediction_length,

        args.num_seed,
        args.num_fold,
    )
