
from datetime import datetime, timedelta
import pickle
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.experiments.experiment import Experiment
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.inputs import TrainingInput


session = sagemaker.Session()
image_uri = "101778506059.dkr.ecr.ap-southeast-1.amazonaws.com/ss-trainer:latest"

def launch_fold_tuning(
    trial_id,
    seed,
    target_symbols,
    fold_number,
    dataset_start_date,
    period_train_len,
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
        hyperparameters={ # 고정값
            "seed": seed,
            "target-symbols": ",".join(target_symbols),
            "period-start-date": period_start_date,
            "period-train-len": period_train_len,
            "period-val-len": period_val_len,
            "input-length": input_length,
            "prediction-length": prediction_length,
        },
        base_job_name=f"ss-trainer-{trial_id}-{seed}-{fold_number}",
        input_mode="FastFile",
        use_spot_instances=True,
        max_wait=24 * 60 * 60,
    )

    hyperparameter_ranges = {
        "learning-rate": ContinuousParameter(1e-4, 1e-1),
        "dropout": ContinuousParameter(1e-4, 1e-1),
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
        max_parallel_jobs=1,
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

def main():
    train_data_dir = "s3://ss-data-train/dataset/20250522"
    trial_id = datetime.now().strftime("%m%d%H%M%S")
    target_symbols = ["BTCUSDT", "SOLUSDT", "ADAUSDT"]
    dataset_start_date = datetime(2021, 1, 1)
    period_train_len = 90
    period_val_len = 30
    input_length = 240
    prediction_length = 20

    fold_tuners = []
    for seed in range(5):
        for fold_number in range(3):
            tuner = launch_fold_tuning(
                trial_id,
                seed,
                target_symbols,
                fold_number,
                dataset_start_date,
                period_train_len,
                period_val_len,
                input_length,
                prediction_length,
                train_data_dir,
            )
            fold_tuners.append((f["fold"], tuner))

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
    main()