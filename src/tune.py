import os
from typing import List
from config import paths
from logger import get_logger, log_error
from hyperparameter_tuning.tuner import tune_hyperparameters
from data_models.data_validator import validate_multiple_datasets
from schema.data_schema import load_json_data_schema, ForecastingSchema

from preprocessing.preprocess import (
    get_preprocessing_pipelines,
    fit_transform_with_pipeline,
)
from utils import (
    read_tuning_datasets,
    save_json,
    read_json_as_dict,
    set_seeds,
    ResourceTracker,
)

logger = get_logger(task_name="tune")


def read_tuning_schemas(tuning_dir_path: str) -> List[ForecastingSchema]:
    results = []
    datasets = [
        i
        for i in os.listdir(tuning_dir_path)
        if os.path.isdir(os.path.join(tuning_dir_path, i))
    ]

    datasets_paths = [os.path.join(tuning_dir_path, dataset) for dataset in datasets]
    for path in datasets_paths:
        schema = load_json_data_schema(path)
        results.append(schema)
    return results


def run_tuning(
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    preprocessing_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
    hpt_specs_file_path: str = paths.HPT_CONFIG_FILE_PATH,
    hpt_results_dir_path: str = paths.HPT_OUTPUTS_DIR,
    default_tuning_datasets_dir_path: str = paths.DEFAULT_TUNING_DATASETS_DIR,
    user_tuning_datasets_dir_path: str = paths.USER_TUNING_DATASETS_DIR,
    TUNED_HYPERPARAMETERS_FILE_PATH_path: str = paths.TUNED_HYPERPARAMETERS_FILE_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        model_config_file_path (str, optional): The path of the model configuration file.
        preprocessing_config_file_path (str, optional): The path of the preprocessing config file.
        default_hyperparameters_file_path (str, optional): The path of the default hyperparameters file.
        hpt_specs_file_path (str, optional): The path of the configuration file for hyperparameter tuning.
        hpt_results_dir_path (str, optional): Dir path where to save the HPT results.
        default_tuning_datasets_dir_path (str, optional): Dir path where the default tuning datasets are stored.
        user_tuning_datasets_dir_path (str, optional): Dir path where the user tuning datasets are stored.
        TUNED_HYPERPARAMETERS_FILE_PATH_path (str, optional): File path where to save the tuned hyperparameters.
    Returns:
        None
    """

    try:
        with ResourceTracker(logger, monitoring_interval=0.1):
            # load model config
            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)

            tuning_dir_path = (
                default_tuning_datasets_dir_path
                if model_config["tune_on_default_datasets"]
                else user_tuning_datasets_dir_path
            )
            logger.info("Starting training...")
            # load and save schema
            logger.info("Loading and saving schema...")
            data_schemas = read_tuning_schemas(tuning_dir_path)

            # set seeds
            logger.info("Setting seeds...")
            set_seeds(seed_value=model_config["seed_value"])

            # load train data
            logger.info("Loading train data...")
            train_data = read_tuning_datasets(tuning_dir_path=tuning_dir_path)

            # validate the data
            logger.info("Validating train data...")

            validated_data = validate_multiple_datasets(
                data=train_data, data_schema=data_schemas, is_train=True
            )

            logger.info("Loading preprocessing config...")
            preprocessing_config = read_json_as_dict(preprocessing_config_file_path)

            # use default hyperparameters to train model
            logger.info("Loading hyperparameters...")
            logger.info("Tuning hyperparameters on dataset(s)...")

            tuned_hyperparameters = tune_hyperparameters(
                validated_data=validated_data,
                data_schemas=data_schemas,
                preprocessing_config=preprocessing_config,
                model_config=model_config,
                hpt_results_dir_path=hpt_results_dir_path,
                is_minimize=False,  # scoring metric is r-squared - so maximize it.
                default_hyperparameters_file_path=default_hyperparameters_file_path,
                hpt_specs_file_path=hpt_specs_file_path,
            )

            logger.info("Saving tuned hyperparameters")
            save_json(TUNED_HYPERPARAMETERS_FILE_PATH_path, tuned_hyperparameters)

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_tuning()
