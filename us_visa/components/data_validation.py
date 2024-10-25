import json
import sys

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset 
from pandas import DataFrame

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file
from us_visa.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates if the number of columns in the dataframe matches the schema.
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise USvisaException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validates if all required numerical and categorical columns are present in the dataframe.
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return not (missing_categorical_columns or missing_numerical_columns)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Reads the data from a CSV file.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detects dataset drift between the reference dataframe and the current dataframe using Evidently.
        """
        try:
            # Create a data drift report using Evidently's new API
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=reference_df, current_data=current_df)
            
            # Generate and save the report in JSON format
            report = data_drift_report.json()
            json_report = json.loads(report)

            # Write the report to a YAML file
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)

            # Extract drift-related information from the report
            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process and returns a DataValidationArtifact.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")

            # Load the training and testing datasets
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            # Validate columns for training dataframe
            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_msg += "Columns are missing in training dataframe. "
            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_msg += "Columns are missing in test dataframe. "

            # Check if all required columns are present in training and testing data
            if not self.is_column_exist(df=train_df):
                validation_error_msg += "Required columns are missing in training dataframe. "
            if not self.is_column_exist(df=test_df):
                validation_error_msg += "Required columns are missing in test dataframe. "

            validation_status = len(validation_error_msg) == 0

            # If column validation passed, perform data drift detection
            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info("Drift detected.")
                    validation_error_msg = "Drift detected."
                else:
                    logging.info("No drift detected.")
                    validation_error_msg = "Drift not detected."
            else:
                logging.info(f"Validation errors: {validation_error_msg}")

            # Create a DataValidationArtifact object
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USvisaException(e, sys)
