"""Logging utility for NanoAI experiments."""

import csv
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List


class Logger:
    """Logger class for NanoAI experiments."""

    def __init__(self, log_dir: str, log_level: int = logging.INFO):
        """
        Initialize the Logger.

        Args:
            log_dir (str): Directory to store log files.
            log_level (int): Logging level (default: logging.INFO).
        """
        self.log_dir = log_dir
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        os.makedirs(log_dir, exist_ok=True)
        self.event_log_path = os.path.join(log_dir, "event_log.txt")

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.event_log_path), logging.StreamHandler()],
        )

    def log_result(self, result: Dict[str, Any]) -> None:
        """
        Log a result.

        Args:
            result (Dict[str, Any]): The result to log.
        """
        self.results.append(result)
        self._log_event(f"Logged result: {result}", level=logging.INFO)

    def save_results(self, filename: str) -> None:
        """
        Save all logged results to a CSV file.

        Args:
            filename (str): The name of the file to save results to.
        """
        if not self.results:
            self._log_event("No results to save.", level=logging.WARNING)
            return

        try:
            keys = self.results[0].keys()
            with open(filename, "w", newline="") as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.results)
            self._log_event(f"Saved results to {filename}", level=logging.INFO)
        except Exception as e:
            self._log_event(f"Error saving results: {str(e)}", level=logging.ERROR)

    def log_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Log the experiment configuration.

        Args:
            config (Dict[str, Any]): The experiment configuration to log.
        """
        config_file = os.path.join(self.log_dir, "experiment_config.json")
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
            self._log_event(f"Logged experiment config: {config}", level=logging.INFO)
        except Exception as e:
            self._log_event(f"Error logging experiment config: {str(e)}", level=logging.ERROR)

    def _log_event(self, event: str, level: int = logging.INFO) -> None:
        """
        Log an event with the specified logging level.

        Args:
            event (str): The event to log.
            level (int): The logging level (default: logging.INFO).
        """
        logging.log(level, event)

    def close(self) -> None:
        """Close the logger and log the total experiment duration."""
        duration = datetime.now() - self.start_time
        self._log_event(f"Experiment completed. Total duration: {duration}", level=logging.INFO)

    def log_error(self, error_message: str) -> None:
        """
        Log an error message.

        Args:
            error_message (str): The error message to log.
        """
        self._log_event(error_message, level=logging.ERROR)

    def log_warning(self, warning_message: str) -> None:
        """
        Log a warning message.

        Args:
            warning_message (str): The warning message to log.
        """
        self._log_event(warning_message, level=logging.WARNING)

    def log_debug(self, debug_message: str) -> None:
        """
        Log a debug message.

        Args:
            debug_message (str): The debug message to log.
        """
        self._log_event(debug_message, level=logging.DEBUG)

    def log_event(self, event: str) -> None:
        """
        Log an informational event.

        Args:
            event (str): The event to log.
        """
        self._log_event(event, level=logging.INFO)

    def log_metric(self, metric_name: str, value: float, step: int = None) -> None:
        """
        Log a metric value.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            step (int, optional): The step or iteration number.
        """
        message = f"{metric_name}: {value}"
        if step is not None:
            message = f"Step {step}: {message}"
        self._log_event(message, level=logging.INFO)

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            hyperparams (Dict[str, Any]): The hyperparameters to log.
        """
        self._log_event(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}", level=logging.INFO)

    def log_model_summary(self, model: Any) -> None:
        """
        Log a summary of the model architecture.

        Args:
            model (Any): The model to summarize.
        """
        summary = str(model)
        self._log_event(f"Model Summary:\n{summary}", level=logging.INFO)

    def log_dataset_info(
        self, dataset_name: str, train_size: int, val_size: int, test_size: int
    ) -> None:
        """
        Log information about the dataset.

        Args:
            dataset_name (str): The name of the dataset.
            train_size (int): The size of the training set.
            val_size (int): The size of the validation set.
            test_size (int): The size of the test set.
        """
        info = f"Dataset: {dataset_name}\n"
        info += f"Train size: {train_size}\n"
        info += f"Validation size: {val_size}\n"
        info += f"Test size: {test_size}"
        self._log_event(info, level=logging.INFO)

    def log_exception(self, exc_info: tuple) -> None:
        """
        Log an exception with full traceback.

        Args:
            exc_info (tuple): The exception info tuple from sys.exc_info().
        """
        logging.exception("An exception occurred:", exc_info=exc_info)
