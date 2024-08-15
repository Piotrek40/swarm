"""Logging utility for NanoAI experiments."""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from torch import nn


class Logger:
    """Logger class for NanoAI experiments."""

    def __init__(self, log_dir: str, log_level: int = logging.INFO) -> None:
        """
        Initialize the Logger.

        Args:
            log_dir: Directory to store log files.
            log_level: Logging level (default: logging.INFO).
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
            result: The result to log.
        """
        self.results.append(result)
        self._log_event(f"Logged result: {result}", level=logging.INFO)

    def save_results(self, filename: str) -> None:
        """
        Save all logged results to a CSV file.

        Args:
            filename: The name of the file to save results to.
        """
        if not self.results:
            self._log_event("No results to save.", level=logging.WARNING)
            return

        try:
            keys = self.results[0].keys()
            with open(filename, "w", newline="", encoding="utf-8") as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.results)
            self._log_event(f"Saved results to {filename}", level=logging.INFO)
        except IOError as e:
            self._log_event(f"Error saving results: {str(e)}", level=logging.ERROR)

    def log_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Log the experiment configuration.

        Args:
            config: The experiment configuration to log.
        """
        config_file = os.path.join(self.log_dir, "experiment_config.json")
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            self._log_event(f"Logged experiment config: {config}", level=logging.INFO)
        except IOError as e:
            self._log_event(f"Error logging experiment config: {str(e)}", level=logging.ERROR)

    def _log_event(self, event: str, level: int = logging.INFO) -> None:
        """
        Log an event with the specified logging level.

        Args:
            event: The event to log.
            level: The logging level (default: logging.INFO).
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
            error_message: The error message to log.
        """
        self._log_event(error_message, level=logging.ERROR)

    def log_warning(self, warning_message: str) -> None:
        """
        Log a warning message.

        Args:
            warning_message: The warning message to log.
        """
        self._log_event(warning_message, level=logging.WARNING)

    def log_debug(self, debug_message: str) -> None:
        """
        Log a debug message.

        Args:
            debug_message: The debug message to log.
        """
        self._log_event(debug_message, level=logging.DEBUG)

    def log_event(self, event: str) -> None:
        """
        Log an informational event.

        Args:
            event: The event to log.
        """
        self._log_event(event, level=logging.INFO)

    def log_metric(self, metric_name: str, value: float, step: int = None) -> None:
        """
        Log a metric value.

        Args:
            metric_name: The name of the metric.
            value: The value of the metric.
            step: The step or iteration number.
        """
        message = f"{metric_name}: {value}"
        if step is not None:
            message = f"Step {step}: {message}"
        self._log_event(message, level=logging.INFO)

    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            hyperparams: The hyperparameters to log.
        """
        self._log_event(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}", level=logging.INFO)

    def log_model_summary(self, model: nn.Module) -> None:
        """
        Log a summary of the model architecture.

        Args:
            model: The model to summarize.
        """
        summary = str(model)
        self._log_event(f"Model Summary:\n{summary}", level=logging.INFO)

    def log_dataset_info(
        self, dataset_name: str, train_size: int, val_size: int, test_size: int
    ) -> None:
        """
        Log information about the dataset.

        Args:
            dataset_name: The name of the dataset.
            train_size: The size of the training set.
            val_size: The size of the validation set.
            test_size: The size of the test set.
        """
        info = (
            f"Dataset: {dataset_name}\n"
            f"Train size: {train_size}\n"
            f"Validation size: {val_size}\n"
            f"Test size: {test_size}"
        )
        self._log_event(info, level=logging.INFO)

    def log_exception(self, exc_info: Tuple[type, Exception, Any]) -> None:
        """
        Log an exception with full traceback.

        Args:
            exc_info: The exception info tuple from sys.exc_info().
        """
        logging.exception("An exception occurred:", exc_info=exc_info)
