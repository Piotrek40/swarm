import csv
import os
import json
import logging
from datetime import datetime


class Logger:
    def __init__(self, log_dir, log_level=logging.INFO):
        self.log_dir = log_dir
        self.results = []
        self.start_time = datetime.now()
        os.makedirs(log_dir, exist_ok=True)
        self.event_log_path = os.path.join(log_dir, "event_log.txt")

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(self.event_log_path), logging.StreamHandler()],
        )

    def log_result(self, result):
        self.results.append(result)
        self._log_event(f"Logged result: {result}", level=logging.INFO)

    def save_results(self, filename):
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

    def log_experiment_config(self, config):
        config_file = os.path.join(self.log_dir, "experiment_config.json")
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
            self._log_event(f"Logged experiment config: {config}", level=logging.INFO)
        except Exception as e:
            self._log_event(f"Error logging experiment config: {str(e)}", level=logging.ERROR)

    def _log_event(self, event, level=logging.INFO):
        logging.log(level, event)

    def close(self):
        duration = datetime.now() - self.start_time
        self._log_event(f"Experiment completed. Total duration: {duration}", level=logging.INFO)

    def log_error(self, error_message):
        self._log_event(error_message, level=logging.ERROR)

    def log_warning(self, warning_message):
        self._log_event(warning_message, level=logging.WARNING)

    def log_debug(self, debug_message):
        self._log_event(debug_message, level=logging.DEBUG)

    def log_event(self, event):
        self._log_event(event, level=logging.INFO)

    def log_metric(self, metric_name, value, step=None):
        message = f"{metric_name}: {value}"
        if step is not None:
            message = f"Step {step}: {message}"
        self._log_event(message, level=logging.INFO)

    def log_hyperparameters(self, hyperparams):
        self._log_event(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}", level=logging.INFO)

    def log_model_summary(self, model):
        summary = str(model)
        self._log_event(f"Model Summary:\n{summary}", level=logging.INFO)

    def log_dataset_info(self, dataset_name, train_size, val_size, test_size):
        info = f"Dataset: {dataset_name}\n"
        info += f"Train size: {train_size}\n"
        info += f"Validation size: {val_size}\n"
        info += f"Test size: {test_size}"
        self._log_event(info, level=logging.INFO)

    def log_exception(self, exc_info):
        logging.exception("An exception occurred:", exc_info=exc_info)
