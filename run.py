"""
MLOps Mini Pipeline
Computes rolling mean signals on OHLCV cryptocurrency data.
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps Mini Pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to config YAML file")
    parser.add_argument("--output",   required=True, help="Path to output metrics JSON file")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and validate the YAML config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config file: {e}")

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping.")

    required_keys = {"seed", "window", "version"}
    missing = required_keys - config.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config 'seed' must be an integer.")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config 'window' must be a positive integer.")
    if not isinstance(config["version"], str):
        raise ValueError("Config 'version' must be a string.")

    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    """Load and validate the input CSV file.
    Handles both standard CSV and whole-row-quoted CSV formats.
    """
    import csv as csv_module

    try:
        # First attempt: standard read
        df = pd.read_csv(input_path)

        # If only 1 column, the file wraps each row in quotes — re-read correctly
        if len(df.columns) == 1:
            df = pd.read_csv(
                input_path,
                quoting=csv_module.QUOTE_NONE,
                escapechar="\\"
            )

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("Input CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")

    # Normalize column names: strip quotes/spaces, lowercase
    # Handles "close", "Close", " close ", "\"close\"" etc.
    df.columns = df.columns.str.replace('"', '').str.strip().str.lower()

    if df.empty:
        raise ValueError("Input CSV file contains no data rows.")

    required_columns = {"close"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    if df["close"].isnull().all():
        raise ValueError("The 'close' column contains only null values.")

    return df


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int) -> pd.Series:
    """Compute rolling mean on the close column."""
    return df["close"].rolling(window=window, min_periods=window).mean()


def generate_signals(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    Generate binary signals:
      1 if close > rolling_mean
      0 if close <= rolling_mean
    Rows where rolling_mean is NaN (insufficient window) are assigned 0.
    """
    signals = (close > rolling_mean).astype(int)
    signals[rolling_mean.isna()] = 0
    return signals


def compute_metrics(signals: pd.Series, rows: int, latency_ms: int,
                    version: str, seed: int) -> dict:
    """Assemble the final metrics dictionary."""
    signal_rate = round(float(signals.mean()), 4)
    return {
        "version":        version,
        "rows_processed": rows,
        "metric":         "signal_rate",
        "value":          signal_rate,
        "latency_ms":     latency_ms,
        "seed":           seed,
        "status":         "success"
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_error_json(path: str, version: str, message: str) -> None:
    payload = {
        "version":       version,
        "status":        "error",
        "error_message": message
    }
    try:
        write_json(path, payload)
    except Exception:
        pass  # best-effort; don't mask original error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    args = parse_args()

    logger = setup_logging(args.log_file)
    logger.info("Job started")

    version = "unknown"  # fallback for error JSON before config loads

    try:
        # --- Config ---
        config = load_config(args.config)
        seed    = config["seed"]
        window  = config["window"]
        version = config["version"]

        np.random.seed(seed)
        logger.info(f"Config loaded: seed={seed}, window={window}, version={version}")

        # --- Data ---
        df = load_data(args.input)
        rows = len(df)
        logger.info(f"Data loaded: {rows} rows")

        # --- Rolling mean ---
        rolling_mean = compute_rolling_mean(df, window)
        logger.info(f"Rolling mean calculated with window={window}")

        # --- Signals ---
        signals = generate_signals(df["close"], rolling_mean)
        logger.info("Signals generated")

        # --- Metrics ---
        latency_ms = int((time.time() - start_time) * 1000)
        metrics = compute_metrics(signals, rows, latency_ms, version, seed)
        logger.info(
            f"Metrics: signal_rate={metrics['value']}, "
            f"rows_processed={metrics['rows_processed']}"
        )

        # --- Write output ---
        write_json(args.output, metrics)
        logger.info(f"Job completed successfully in {latency_ms}ms")

        # Print final metrics to stdout (required by Docker spec)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        error_msg = str(exc)
        logger.error(f"Job failed: {error_msg}")
        write_error_json(args.output, version, error_msg)
        print(json.dumps({"version": version, "status": "error", "error_message": error_msg}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
