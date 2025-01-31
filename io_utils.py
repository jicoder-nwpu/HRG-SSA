import json
import pickle
import logging
import os


def save_pickle(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(load_path):
    with open(load_path, "rb") as f:
        return pickle.load(f)

def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def load_json(load_path, lower=True):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = f.read()
        if lower:
            obj = obj.lower()
        return json.loads(obj)

def get_or_create_logger(logger_name=None, log_dir=None):
    logger = logging.getLogger(logger_name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.DEBUG)

    stream_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s  [%(levelname)s] %(module)s; %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

