import logging.config
import os


def setup_logging():
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    LOG_MAX_NBYTES = int(os.getenv("LOG_MAX_NBYTES", "104857600"))  # 100 mb default
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "json": {
                "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
                "class": "pythonjsonlogger.json.JsonFormatter",
            },
        },
        "filters": {
            "base64_trunc": {
                "()": "utils.logging.filters.Base64TruncatingFilter",
                "max_show": 24,
            }
        },
        "handlers": {
            "console": {
                "level": LOG_LEVEL,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["base64_trunc"],
            },
            "file": {
                "level": LOG_LEVEL,
                "()": "utils.logging.handlers.rotating_handler_factory",
                "filename": "logs/logs.txt",
                "maxBytes": LOG_MAX_NBYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
                "formatter": "standard",
                "filters": ["base64_trunc"],
            },
            "json_file": {
                "level": LOG_LEVEL,
                "()": "utils.logging.handlers.rotating_handler_factory",
                "filename": "logs/logs.json",
                "maxBytes": LOG_MAX_NBYTES,
                "backupCount": LOG_BACKUP_COUNT,
                "encoding": "utf-8",
                "formatter": "json",
                "filters": ["base64_trunc"],
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file", "json_file"],
                "level": LOG_LEVEL,
                "propagate": True,
            },
        },
    }

    logging.config.dictConfig(logging_config)
