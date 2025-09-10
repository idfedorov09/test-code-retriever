import gzip
import os
import shutil
from logging.handlers import RotatingFileHandler


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def rotating_handler_factory(
    filename: str,
    maxBytes: int,
    backupCount: int,
    encoding: str = "utf-8",
    delay: bool = True,
    **kwargs,
) -> RotatingFileHandler:
    """
    Фабрика для dictConfig: возвращает RotatingFileHandler с gzip-архивацией при ротации
    """
    _ensure_dir(filename)
    handler = RotatingFileHandler(
        filename=filename,
        maxBytes=int(maxBytes),
        backupCount=int(backupCount),
        encoding=encoding,
        delay=delay,
    )

    handler.namer = lambda default_name: f"{default_name}.gz"

    def _rotator(source: str, dest: str) -> None:
        with open(source, "rb") as sf, gzip.open(dest, "wb") as df:
            shutil.copyfileobj(sf, df)
        os.remove(source)

    handler.rotator = _rotator
    return handler
