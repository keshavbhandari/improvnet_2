"""Includes functionality for loading config files."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from importlib import resources
from typing import Any, cast


def load_config(load_path: Path | str | None = None) -> dict[str, Any]:
    """Returns a dictionary loaded from the config.json file."""
    if load_path is not None:
        with open(load_path, "r") as f:
            return cast(dict[str, Any], json.load(f))
    else:
        with (
            resources.files("improvnet.tokenizer")
            .joinpath("config.json")
            .open("r") as f
        ):
            return cast(dict[str, Any], json.load(f))

def get_logger(name: str | None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        if name is not None:
            formatter = logging.Formatter(
                "[%(asctime)s]: [%(levelname)s] [%(name)s] %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "[%(asctime)s]: [%(levelname)s] %(message)s"
            )

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


@lru_cache(maxsize=1)
def warn_once(logger_name: str, message: str):
    logger = logging.getLogger(logger_name)
    logger.warning(message)


@lru_cache(maxsize=1)
def load_maestro_metadata_json() -> dict[str, Any]:
    """Loads MAESTRO metadata json ."""
    with (
        resources.files("improvnet.tokenizer")
        .joinpath("maestro_metadata.json")
        .open("r") as f
    ):
        return cast(dict[str, Any], json.load(f))


@lru_cache(maxsize=1)
def load_aria_midi_metadata_json(
    metadata_load_path: Path | str | None = None,
) -> dict[int, dict[str, Any]]:
    """Loads MAESTRO metadata json."""
    if metadata_load_path is None:
        metadata_load_path = Path(
            str(
                resources.files("improvnet.tokenizer").joinpath(
                    "aria_midi_metadata.json"
                )
            )
        )
    with open(str(metadata_load_path), "r") as f:
        return {
            int(k): v
            for k, v in cast(dict[int, dict[str, Any]], json.load(f)).items()
        }