import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from yaml import safe_load

global_path = str(Path(__file__).parent.parent) + "\\"


PATH_TO_ENV = Path(__file__).parent.parent / ".env"
load_dotenv(PATH_TO_ENV)


@dataclass
class PathConfig:
    to_save_full: str
    to_save_mask: str
    path_minio_load: str
    bucket_name_raw: str
    bucket_name_good: str
    prefix: str


@dataclass
class ClientS3:
    ACCESS_KEY: str
    SECRET_ACCESS_KEY: str
    ENDPOINT_URL: str


def load_yml(path: str = "config.yml"):
    with open(path, "r") as f:
        data = safe_load(f)

        return PathConfig(
            global_path + data["path"]["to_save_full"] + "\\",
            global_path + data["path"]["to_save_mask"] + "\\",
            global_path + data["path"]["minio_load"] + "\\",
            data["path"]["bucket_name_raw"],
            data["path"]["bucket_name_good"],
            data["path"]["prefix"],
        )


config = load_yml()

configClient = ClientS3(
    os.getenv("ACCESS_KEY"), os.getenv("SECRET_ACCESS_KEY"), os.getenv("ENDPOINT_URL")
)
