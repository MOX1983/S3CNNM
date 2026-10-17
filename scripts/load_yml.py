from dataclasses import dataclass
from pathlib import Path, WindowsPath

from yaml import safe_load

global_path = Path(__file__).parent.parent


@dataclass
class PathConfig:
    to_save_full: WindowsPath
    to_save_mask: WindowsPath
    path_minio_load: WindowsPath
    bucket_name_raw: WindowsPath
    bucket_name_good: WindowsPath


def load_yml(path: str = "config.yml"):
    with open(path, "r") as f:
        data = safe_load(f)

        return PathConfig(
            global_path / data["path"]["to_save_full"],
            global_path / data["path"]["to_save_mask"],
            global_path / data["path"]["minio_load"],
            global_path / data["path"]["bucket_name_raw"],
            global_path / data["path"]["bucket_name_good"],
        )


config = load_yml()
