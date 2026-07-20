import os
import yaml

def test_config_files_exist():
    base_dir = "config"
    files = ["model-templates.yaml", "hardware-profiles.yaml", "framework-versions.yaml"]
    for f in files:
        path = os.path.join(base_dir, f)
        assert os.path.exists(path), f"{f} does not exist"
        with open(path, "r") as stream:
            data = yaml.safe_load(stream)
            assert data is not None
