import argparse
import os

from torchassistant.session import create_and_save_session
from torchassistant.utils import add_cwd_to_python_path, load_config


def run_init():
    add_cwd_to_python_path()

    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config_dict = load_config(path)

    session_dir = config_dict["session_dir"]

    if os.path.exists(session_dir):
        print(f"Session already exists under {session_dir}")
    else:
        create_and_save_session(config_dict, session_dir)
