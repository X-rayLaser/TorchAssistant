import argparse
import json
import os

from scaffolding.session import TrainingSession, ConfigParser
from scaffolding import parse


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')

    cmd_args = parser.parse_args()
    path = cmd_args.config

    config_dict = load_config(path)
    config_dict = config_dict["pipeline"]
    session_config = ConfigParser(config_dict).get_config()

    # todo: take this from session_config
    session_dir = session_config.session_dir

    # todo: consider to move this code inside session object
    if os.path.exists(session_dir):
        print(f"Session already exists under {session_dir}")
    else:
        TrainingSession.create_session(session_config, session_dir)
