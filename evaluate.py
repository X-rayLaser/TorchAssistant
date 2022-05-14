import argparse
import json

from scaffolding.training import train
from scaffolding import parse


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a model using a specified set of metrics'
    )
    parser.add_argument('config', type=str, help='Path to the configuration file')
