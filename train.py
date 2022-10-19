import argparse
import json
from torchassistant.training import train
from torchassistant.session import SessionSaver


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('session_path', type=str, help='Path to the session file')

    # maximal number of data points in moving average used to compute metrics
    # todo: make this configurable
    window_width = 100
    cmd_args = parser.parse_args()
    path = cmd_args.session_path

    saver = SessionSaver(path)
    session = saver.load_from_latest_checkpoint()

    def save_checkpoint(epoch):
        saver.save_checkpoint(session)

    train(session, log_metrics=saver.log_metrics, save_checkpoint=save_checkpoint,
          stat_ivl=window_width)
