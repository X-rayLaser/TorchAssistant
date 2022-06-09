import argparse
import json
from scaffolding.training import train
from scaffolding.session import TrainingSession


def load_config(path):
    with open(path) as f:
        s = f.read()

    return json.loads(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ML pipeline according to a specified configuration file'
    )
    parser.add_argument('session_path', type=str, help='Path to the session file')

    cmd_args = parser.parse_args()
    path = cmd_args.session_path

    session = TrainingSession(path)

    #store_path = os.path.join(checkpoints_dir, 'store.json')

    train(session)


# todo: refactor code more (achieve better cohesion, loose coupling)
# todo: 2 separate passes: 1) create the session and save it to some file; 2) load the session (without using config)
# todo: session should contain all information including current epoch, checkpoints with losses and metrics etc.
# todo: session can export best checkpoint according to a criterion
# todo: separate processing of inputs and targets
# todo: tests using GPU device
# todo: allow to change inference device
# todo; support training GANs (e.g. DCGAN)
# todo: support training GANs with growing neural nets
# todo: consider making a batch adapter a part of prediction pipeline (rather than data pipeline)
# todo: support batch sizes > 1 (this will involve some extra transformations like padding, etc.)
# todo: ensure other examples work
# todo: greedy search vs beam search decoding for seq2seq inference
# todo: extra scripts to fine tune and export
# todo: debug tool (show predictions for input as well as all inputs, outputs and transformations)
# todo: think of a better way to implement dynamic arguments and the way that one component (data pipeline) may
# affect other components (models)
# todo: test suite
