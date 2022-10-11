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
    parser.add_argument('--ivl', default=100, type=int,
                        help='# of most recent iterations used to compute metrics')

    cmd_args = parser.parse_args()
    path = cmd_args.session_path

    saver = SessionSaver(path)
    session = saver.load_from_latest_checkpoint()

    def save_checkpoint(epoch):
        saver.save_checkpoint(session)

    train(session, log_metrics=saver.log_metrics, save_checkpoint=save_checkpoint,
          stat_ivl=cmd_args.ivl)


# todo: support registration of forward hooks (specify model, layer name and hook function)
# todo: training loops with unequal number of iterations
# todo: session can export best checkpoint according to a criterion
# todo: separate processing of inputs and targets
# todo: allow to change inference device
# todo; support training GANs (e.g. DCGAN)
# todo: support training GANs with growing neural nets
# todo: greedy search vs beam search decoding for seq2seq inference
# todo: extra scripts to fine tune and export
# todo: debug tool (show predictions for input as well as all inputs, outputs and transformations)
# todo: test suite
