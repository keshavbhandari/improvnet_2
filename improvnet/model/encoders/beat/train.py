"""Train or resume the beat encoder using its standalone configuration."""
from improvnet.model.encoders.train import main as shared_main, train as shared_train
from improvnet.model.encoders.beat.model import BeatEncoder


def train(config):
    """Train BeatEncoder with the shared optimizer and checkpoint engine."""
    return shared_train(config, model_class=BeatEncoder)


def main():
    shared_main(default_config='improvnet.model.encoders.beat.config', model_class=BeatEncoder)


if __name__ == '__main__':
    main()
