"""Train or resume the style encoder using its standalone configuration."""
from improvnet.model.encoders.train import main as shared_main, train as shared_train
from improvnet.model.encoders.style.model import StyleEncoder


def train(config):
    """Train StyleEncoder with the shared optimizer and checkpoint engine."""
    return shared_train(config, model_class=StyleEncoder)


def main():
    shared_main(default_config='improvnet.model.encoders.style.config', model_class=StyleEncoder)


if __name__ == '__main__':
    main()
