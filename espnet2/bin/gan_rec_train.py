#!/usr/bin/env python3
from espnet2.tasks.gan_rec import GANRECTask


def get_parser():
    parser = GANRECTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based REC training

    Example:

        % python gan_tts_train.py --print_config --optim1 adadelta
        % python gan_tts_train.py --config conf/train.yaml
    """
    GANRECTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
