import argparse
from random import seed
import os
from utils.config_enums import Mode

def parse_args():
    descript = 'Pytorch Implementation'
    parser = argparse.ArgumentParser(description = descript)
    parser.add_argument('--data_path', type = str, default = 'outputs/')
    parser.add_argument('--root_dir', type = str, default = 'outputs/')
    parser.add_argument('--log_path', type = str, default = 'logs/')
    parser.add_argument('--mode', type = Mode, default = Mode.EVAL)
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoints/')
    parser.add_argument('--lr', type = str, default = '[0.0001]*3000', help = 'learning rates for steps(list form)')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--clip_length', type = int, default = 8)
    parser.add_argument('--seed', type = int, default = 2025, help = 'random seed (-1 for no manual seed)')
    parser.add_argument('--model_name', type = str, default = "trans_{}.pkl".format(seed), help = 'the path of pre-trained model file')

    return init_args(parser.parse_args())


def init_args(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.data_path)

    return args
