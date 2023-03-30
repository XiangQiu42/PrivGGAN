import argparse
from utils.dp_utils import get_noise_mul, get_epsilon
from config import *


# This script is to help define the noise multiper

def arg_parser():
    # init the common args, expect the model specific args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="citeseer", choices=['cora_ml', 'dblp', 'citeseer'])
    parser.add_argument('--target_eps', type=float, default=0.2)
    _args = parser.parse_args()
    return _args


def main(args, m_args):
    delta = 1e-5
    batch_size = m_args.batch_size
    n_steps = m_args.max_iterations  # training iterations
    N = m_args.node_num
    eps = args.target_eps

    prob = batch_size / N  # The sampling rate

    sigma2 = get_noise_mul(N, batch_size, args.target_eps, 8000, target_delta=delta)
    print(f'eps,delta,gamma = ({eps},{delta},{prob}) ==> Noise level sigma (RDP)=', sigma2)

    epsilon = get_epsilon(prob, mul=sigma2, num_steps=n_steps)
    print("Privacy cost is: epsilon={}, delta={} (RDP)".format(epsilon, delta))


if __name__ == '__main__':
    common_args = arg_parser()
    model_args = load_config(common_args.dataset_name)
    main(common_args, model_args)
