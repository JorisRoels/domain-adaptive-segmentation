"""
    This is a script that illustrates segmentation of a dataset with a particular model
"""
import torch

"""
    Necessary libraries
"""
import argparse
import yaml
import time

from neuralnets.util.io import print_frm, read_pngseq
from neuralnets.util.tools import set_seed
from neuralnets.util.validation import segment_read, segment_ram

from util.tools import parse_params, process_seconds
from networks.factory import generate_model

from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the network configuration file", type=str,
                        default='../train_supervised.yaml')
    parser.add_argument("--model", "-m", help="Path to the network parameters", type=str, required=True)
    parser.add_argument("--dataset", "-d", help="Path to the dataset that needs to be segmented", type=str,
                        required=True)
    parser.add_argument("--block_wise", "-bw", help="Flag that specifies to compute block wise or not",
                        action='store_true', default=False)
    parser.add_argument("--output", "-o", help="Path to store the output segmentation", type=str, required=True)
    parser.add_argument("--gpu", "-g", help="GPU device for computations", type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as file:
        params = parse_params(yaml.load(file, Loader=yaml.FullLoader))

    """
    Fix seed (for reproducibility)
    """
    set_seed(params['seed'])

    """
        Build the network
    """
    print_frm('Building the network')
    net = generate_model(params['method'], params)
    print_frm('Loading model parameters')
    net.load_state_dict(torch.load(args.model, map_location=torch.device('cuda:' + str(args.gpu)))['state_dict'])

    """
        Load the data if necessary (can also be done block-wise)
    """
    print_frm('Loading the data')
    if not args.block_wise:
        x = read_pngseq(args.dataset)

    """
        Segment the dataset
    """
    print_frm('Segmenting the data')
    t_start = time.perf_counter()
    if args.block_wise:
        segment_read(args.dataset, net.get_unet(load_best=False), params['input_size'], write_dir=args.output,
                     write_probs=True, in_channels=params['in_channels'], batch_size=params['test_batch_size'],
                     track_progress=True, device=args.gpu)
    else:
        segment_ram(x, net.get_unet(load_best=False), params['input_size'], write_dir=args.output, write_probs=True,
                    in_channels=params['in_channels'], batch_size=params['test_batch_size'], track_progress=True,
                    device=args.gpu)
    t_stop = time.perf_counter()
    print_frm('Elapsed segmentation time: %d hours, %d minutes, %.2f seconds' % process_seconds(t_stop - t_start))
