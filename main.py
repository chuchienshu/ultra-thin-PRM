from datasets import pascal_voc_classification, image_transform, fetch_voc
from model import fc_resnet50
from prm.prm import peak_response_mapping
from losses import multilabel_soft_margin_loss
from tensorboardX import SummaryWriter
from solver import Solver
import os
import yaml, json
from utils import *
import PIL.Image
import argparse


def main(args):

    with open("config.yml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    train_trans = image_transform(**config['train_transform'])
    test_trans = image_transform(**config['test_transform'])

    config['dataset'].update({'transform': train_trans,
                              'target_transform': None})
    dataset = pascal_voc_classification(**config['dataset'])

    config['data_loaders']['dataset'] = dataset
    data_loader = fetch_voc(**config['data_loaders'])

    train_logger = SummaryWriter(log_dir = os.path.join(config['log'], 'train'), comment = 'training')

    solver = Solver(config)

    if args.train:
        solver.train(data_loader, train_logger)
    if args.run_demo:
        # Load demo images and pre-computed object proposals
        # change the idx to test different samples
        idx = 1
        raw_img = PIL.Image.open('./data/sample%d.jpg' % idx).convert('RGB')
        input_var = test_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
        with open('./data/sample%d.json' % idx, 'r') as f:
            proposals = list(map(rle_decode, json.load(f)))
        solver.inference(input_var, raw_img, 19, proposals=proposals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train','-T', type=bool, default=False, help='set train mode up')
    parser.add_argument('--run_demo','-I', type=bool, default=True, help='run demo')
    args = parser.parse_args()
    main(args)