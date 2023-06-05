import argparse
import os
import sys
from model import Model
import torch
import random
import numpy

isLoadModel = False
LOAD_MODEL_PATH = ""

def setRandomSeed(self):
    numpy.random.seed(self.args.seed)
    torch.manual_seed(self.args.seed)
    torch.cuda.manual_seed(self.args.seed)
    random.seed(self.args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MISRec')
    #dataset params
    parser.add_argument('--dataset', type=str, default="Yelp")
    parser.add_argument('--seed', type=int, default=29)

    parser.add_argument('--hide_dim', type=int, default=8)
    parser.add_argument('--layer', type=str, default="[8]")
    parser.add_argument('--slope', type=float, default=0.4)

    parser.add_argument('--reg', type=float, default=0.05)
    parser.add_argument('--decay', type=float, default=0.98)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--minlr', type=float, default=0.0001)
    parser.add_argument('--test_batch', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=1000)
    #early stop params
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--ng_number', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--fuse', type=str, default="mean")

    parser.add_argument('--dgi_graph_act', type=str, default="sigmoid")
    parser.add_argument('--lam', type=str, default='[0.1,0.001]')
    parser.add_argument('--clear', type=int, default=0)
    parser.add_argument('--subNode', type=int, default=10)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=1)


    parser.add_argument('--time_step', type=float, default=360)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--startTest', type=int, default=0)
    parser.add_argument('--datasetDir', type=str, default="/home/MISR/dataset/Yelp")



    args = parser.parse_args()
    print(args)

    Par = Model(args, isLoadModel)

    print('ModelNmae = ' + Par.getModelName())

    Par.run()
    Par.test()
