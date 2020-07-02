from __future__ import print_function, absolute_import
from reid.volta import *
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

from reid.utils.logging import Logger
import os.path as osp
import sys
from torch.backends import cudnn
from reid.utils.serialization import load_checkpoint
from torch import nn
import time
import pickle



def resume(args):
    import re
    pattern=re.compile(r'step_(\d+)\.ckpt')
    start_step = -1
    ckpt_file = ""

    # find start step
    files = os.listdir(args.logs_dir)
    files.sort()
    for filename in files:
        try:
            iter_ = int(pattern.search(filename).groups()[0])
            if iter_ > start_step:
                start_step = iter_
                ckpt_file = osp.join(args.logs_dir, filename)
        except:
            continue

    # if need resume
    if start_step >= 0:
        print("continued from iter step", start_step)

    return start_step, ckpt_file

def main(args):
    cudnn.benchmark = True
    cudnn.enabled = True
    save_path = args.logs_dir

    sys.stdout = Logger(osp.join(args.logs_dir, 'log'+ str(args.EF)+ time.strftime(".%m_%d_%H:%M:%S") + '.txt'))

    # get all the labeled and unlabeled data for training
    dataset_all = datasets.create(args.dataset, osp.join(args.data_dir, args.dataset))
    num_all_examples = len(dataset_all.train) 
    l_data, u_data = get_one_shot_in_cam1(dataset_all, load_path="./examples/oneshot_{}_used_in_paper.pickle".format(dataset_all.name))

    resume_step, ckpt_file = -1, ''
    if args.resume:
        resume_step, ckpt_file = resume(args) 

    # initialize the VOLTA algorithm 
    volta = VOLTA(model_name=args.arch, batch_size=args.batch_size, num_classes=dataset_all.num_train_ids, 
            data_dir=dataset_all.images_dir, l_data=l_data, u_data=u_data, save_path=args.logs_dir, max_frames=args.max_frames, 
            worker=args.worker)

    num_u = len(u_data)
    num_to_select = 0
    is_end = False

    new_train_data = l_data 
    for step in range(1000):
        # for resume
        if step < resume_step: 
            continue

        print("This is running VOLTA with step {}:\t  Logs-dir {}".format(step,  save_path))
        
        # train the model or load ckpt        
        volta.train(new_train_data, step, epochs=args.epoches, step_size=args.step_size, warmup_epoch=5, \
                        init_lr=args.init_lr, beta=args.beta) if step != resume_step else volta.resume(ckpt_file, step)

        # pseudo-label and confidence score
        pred_y, pred_score = volta.estimate_label(args.alpha)

        # threshold based selection
        if(num_to_select == num_u):
            is_end = True
        thrshd = args.threshold - step * args.decreasing_rate
        num_to_select_last_step = num_to_select
        num_to_select = (pred_score > thrshd).sum()
        if(num_to_select - num_to_select_last_step) < num_u * args.end_ratio:
            num_to_select = num_u
        volta.num_to_select = num_to_select

        # select data
        selected_idx = volta.select_top_data(pred_score, num_to_select)

        # add new data
        new_train_data = volta.generate_new_train_data(selected_idx, pred_y)

        # evluate
        volta.evaluate(dataset_all.query, dataset_all.gallery, args.alpha)

        if(is_end):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative Local-Global Collaboration Learning')
    parser.add_argument('-d', '--dataset', type=str, default='DukeMTMC-VideoReID',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-a', '--arch', type=str, default='local_global', choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-l', '--init_lr', type=float)
    parser.add_argument('--EF', type=int, default=10)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='./data')
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir,'logs'))
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_frames', type=int, default=900)
    parser.add_argument('--worker', type=int, default=6)
    parser.add_argument('--epoches', type=int, default=70)
    parser.add_argument('--step_size', type=int, default=55)

    parser.add_argument('--threshold', type=float, default=0.82)
    parser.add_argument('--decreasing_rate', type=float, default=0.02)
    parser.add_argument('--end_ratio', type=float, default=0.006)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--beta', type=float, default=0.003)
    
    
    main(parser.parse_args())
