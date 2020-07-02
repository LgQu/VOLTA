from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import pickle

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter

import tqdm
import time
from torch.backends import cudnn


def extract_features(model, data_loader, print_freq=1, metric=None):
    cudnn.benchmark = False
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
 
    vid_features = OrderedDict()
    img_features = OrderedDict()
    labels = OrderedDict()    
    
    with tqdm.tqdm(total=len(data_loader)) as pbar:   
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            outputs = extract_cnn_feature(model, imgs)
            for fname, vid_output, img_output, pid in zip(fnames, outputs[0], outputs[1], pids):
                vid_features[fname] = vid_output
                img_features[fname] = img_output
                labels[fname] = pid
            pbar.update(1)

    print("Extract {} batch videos".format(len(data_loader)))
    cudnn.benchmark = True
    return vid_features, img_features, labels


def pairwise_distance_global(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([features["".join(f)].unsqueeze(0) for f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist.cpu()
    
    
def pairwise_distance_local(features, query=None, gallery=None, metric=None, use_gpu=True):
    if query is None and gallery is None:
        pass    

    q_img_feas = [features["".join(f)] for f, _, _, _ in query]
    g_img_feas = [features["".join(f)] for f, _, _, _ in gallery]
    q_num_tracklet = len(q_img_feas)
    g_num_tracklet = len(g_img_feas) 
        
    min_dist = torch.zeros((q_num_tracklet, g_num_tracklet))
    min_dist = min_dist.cuda() if use_gpu else min_dist
    # euclidean distance
    with tqdm.tqdm(total=q_num_tracklet) as pbar:
        for qq in range(q_num_tracklet):
            q_img_fea = q_img_feas[qq].cuda() if use_gpu else q_img_feas[qq]
            m = q_img_fea.size(0)
            for gg in range(g_num_tracklet):
                g_img_fea = g_img_feas[gg].cuda() if use_gpu else g_img_feas[gg]
                n = g_img_fea.size(0)
                dist_set2set = torch.pow(q_img_fea, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                            torch.pow(g_img_fea, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                dist_set2set.addmm_(1, -2, q_img_fea,g_img_fea.t()) 
                min_dist[qq][gg] = dist_set2set.min()

            pbar.update(1)

    min_dist = min_dist.cpu() if use_gpu else min_dist
    return min_dist


def evaluate_all(distmat, log_path, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
#        'allshots': dict(separate_camera_set=False,
#                         single_gallery_shot=False,
#                         first_match_break=False),
#        'cuhk03': dict(separate_camera_set=True,
#                       single_gallery_shot=True,
#                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, 
#                      cmc_scores['allshots'][k - 1],
#                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))
                      
    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['market1501'][0]


class Evaluator(object):
    def __init__(self, model, log_path, alpha):
        super(Evaluator, self).__init__()
        self.model = model
        self.log_path = log_path
        self.step = 0
        self.alpha = alpha

    def evaluate(self, data_loader, query, gallery, train_batch_size, metric=None):
        vid_features, img_features, _ = extract_features(self.model, data_loader, train_batch_size)
        distmat_g = pairwise_distance_global(vid_features, query=query, gallery=gallery, metric=metric)
        distmat_l = pairwise_distance_local(img_features, query=query, gallery=gallery, metric=metric)  

        return evaluate_all(self.alpha * distmat_g + (1 - self.alpha) * distmat_l, self.log_path, query=query, gallery=gallery)

