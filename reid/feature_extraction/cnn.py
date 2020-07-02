from __future__ import absolute_import
from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from ..utils import to_torch


def extract_cnn_feature(model, inputs, modules=None):
    model.eval()    
    with torch.no_grad():
        inputs = to_torch(inputs)
        if modules is None:
            outputs = model(inputs)
            vid_outputs = F.normalize(outputs[0].detach(), p=2, dim=1)  
            img_outputs = F.normalize(outputs[1].detach(), p=2, dim=2) 
            return vid_outputs, img_outputs
            
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o): outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
        return list(outputs.values())
