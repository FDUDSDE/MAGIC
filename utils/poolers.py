import torch.nn as nn
import torch
import numpy as np


class Pooling(nn.Module):
    def __init__(self, pooler):
        super(Pooling, self).__init__()
        self.pooler = pooler

    def forward(self, graph, feat, n_types=None):
        # Implement node type-specific pooling
        with graph.local_scope():
            if not n_types:
                if self.pooler == 'mean':
                    return feat.mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat.sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat.max(0, keepdim=True)
                else:
                    raise NotImplementedError
            else:
                result = []
                for i in range(n_types):
                    mask = (graph.ndata['type'] == i)
                    if not mask.any():
                        result.append(torch.zeros((1, feat.shape[-1]), device=feat.device))
                    elif self.pooler == 'mean':
                        result.append(feat[mask].mean(0, keepdim=True))
                    elif self.pooler == 'sum':
                        result.append(feat[mask].sum(0, keepdim=True))
                    elif self.pooler == 'max':
                        result.append(feat[mask].max(0, keepdim=True))
                    else:
                        raise NotImplementedError
                result = torch.cat(result, dim=-1)
                return result
                
