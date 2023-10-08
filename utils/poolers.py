import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self, pooler):
        super(Pooling, self).__init__()
        self.pooler = pooler

    def forward(self, graph, feat, t=None):
        feat = feat
        # Implement node type-specific pooling
        with graph.local_scope():
            if t is None:
                if self.pooler == 'mean':
                    return feat.mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat.sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat.max(0, keepdim=True)
                else:
                    raise NotImplementedError
            elif isinstance(t, int):
                mask = (graph.ndata['type'] == t)
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError
            else:
                mask = (graph.ndata['type'] == t[0])
                for i in range(1, len(t)):
                    mask |= (graph.ndata['type'] == t[i])
                if self.pooler == 'mean':
                    return feat[mask].mean(0, keepdim=True)
                elif self.pooler == 'sum':
                    return feat[mask].sum(0, keepdim=True)
                elif self.pooler == 'max':
                    return feat[mask].max(0, keepdim=True)
                else:
                    raise NotImplementedError
