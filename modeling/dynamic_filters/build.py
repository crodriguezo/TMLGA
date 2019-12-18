import torch
import numpy as np
from torch import nn

import modeling.dynamic_filters as DF
import utils.pooling as POOLING

class DynamicFilter(nn.Module):
    def __init__(self, cfg):
        super(DynamicFilter, self).__init__()
        self.cfg = cfg

        factory = getattr(DF, cfg.DYNAMIC_FILTER.TAIL_MODEL)
        self.tail_df = factory(cfg)

        factory = getattr(POOLING, cfg.DYNAMIC_FILTER.POOLING)
        self.pooling_layer = factory()

        factory = getattr(DF, cfg.DYNAMIC_FILTER.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths=None):
        output, _ = self.tail_df(sequences, lengths)
        output = self.pooling_layer(output, lengths)
        output = self.head_df(output)
        return output, lengths 
