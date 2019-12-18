import torch
from utils.rnns import (mean_pooling,
                    max_pooling,
                    gather_last)


class MeanPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MeanPoolingLayer, self).__init__()


    def forward(self, batch_hidden_states, lengths, **kwargs):
        return mean_pooling(batch_hidden_states, lengths)


class MaxPoolingLayer(torch.nn.Module):

    def __init__(self):
        super(MaxPoolingLayer, self).__init__()

    def forward(self, batch_hidden_states, lengths, **kwargs):
        return max_pooling(batch_hidden_states, lengths)


class GatherLastLayer(torch.nn.Module):

    def __init__(self, bidirectional=False):
        super(GatherLastLayer, self).__init__()
        self.bidirectional = bidirectional

    def forward(self, batch_hidden_states, lengths, **kwargs):
        return gather_last(batch_hidden_states, lengths,
                           bidirectional=self.bidirectional)


class GatherFirstLayer(torch.nn.Module):

    def __init__(self):
        super(GatherFirstLayer, self).__init__()

    def forward(self, batch_hidden_states, **kwargs):
        return batch_hidden_states[:,0,:]
