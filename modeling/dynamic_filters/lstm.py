import torch
import torch.nn as nn
from utils.rnns import feed_forward_rnn

class LSTM(nn.Module):

    def __init__(self, cfg):
        super(LSTM, self).__init__()
        self.input_size   = cfg.DYNAMIC_FILTER.LSTM.INPUT_SIZE
        self.num_layers   = cfg.DYNAMIC_FILTER.LSTM.NUM_LAYERS
        self.hidden_size  = cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE
        self.bias         = cfg.DYNAMIC_FILTER.LSTM.BIAS
        self.dropout      = cfg.DYNAMIC_FILTER.LSTM.DROPOUT
        self.bidirectional= cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL
        self.batch_first  = cfg.DYNAMIC_FILTER.LSTM.BATCH_FIRST

        self.lstm = nn.GRU(input_size   = self.input_size,
                            hidden_size  = self.hidden_size,
                            num_layers   = self.num_layers,
                            bias         = self.bias,
                            dropout      = self.dropout,
                            bidirectional= self.bidirectional,
                            batch_first = self.batch_first)

    def forward(self, sequences, lengths):
        if lengths is None:
            raise "ERROR in this tail you need lengths of sequences."
        return feed_forward_rnn(self.lstm,
                                sequences,
                                lengths=lengths)

