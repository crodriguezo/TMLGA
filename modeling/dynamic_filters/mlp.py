import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        self.input_dim = cfg.DYNAMIC_FILTER.MLP.INPUT_DIM
        self.output_dim = cfg.DYNAMIC_FILTER.MLP.OUTPUT_DIM

        self.mlp = nn.Sequential(
                        nn.Linear(self.input_dim, self.output_dim),
                        nn.Tanh()
                        )

    def forward(self, sentence_embed):
        return self.mlp(sentence_embed)
