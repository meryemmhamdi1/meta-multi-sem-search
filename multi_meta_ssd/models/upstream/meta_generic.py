from torch import nn
from multi_meta_ssd.log import *

class MetaGeneric(nn.Module):
    def __init__(self,
                 tokenizer,
                 base_model,
                 device,
                 meta_learn_config):

        super(MetaGeneric, self).__init__()

        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.meta_learn_config = meta_learn_config