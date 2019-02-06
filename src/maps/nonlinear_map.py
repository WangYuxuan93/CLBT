import torch
import logging
from torch import nn

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class NonLinearMap(nn.Module):

    def __init__(self, args):
        super(NonLinearMap, self).__init__()

        self.activation = args.activation
        self.emb_dim = args.emb_dim
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        logger.info("Non-linear mapping:\nActivation:{}\nLayers:{}\nHidden Size:{}".format(self.activation, 
                    self.n_layers, self.hidden_size))

        if args.activation == 'leaky_relu':
            activate = nn.LeakyReLU(0.1)
        elif args.activation == 'tanh':
            activate = nn.Tanh()
        else:
            print ("Activation type: {} not defined!".format(parmas.activation))
            exit(1)
        layers = []
        for i in range(self.n_layers):
            input_dim = self.emb_dim if i == 0 else self.hidden_size
            output_dim = self.emb_dim if i == self.n_layers-1 else self.hidden_size
            layers.append(nn.Linear(input_dim, output_dim, bias=True))
            if i < self.n_layers-1:
                layers.append(activate)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        #assert x.dim() == 2 and x.size(1) == self.emb_dim
        assert x.size(-1) == self.emb_dim
        return self.layers(x)