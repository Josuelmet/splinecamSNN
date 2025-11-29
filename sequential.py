import scipy
from tqdm import tqdm

import torch
from torch import nn

import snntorch
import spikingjelly.activation_based as sj
import spikingjelly.activation_based.neuron
import spikingjelly.activation_based.layer

from layer import LinearLIF, LIFBase
from utils import isanyinstance, clone



# Basic skip layer: computes out_mod(in_mod(x) + skip_mod(skip))
class SkipLayer(nn.Module):
    def __init__(self, in_mod, out_mod, skip_mod, keep_idx=None):
        super().__init__()
        self.in_mod   = in_mod
        self.out_mod  = out_mod
        self.skip_mod = skip_mod
        self.keep_idx = keep_idx

    def __iter__(self):
        return iter([self.in_mod, self.out_mod, self.skip_mod])

    def forward(self, x, skip):
        out = self.out_mod(self.skip_mod(skip) + self.in_mod(x))
        if self.keep_idx is not None:
            return out[self.keep_idx]
        return out

# Sequential module that can handle skip layers
class SkipSequential(nn.Sequential):
    def forward(self, x):
        x_init = clone(x)
        for m in self:
            x = m(x, x_init) if isinstance(m, SkipLayer) else m(x)
        return x



# Sequential module of LinearLIF layers.
# Contains code to:
#    (1) convert from a Sequential of SNNTorch, SpikingJelly, or LIFBase layers
#    (2) call forward(), optionally returning spikes and/or membrane potentials from all layers
#    (3) calculate local complexity of the network around an anchor point
class SequentialLIF(SkipSequential):

    # When initializing SequentialLIF objects directly via __init__, make sure every entry is a child of LIFBase.
    # forward() assumes that every layer is an LIFBase child.

    # Method to convert from a Sequential of SNNTorch, SpikingJelly, or LIFBase layers.
    # Assumes SNNTorch and SpikingJelly layers are linear LIF layers, which are standard in practice.
    @staticmethod
    def from_sequential(seq: nn.Sequential):
        modules = []
        i = 0
        while i < len(seq):
            # If the current module is already LIFBase, just add it and move on.
            if isinstance(seq[i], LIFBase):
                modules.append(seq[i])
                i += 1
                continue
            # Otherwise, first we identify input, output, and possibly skip modules...
            if isinstance(seq[i], SkipLayer):
                in_mod, out_mod, skip_mod = seq[i]
                i += 1
            else:
                in_mod, out_mod, skip_mod = seq[i], seq[i+1], None
                i += 2
            # ... then we convert them into LinearLIF layers, and append. 
            if isanyinstance(out_mod, snntorch.Leaky, snntorch.RLeaky):
                modules.append(LinearLIF.from_snntorch(in_mod, out_mod, skip_mod))
            elif isanyinstance(out_mod, sj.neuron.LIFNode, sj.neuron.ParametricLIFNode, sj.layer.LinearRecurrentContainer):
                modules.append(LinearLIF.from_spikingjelly(in_mod, out_mod, skip_mod))
            else:
                raise ValueError(f"Output module {out_mod} is not a supported LIF layer. Make sure LIF layers follow Linear layers.")
        return SequentialLIF(*modules)

    def forward(self, x, batch_first=False, return_all=False, return_mem=False):
        all_spikes = []
        all_mem = []
        x_init = clone(x)
        for module in self:
            assert isinstance(module, LIFBase) # we assume every layer is an LIF layer (w/ input, recurrent, and skip connections already included)
            x, mem = module(x, x_init, batch_first=batch_first) # spikes, membrane
            all_spikes.append(x)
            all_mem.append(mem)

        if return_all:
            if return_mem:
                return x, all_spikes, all_mem
            return x, all_spikes
        if return_mem:
            return x, mem
        return x

    def local_complexity(self, anchor, num_vecs, radius, projections=None):
        # 1) Input checks
        assert anchor.ndim in [2,3] # (T, d) or (T, batch=1, d)
        if anchor.ndim == 3:
            assert anchor.shape[1] == 1
            anchor = anchor.squeeze(1)
        dim = len(anchor.flatten())
        assert num_vecs <= dim

        # 2) Make orthonormal directions, repeated w/ different signs
        if projections is None:
            # Make a random orthonormal vectors of the same dimensionality as anchor
            vecs = torch.Tensor(scipy.linalg.orth(torch.randn(dim, num_vecs)))
        else:
            # Make random orthonormal linear combinations of projection vectors,
            # which are presumably orthonormal and of effective shape (num_projections, dim)
            projections = projections.flatten(1)
            assert num_vecs <= len(projections)
            vecs = torch.Tensor(scipy.linalg.orth(torch.randn(len(projections), num_vecs)))
            vecs = projections.T @ vecs
        # Repeat each column once, and scale by radius
        vecs = vecs.repeat_interleave(2,1) * radius
        # Flip sign of every other column
        vecs[:, 1::2] *= -1
        # Change shape from (dim, num_vecs*2) to (T, num_vecs*2, d)
        vecs = vecs.reshape(anchor.shape + (num_vecs*2,))
        vecs = vecs.transpose(1,2)

        # 3) simulate net w/ anchor (T, d) + directions (T, num_vecs*2, d)
        inputs = anchor.unsqueeze(1) + vecs
        # Pass thru net, and concatenate all outputs from all neurons as one dimension
        signs = torch.cat(self(inputs, batch_first=False, return_all=True)[1], dim=-1)
        # Change shape from (T, num_vecs*2, all neurons) to (num_vecs*2, T * all neurons)
        signs = signs.transpose(0,1).flatten(1)
        # Count number of unique (T * all neurons) vectors.
        # Since spiking neurons are already either 0 or 1, we do not need to calculate signs.
        return torch.unique(signs, dim=0).shape[0]

    
    def splinecam_approximation(self, proj_x, proj_y, radius, xlim, ylim, num):
        """
        Approximate splinecam-SNN along a 2D slice of input space defined by proj_x and proj_y.
        
        Inputs:
        - proj_x: (T, d) input projection (x-axis)
        - proj_y: (T, d) input projection (y-axis)
        - radius: float passed to local_complexity
        - xlim: tuple defining minimum and maximum scalings of proj_x
        - ylim: tuple defining minimum and maximum scalings of proj_y
        - num: number of steps to discretize xlim and ylim along
        """
    
        out = torch.zeros(num, num)
        for i, y in enumerate(tqdm(torch.linspace(ylim[1], ylim[0], num))):
            for j, x in enumerate(torch.linspace(xlim[0], xlim[1], num)):
                arr[i,j] = self.local_complexity(
                    proj_x * x + proj_y * y, num_vecs=2, radius=0.2,
                    projections = torch.stack((proj_x, proj_y))
                )
        return out
