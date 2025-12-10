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

    def local_complexity(self, anchors, num_vecs, radius, projections=None, batch_first=True):
        """
        Estimate local complexity of piecewise-constant SNN (self) at anchor points.

        Inputs:
        - anchors: (T, d) if 2D, otherwise (num anchors, T, d) if batch_first is True else (T, num anchors, d)
        - num_vecs: int, should be less than T*d or len(projections)
        - radius: float
        - projections: iterable of Tensors of shape (T, d)
        - batch_first: bool

        Output:
        - An integer Tensor of shape (num anchors,)
        """
        # 1) Define anchors, a (batch, T, d) Tensor.
        assert anchors.ndim in [2,3]
        if anchors.ndim == 2: # (T, d)
            anchors = anchors.unsqueeze(0) # (1, T, d)
            batch_first = True
        elif not batch_first: # (T, batch, d)
            anchors = anchors.transpose(0,1) # (batch, T, d)
        batch, T, d = anchors.shape

        # 2) Make orthonormal directions
        if projections is None:
            # Make a random orthonormal vectors of the same dimensionality as anchor
            assert num_vecs <= T*d
            # shape = (batch, T*d, num_vecs)
            vecs = torch.stack([
                torch.Tensor(scipy.linalg.orth(torch.randn(T*d, num_vecs))) for _ in range(batch)
            ])
        else:
            # Make random orthonormal linear combinations of projection vectors,
            # which are presumably orthonormal and of shape (num_projections, T, d)
            projections = projections if isinstance(projections, torch.Tensor) else torch.stack(projections)
            assert projections.ndim == 3 and projections.shape[1:] == (T, d)
            projections = projections.flatten(1) # shape = (num projections, T*d)
            # make random orthonormal combinations of the projection vectors
            assert num_vecs <= len(projections)
            # shape = (batch, num projections, num_vecs)
            vecs = torch.stack([
                torch.Tensor(scipy.linalg.orth(torch.randn(len(projections), num_vecs))) for _ in range(batch)
            ])
            # for each (len(projections) x num_vecs) 2D matrix in vecs, matrix-multiply it
            # with projections (len(projections) x dim) to get a (dim x num_vecs) 2D matrix.
            vecs = projections.T.unsqueeze(0) @ vecs

        # 3) Duplicate the orthonormal directions, multiply the duplicates by -1, and scale all by radius.
        # repeat twice along the last dimension. shape = (batch, dim=T*d, num_vecs*2)
        vecs = vecs.repeat_interleave(2,-1)
        # the number of vectors has been duplicated so we can multiply every other one by -1.
        vecs[..., 1::2] *= -1
        # scale by radius
        vecs *= radius
        # shape = (batch, num_vecs*2, dim=T*d)
        vecs = vecs.transpose(1,2)
        # shape = (batch * num_vecs * 2, T, d)
        vecs = vecs.reshape((-1,) + anchors.shape[1:])

        # 4) simulate net w/ anchors (batch, T, d) + directions (batch * num_vecs * 2, T, d)
        # repeat anchors to be same shape as vecs, then add them
        inputs = vecs + anchors.repeat_interleave(num_vecs*2, 0)
        # Pass into net, and concatenate activity from all neurons as one dimension
        # shape = (batch * num_vecs * 2, T, all neurons)
        signs = torch.cat(self(inputs, batch_first=True, return_all=True)[1], dim=-1)
        # organize signs along batch dimension, and merge the timestep and neuron dimensions
        signs = signs.reshape(batch, num_vecs*2, -1)

        return torch.Tensor([len(mat.unique(dim=0)) for mat in signs]).int() # shape = (batch,)


    def local_complexity_batched(self, anchors, num_vecs, radius, projections=None, batch_first=True, batch_size=-1):
        """
        Calls local_complexity, but breaking the computation up into batch_size elements from anchors at a time.
        """
        out = torch.zeros(anchors.shape[0] if batch_first == True else anchors.shape[1])
        batch_size = len(out) if batch_size < 1 else batch_size
        for i in range(len(out) // batch_size):
            out[i*batch_size : (i+1)*batch_size] = self.local_complexity(
                anchors=anchors, num_vecs=num_vecs, radius=radius, projections=projections, batch_first=batch_first
            )
        return out
            
    
    def splinecam_approximation(self, proj_x, proj_y, radius, xlim, ylim, num, batch_size=-1):
        """
        Approximate splinecam-SNN along a 2D slice of input space defined by proj_x and proj_y.
        
        Inputs:
        - proj_x: (T, d) input projection (x-axis)
        - proj_y: (T, d) input projection (y-axis)
        - radius: float passed to local_complexity
        - xlim: tuple defining minimum and maximum scalings of proj_x
        - ylim: tuple defining minimum and maximum scalings of proj_y
        - num: number of steps to discretize xlim and ylim along
        - batch_size: number of points to compute local_complexity on at a time

        Outputs:
        - A 2D (num, num) Tensor of local complexities
        """

        x = torch.linspace(xlim[0], xlim[1], num)
        y = torch.linspace(ylim[0], ylim[1], num)
        xx, yy = torch.meshgrid(x, y.flip(0), indexing='xy')
        anchors = xx.reshape(-1,1,1) * proj_x + yy.reshape(-1,1,1) * proj_y
        
        return self.local_complexity_batched(
            anchors=anchors, num_vecs=2, radius=radius, projections=(proj_x, proj_y), batch_first=True, batch_size=batch_size
        )
