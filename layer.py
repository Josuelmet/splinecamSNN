from abc import ABC, abstractmethod

import torch
from torch import nn

import snntorch
import spikingjelly.activation_based as sj
import spikingjelly.activation_based.neuron
import spikingjelly.activation_based.layer

from utils import isanyinstance, clone



# Helper method
def tensor(nums):
    try:
        iter(nums)
        return torch.Tensor(nums).reshape(-1)
    except TypeError:
        return torch.Tensor([nums])


# Abstract LIF base class.
# Defines initialization and forward pass of LIF neurons.
class LIFBase(nn.Module, ABC):
    def __init__(self, W_in, W_rec, leak, reset, V_th=1, W_skip=None):
        super().__init__()
        self.W_in = clone(W_in)
        self.W_rec = clone(W_rec)
        self.leak = clone(tensor(leak))
        assert torch.all(self.leak.abs() <= 1)
        assert reset in ["hard", "subtract_leak", "subtract_noleak"]
        self.reset = reset
        self.V_th = clone(tensor(V_th))
        self.W_skip = clone(W_skip)
        self.hidden_dim = self.W_rec.shape[0]
        for w in [self.W_in] + ([] if self.W_skip is None else [self.W_skip]):
            assert w.shape[0] == self.hidden_dim


    def __repr__(self):
        repr = f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, reset={self.reset}"
        if self.W_skip is not None:
            return repr + f", W_skip)"
        return repr + ")"

    @abstractmethod
    def calc_membrane(self, hidden):
        pass

    def calc_reset(self, new_hidden, spikes):
        if self.reset == "hard":
            return new_hidden * spikes
        if self.reset == "subtract_noleak":
            return spikes * self.V_th
        if self.reset == "subtract_leak":
            return spikes * self.V_th * self.leak

    def calc_membrane_and_reset(self, hidden, spikes):
        new_hidden = self.calc_membrane(hidden)
        return new_hidden - self.calc_reset(new_hidden, spikes)

    def forward(self, x, x_skip=None, batch_first=False):
        assert len(x.shape) == 3, f"Input shape {x.shape} is not of length 3"
        # x is by default (time, batch, input)
        if batch_first:
            x = x.transpose(0, 1) # switch to default shape, timesteps first
            x_skip = x_skip.transpose(0, 1) # same as above
        hidden = torch.zeros(x.shape[:2] + (self.hidden_dim,), dtype=self.leak.dtype)
        spikes = torch.zeros(hidden.shape)
        for t in range(x.shape[0]):
            hidden[t] = x[t] @ self.W_in.T + spikes[t-1] @ self.W_rec.T
            if x_skip is not None and self.W_skip is not None:
                hidden[t] += x_skip[t] @ self.W_skip.T
            if t > 0:
                hidden[t] += self.calc_membrane_and_reset(hidden[t-1], spikes[t-1])
            spikes[t] = (hidden[t].real >= self.V_th).float()
        if batch_first:
            return spikes.transpose(0, 1), hidden.transpose(0, 1)
        return spikes, hidden


# Subthreshold-linear LIF class.
# Mainly contains code to convert from SNNTorch and SpikingJelly layers.
class LinearLIF(LIFBase):

    def calc_membrane(self, hidden):
        return self.leak * hidden

    @staticmethod
    def from_snntorch(linear: nn.Linear, lif, skip=None):
        # linear and skip layer checks
        for module in [linear] + ([] if skip is None else [skip]):
            assert isinstance(module, nn.Linear)
            assert module.bias is None
        # lif layer checks
        assert isanyinstance(lif, snntorch.Leaky, snntorch.RLeaky)
        assert not lif.inhibition # inhibition makes sure only the highest-membrane neuron fires
        assert lif.graded_spikes_factor == 1 # ensure output spikes are all 1
        assert lif._reset_mechanism in ["zero", "subtract"] # third option is "none"
        # snntorch.RLeaky zeros out contributions from previous spikes
        # if reset_mechanism is zero and reset_delay is True
        if lif._reset_mechanism == "zero" and isinstance(lif, snntorch.RLeaky):
            assert not lif.reset_delay
        else:
            assert lif.reset_delay
        #assert lif.reset_delay
        if type(lif) == snntorch.RLeaky:
            assert lif.all_to_all # ensure recurrent weights are dense + square
            assert lif.linear_features == linear.out_features # check shapes
            assert lif.recurrent.bias is None # ensure no recurrent bias

        return LinearLIF(
            W_in = linear.weight.data,
            W_rec = lif.recurrent.weight.data if isinstance(lif, snntorch.RLeaky) else \
                    torch.zeros((linear.out_features,) * 2),
            leak = lif.beta,
            reset = "hard" if lif._reset_mechanism == "zero" else "subtract_noleak",
            V_th = lif.threshold,
            W_skip = None if skip is None else skip.weight.data
        )


    @staticmethod
    def from_spikingjelly(linear: nn.Linear, second_layer, skip=None):
        # linear layer checks
        for module in [linear] + ([] if skip is None else [skip]):
            assert isinstance(module, nn.Linear)
            assert module.bias is None
        # lif layer checks
        if isinstance(second_layer, sj.layer.LinearRecurrentContainer):
            lif = second_layer.sub_module
            assert second_layer.rc.bias is None # ensure no recurrent bias
        else:
            lif = second_layer
        assert isanyinstance(lif, sj.neuron.LIFNode, sj.neuron.ParametricLIFNode)
        assert lif.v_reset in [0, None]

        # W_in and W_rec assignment
        W_in = clone(linear.weight.data)
        W_skip = clone(skip.weight.data) if skip is not None else None
        if isinstance(second_layer, sj.neuron.BaseNode):
            W_rec = torch.zeros((linear.out_features, ) * 2)
        else:
            # The LinearRecurrentContainer in spikingjelly is a bit odd.
            # Its weight matrix rc encodes both input[t] -> spike[t] and spike[t-1] -> spike[t].
            hidden_dim = linear.out_features
            W_rec = clone(second_layer.rc.weight.data[:, hidden_dim:])
            W_in = clone(second_layer.rc.weight.data[:, :hidden_dim]) @ W_in
            if W_skip is not None:
                W_skip = clone(second_layer.rc.weight.data[:, :hidden_dim]) @ W_skip


        # calculate leak and W_in
        leak = 1 - (1 / lif.tau if isinstance(lif, sj.neuron.LIFNode) else lif.w.sigmoid())
        if lif.decay_input:
            W_in *= 1 - leak
            W_rec *= 1 - leak
            if W_skip is not None:
                W_skip *= 1 - leak

        return LinearLIF(
            W_in = W_in, leak = leak, W_rec = W_rec,
            reset = "subtract_leak" if lif.v_reset is None else "hard", # see spikingjelly documentation
            V_th = lif.v_threshold,
            W_skip = W_skip
        )


# Exponential integrate-and-fire, for completeness
class EIF(LIFBase):
    def __init__(self, *args, sharpness, **kwargs):
        super().__init__(*args, **kwargs)
        # calculate v_rh such that calc_membrane = V_th when hidden = V_th
        self.sharpness = clone(tensor(sharpness))
        self.v_rh = self.V_th - self.sharpness * (torch.log(self.V_th/self.leak - self.V_th) - torch.log(self.sharpness))

    def calc_membrane(self, hidden):
        hidden = torch.min(hidden, self.V_th) # prevents instability when hidden > V_th
        return self.leak * (hidden + self.sharpness * torch.exp((hidden - self.v_rh) / self.sharpness))
