import torch


#Calculate the contribution from vertex V to the membrane potential at each timestep
def calculate_per_timestep_responses(P: torch.Tensor, leak: torch.Tensor, W: torch.Tensor,
                                     verify=True, rtol=1e-5, atol=1e-5):
    """
    Calculate the contribution from vertex V to the membrane potential at each timestep.

    Inputs:
    - P: (2, T, d) projection matrix from a 2D vertex to an input time-series
    - leak: (hidden_dim,) leakage of each neuron
    - W: (hidden_dim, d) input weight matrix

    Outputs:
    - W_v: (T, hidden_dim, 2) input from V to the hidden membranes at each timestep (not accounting for inputs from previous timesteps)
    - response_t0_to_t1_from_V: (T, T, hidden_dim, 2)
      contribution, from input timestep t_0 to hidden membranes at time t_1, as a function of the input vertex V.
    """
    # P has shape (2, T (timesteps), d (input dimensions))
    T, d = P.shape[1:]
    hidden_dim, dtype = W.shape[0], leak.dtype
    # calculate the effective weight from V to the membrane at each timestep
    W_v = P @ W.T # shape = (2, T, hidden_dim), since W has shape (hidden_dim, d)
    W_v = W_v.transpose(0,1).transpose(1,2) # shape = (T, hidden_dim, 2)

    # calculate response from impulse at time t_0 to the membrane at time t_1.
    response_t0_to_t1 = torch.zeros(T, T, hidden_dim, dtype=dtype)
    for t in range(T):
        response_t0_to_t1[t,t:] = torch.stack([leak ** t for t in range(T-t)])
    # weight the per-timestep impulse repsonses by the input time-series
    # produced by a 2D vertex V via the timestep projection matrix P.
    # shape = (T, T, hidden_dim, 2)
    response_t0_to_t1_from_V = response_t0_to_t1.unsqueeze(-1) * W_v.unsqueeze(1)

    if verify:
        # Verifying that our subthreshold activations are correct
        n_rand = 100
        V_rand = torch.randn(n_rand, 2).to(dtype)
        # Let's assume the neuron does not spike.
        # Input at every timestep t_0 will affect all timesteps t_1, for any t_1 >= t_0.
        # Then, we can just sum this matrix along dimension 0 (t_0) to get the
        # subthreshold membrane at all times t_1, for all hidden neurons, as a function of any 2D input vertex.
        subthreshold_activity = response_t0_to_t1_from_V.sum(0) # shape = (T, hidden_dim, 2).
        our_hidden = (subthreshold_activity @ V_rand.T).transpose(1,2) # shape = (T, n_rand, 2)

        true_hidden = torch.zeros(T, n_rand, hidden_dim, dtype=dtype)
        for t in range(T):
            # recall that V_rand.shape = (n_rand, 2), P.shape = (2, T, d) and W.shape = (hidden_dim, d)
            true_hidden[t] = leak.unsqueeze(0) * true_hidden[t-1] + V_rand @ (P[:,t] @ W.T).to(dtype)
            # alternative: true_hidden[t] = V_rand @ W_v[t].T + leak.unsqueeze(0) * true_hidden[t-1]

        assert torch.allclose(our_hidden, true_hidden, rtol=rtol, atol=atol), \
        f"subthreshold activation calculation does not match the truth. Max error: {(our_hidden - true_hidden).abs().max()}"

    return response_t0_to_t1_from_V



# Piecewise-linearity calculation
def calc_W_b(net, P, Vs, t, neuron, layer, response_t0_to_t1_from_V=None, rtol=1e-5, atol=1e-5):
    """
    Calculate the effective W (1,2) and b (1,) from the vertices in Vs to a given neuron at time t at layer l.

    Inputs:
    - net: SequentialLIF
    - P: (2, T, d) projection matrix from a 2D vertex to an input time-series
    - Vs: (-1, 2) matrix of 2D vertices
    - t: timestep (int)
    - neuron: neuron index (int)
    - layer: layer index (int)
    - response_t0_to_t1_from_V: (optional) output from calculate_per_timestep_responses

    Outputs:
    - Ws: (-1, 2) matrix of effective locally linear weights
    - bs: (-1,) matrix of effective locally linear biases
    - response_t0_to_t1_from_V
    """

    # identify layer and calculate its input weights
    if response_t0_to_t1_from_V is None:
        lif = net[layer]
        W = torch.zeros_like(lif.W_in)
        if layer == 0:
            W = lif.W_in + (lif.W_skip if lif.W_skip is not None else 0)
        else:
            W = lif.W_skip if lif.W_skip is not None else torch.zeros(lif.W_in.shape[0], P.shape[-1]) # (h, d)
        response_t0_to_t1_from_V = calculate_per_timestep_responses(
            P=P, leak=lif.leak, W=W, rtol=rtol, atol=atol
        )
    # calculate spikes and membrane of layer at all times until t
    # Vs = (batch, 2) P = (2, T, d) --> inputs = (batch, T, d)
    inputs = (Vs @ (P.flatten(1))).reshape((-1,) + P.shape[1:])
    spikes, mem = net[:layer+1](inputs, batch_first=True, return_mem=True) # both shapes are (batch, T, hidden_dim)
    # we do not need the spikes at time t, only those from [0, 1, ..., t-1]
    spikes = spikes[:, :t, neuron] # (batch, t-1)
    # whereas we do need the membrane at time t.
    final_mem = mem[:, t, neuron] # (batch,)

    # initialize effective weights and biases from Vs to neuron at time t at layer.
    Ws = torch.zeros(len(Vs), 2, dtype=mem.dtype)

    # response_t0_to_t1_from_V shape = (T, T, hidden_dim, 2);
    # its [i,j] entry is effective input, from V, from time i to time j.
    if net[layer].reset == "hard" and t > 0:
        for i in range(len(Vs)):
            # calculate effective input, from V, from time s to time t,
            # where s is the first timestep after the latest spike.
            last_spike_time = spikes[i].nonzero().flatten()
            last_spike_time = last_spike_time[-1].item() if len(last_spike_time) > 0 else -1
            Ws[i] = response_t0_to_t1_from_V[last_spike_time+1:, t, neuron].sum(0)
    else:
        # If there are no hard resets, then the input, from V, to time t
        # is just the sum of all inputs from V from all times <= t.
        # By definition, there are no inputs from time i -> j if i > j,
        # so we can just sum this matrix along dim 0 (t_0, or i).
        Ws[:] = response_t0_to_t1_from_V.sum(0)[t, neuron]

    # Calculate biases as the difference between final membrane and weighted inputs
    bs = final_mem - (Ws * Vs).sum(1)
    return Ws.real, bs.real, response_t0_to_t1_from_V
