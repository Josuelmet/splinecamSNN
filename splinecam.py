import networkx as nx
import numpy as np

import torch

from adjacency import SplineAdjacencyMatrix
from splinecam_utils import calc_W_b



# SplineCam-SNN Algorithm
def splinecam_snn(SNN, P, xlim, ylim, rtol=1e-4, atol=1e-4):

    # Initialize vertices V and their subgraphs As.
    A = SplineAdjacencyMatrix(xlim[0], xlim[1], ylim[0], ylim[1])

    for layer in range(len(SNN)):
        # for efficiency, we will store this array after the first calc_W_b call,
        # and then re-use it in other calc_W_b calls, until we move to a new layer.
        response_t0_to_t1_from_V = None
        for t in range(P.shape[1]): # P has shape (2, T, d)
            for neuron in range(SNN[layer].hidden_dim):
                # Get the center vertex of each cycle in A
                V_centers = torch.stack([A.V[a].mean(0) for a in A.cycles]) # shape = len(A.cycles), 2
                # Calculate the subthreshold activation linarity for each vertex in V_centers
                # For efficiency, we store and re-use response_t0_to_t1_from_V
                Ws, bs, response_t0_to_t1_from_V = calc_W_b(
                    SNN, P, V_centers, t, neuron, layer, response_t0_to_t1_from_V, rtol=rtol, atol=atol
                )
                #print(Ws, bs)

                # For each cycle in A.cycles, see if/how our neuron, at this time, bisects it. Update A accordingly.
                for i in range(len(A.cycles)):
                    # Get preacts for this area -> calculate signs, derivative of signs, and vertices to reuse or break.
                    cycle = torch.Tensor(A.cycles[i]).int()
                    V_th = SNN[layer].V_th if len(SNN[layer].V_th) == 1 else SNN[layer].V_th[neuron] # accomodate for vector V_th
                    #preacts    = (A.V[cycle] @ Ws[i] + bs[i] - V_th).real # accomodate for complex hidden states
                    preacts    = A.V[cycle] @ Ws[i] + bs[i] - V_th
                    signs      = preacts.sign() * ~torch.isclose(preacts, torch.zeros_like(preacts), rtol=rtol, atol=atol)
                    #signs      = (preacts * (preacts.abs() >= atol)).sign()
                    signs_diff = torch.cat((signs, torch.Tensor([signs[0]]))).diff()
                    if signs_diff.abs().sum() != 4:
                        continue # this should make sure that there is in fact a sign transition in A_i.
                        # TODO: try replacing the above code with -1 and +1 check from the paper
                    v_new_edge = cycle[signs == 0].int().tolist()     # we are interested in vertices that have a preactivation sign of 0.
                    break_idx = torch.where(signs_diff.abs() == 2)[0] # we are also interested in where either +1 -> -1, -1 -> +1, or 0.
                    assert len(v_new_edge) + len(break_idx) == 2 # there should be either 0 or 2 breakpoints per region

                    # Insert new vertex
                    if len(break_idx) >= 1:
                        # Calculate the new vertices
                        p1s, p2s = preacts[break_idx],  preacts[(break_idx + 1) % len(cycle)]
                        # alphas should not be < 0 or > 1. They also should not be = 0 or = 1. If they were, the resulting vertex should already be marked for reusage.
                        alphas = p2s / (p2s - p1s); assert all(alphas > 0) and all(alphas < 1)
                        v_break1, v_break2 = cycle[break_idx], cycle[(break_idx + 1) % len(cycle)]
                        for j in range(len(break_idx)):
                            v_new_edge.append(A.insert_new_vertex(v_break1[j].item(), v_break2[j].item(), alphas[j])) # add a new vertex, and add it to v_new_edge.

                    # Update A
                    A.draw_new_edge(*v_new_edge, i)

    return A




# -------------------------
# Accuracy-checking methods
# -------------------------

# Region-sign verification
# Make sure that, within each cycle of G, all vertices share the same signs at all times (excluding 0)
def check_region_signs(SNN, A, P, rtol=1e-5, atol=1e-5, verbose=False):
    for i, cycle in enumerate(A.cycles):
        shift = 0.1
        V_cycle = (1-shift) * A.V[cycle] + shift * A.V[cycle].mean(0)
        # Calculate h for all vertices in this cycle
        inputs = (V_cycle @ (P.flatten(1))).reshape((-1,) + P.shape[1:])
        _, all_spikes = SNN(inputs, batch_first=True, return_all=True)
        for layer in range(len(SNN)):
            spikes = all_spikes[layer] # spikes shape = (len(V_cycle), T, hidden_dim)
            for t in range(spikes.shape[1]):
                for neuron in range(spikes.shape[2]):
                    assert spikes[:,t,neuron].sum() in [0, len(V_cycle)], f"At {layer=}, {t=}, {neuron=}, signs in cycle {i} ({cycle}) do not agree: {spikes[:,t,neuron]}"
    if verbose:
        print("All regions agree on signs")


# Cycle-tracking verification (doesn't always work) (do NOT run this on larger graphs - it takes a while.)
def check_cycles(A, verbose=False):
    def orient(cycle):
        start = np.argmin(cycle)
        if cycle[(start + 1) % len(cycle)] < cycle[(start - 1) % len(cycle)]:
            return tuple(cycle[start:] + cycle[:start]) # forwards
        else:
            return tuple(cycle[:start+1][::-1] + cycle[start+1:][::-1]) # backwards

    A_c = set([orient(c) for c in A.cycles])
    nx_c = set([orient(c) for c in nx.minimum_cycle_basis(G)])
    A_not_in_nx = A_c - nx_c
    
    if verbose:
        print(A_not_in_nx)
    assert not len(A_not_in_nx)
    if verbose:
        print('nx and A agree on cycles')


# ------------------
# Plotting functions
# ------------------

set_axis_lims = lambda axis, xlim, ylim: (axis.set_xlim(xlim[0], xlim[1]), axis.set_ylim(ylim[0], ylim[1]))

def plot_splines(axis, A, xlim, ylim):
    set_axis_lims(axis, xlim, ylim)
    G = A.to_Graph()
    nx.draw_networkx_edges(G, pos={i: v.numpy() for i, v in enumerate(A.V.detach())}, ax=axis, hide_ticks=False)

def plot_vertices(axis, A, xlim, ylim, d=0.6):
    set_axis_lims(axis, (xlim[0]-d, xlim[1]+d), (ylim[0]-d, ylim[1]+d))
    G = A.to_Graph()
    nx.draw(G, pos={i: v.numpy() for i, v in enumerate(A.V.detach())}, with_labels=True, ax=axis)
