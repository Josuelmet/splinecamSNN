import networkx
import numpy as np
from scipy import sparse

import torch


class SplineAdjacencyMatrix:

    def __init__(self, xleft=-1, xright=1, ydown=-1, yup=1):
        assert xleft < xright and ydown < yup
        self.V = torch.Tensor([[xright,yup], [xright, ydown], [xleft, ydown], [xleft, yup]])
        self.cycles = [[0, 1, 2, 3]]
        self.edge_to_cycles = {(0,1):(0,), (0,3):(0,), (1,2):(0,), (2,3):(0,)}

    def sort_and_check_and_get_edge(self, v1, v2):
        # Check that v1, v2 are valid vertices
        v1, v2 = sorted((v1, v2))
        assert all([isinstance(v_i, int) and v_i >= 0 and v_i < self.V.shape[0] for v_i in (v1, v2)]), f"({v1},{v2}) are not ints"
        # If edge (v1, v2) exists, also return the cycles it is a part of. Otherwise, also return an empty tuple.
        try:    their_cycles = self.edge_to_cycles[(v1,v2)]
        except: their_cycles = tuple()
        return v1, v2, their_cycles


    def insert_new_vertex(self, v1, v2, alpha):
        # Check that alpha*V[v1] + (1-alpha)*V[v2] is valid. Check that v1, v2 are valid. Check that edge (v1,v2) exists.
        assert alpha > 0 and alpha < 1, f"invalid alpha: 0 < {alpha} < 1 is not True"
        V_new_val = alpha * self.V[v1] + (1 - alpha) * self.V[v2]
        v1, v2, affected_cycles = self.sort_and_check_and_get_edge(v1, v2)
        assert len(affected_cycles) > 0, f"edge ({v1},{v2}) is not recognized in edge_to_cycles"

        # Update cycles
        vnew = self.V.shape[0]
        for c_idx in affected_cycles:
            c = self.cycles[c_idx]
            c_break_indices = tuple(sorted((c.index(v1), c.index(v2))))
            # Insert vnew at c_break_indices[1], unless the pair of indices is (0, len(c)-1), in which case we just add vnew at the end.
            assert c_break_indices[1] - c_break_indices[0] == 1 or c_break_indices == (0, len(c)-1), f"({v1},{v2}) are not adjacent in {c}"
            c.insert(c_break_indices[1] + (c_break_indices == (0, len(c)-1)), vnew)

        # Update edge_to_cycles
        self.edge_to_cycles[(v1, vnew)] = self.edge_to_cycles[(v1, v2)]
        self.edge_to_cycles[(v2, vnew)] = self.edge_to_cycles[(v1, v2)]
        self.edge_to_cycles.pop((v1, v2))

        # Update V
        self.V = torch.vstack((self.V, V_new_val.unsqueeze(0)))
        return vnew


    def draw_new_edge(self, v1, v2, cycle_idx):
        # Check that v1, v2 are valid. Check that edge (v1,v2) does not exist.
        v1, v2, existing_cycles = self.sort_and_check_and_get_edge(v1, v2)
        assert len(existing_cycles) == 0, f"edge ({v1},{v2}) already exists in edge_to_cycles"
        # Check that v1 and v2 are in the cycle at cycle_idx. Check, again, that edge (v1, v2) does not exist.
        c = self.cycles[cycle_idx]
        assert v1 in c and v2 in c, f"({v1},{v2}) are not in cycle {c}"
        edge_idx1, edge_idx2 = sorted((c.index(v1), c.index(v2)))
        assert edge_idx2 - edge_idx1 >= 2 and (edge_idx1, edge_idx2) != (0, len(c) - 1), f"{v1},{v2} are already adjacent in cycle {c}"

        # Draw the edge (bisect the cycle)
        self.cycles.append(c[edge_idx1 : edge_idx2+1])
        self.cycles[cycle_idx] = c[:edge_idx1+1] + c[edge_idx2:]
        new_cycle_idx = len(self.cycles) - 1

        # Update edge_to_cycles
        self.edge_to_cycles[(v1, v2)] = (cycle_idx, new_cycle_idx) # Add new edge, store that it faces cycle_idx and the new cycle.
        for edge in zip(self.cycles[-1][:-1], self.cycles[-1][1:]): # For each edge in the new cycle self.cycles[-1]:
            edge = tuple(sorted(edge))
            c_list = list(self.edge_to_cycles[edge])
            c_list[c_list.index(cycle_idx)] = new_cycle_idx # Replace cycle_idx with new_cycle_idx
            self.edge_to_cycles[edge] = tuple(c_list)

        # TODO: if necessary, when running draw_new_edge, check that the cycle length equals the length of the cycle as a set.
        # TODO: if necessary, we may want to check to make sure that the new edge to draw isn't already colinear with the points in the cycle c.

    def to_Graph(self):
        adj = sparse.lil_matrix((len(self.V), len(self.V)), dtype=bool)
        for cycle in self.cycles:
            adj[cycle, np.roll(cycle, -1)] = True
        adj = sparse.triu(adj.T + adj)
        return nx.from_scipy_sparse_array(adj)
