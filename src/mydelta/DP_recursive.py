from enum import Enum, IntEnum
import numpy as np
from .Tree import NodeType

BtType = IntEnum('BacktrackType',[ 'NONE', 'Prune_1_L', 'Prune_1_R', 'Prune_2_L', 'Prune_2_R', 'Match_L_L', 'Match_L_R', 'USED', 'NOBT'])

class DynProg(object):
    def __init__(self,prune_cost,cost):
        self.prune_cost = prune_cost
        self.cost = cost
        self.cost_dict = {}
        self.bt_dict = {}
        self.bt_types = (BtType.Match_L_L,BtType.Match_L_R,BtType.Prune_1_R,BtType.Prune_1_L,BtType.Prune_2_R,BtType.Prune_2_L)
        self.bt_types = (BtType.Prune_2_R,BtType.Prune_2_L,BtType.Prune_1_R,BtType.Prune_1_L,BtType.Match_L_L,BtType.Match_L_R)
    def comp_nodes(self,node_a,node_b):
        if node_a is None or node_b is None:
            return 0
        if node_a.node_type == NodeType.terminal and node_b.node_type == NodeType.terminal:
            return self.comp_terminal_terminal(node_a,node_b)
        if node_a.node_type == NodeType.terminal and node_b.node_type != NodeType.terminal:
            return self.comp_terminal_internal(node_a,node_b,lr=(BtType.Prune_2_L,BtType.Prune_2_R),sl=slice(None,None,1))
        if node_a.node_type != NodeType.terminal and node_b.node_type == NodeType.terminal:
            return self.comp_terminal_internal(node_b,node_a,lr=(BtType.Prune_1_L,BtType.Prune_1_R),sl=slice(None,None,-1))
        return self.comp_internal_internal(node_a,node_b)
    def comp_terminal_terminal(self,node_a,node_b):
        key = (node_a.name,node_b.name)
        if key in self.cost_dict:
            return self.cost_dict[key]
        cost1 = self.cost.get((node_a.cell_class,node_b.cell_class),0)
        self.cost_dict[key] = cost1
        self.bt_dict[key] = BtType.NONE
        return cost1
    def comp_terminal_internal(self,node_terminal,node_internal,lr,sl):
        key = (node_terminal.name,node_internal.name)[sl]
        if key in self.cost_dict:
            return self.cost_dict[key]
        cost1 = cost2 = -1e10
        if node_internal.child_L is not None:
            cost1 = self.comp_nodes(*(node_terminal,node_internal.child_L)[sl]) - self.prune_cost*node_internal.child_R.leaves
        if node_internal.child_R is not None:
            cost2 = self.comp_nodes(*(node_terminal,node_internal.child_R)[sl]) - self.prune_cost*node_internal.child_L.leaves
        cost_final = np.maximum(cost1,cost2)
        self.cost_dict[key] = cost_final
        self.bt_dict[key] = lr[0] if cost1 <= cost2 else lr[1]
        return cost_final
    def comp_internal_internal(self,node_a,node_b):
        key = (node_a.name,node_b.name)
        if key in self.cost_dict:
            return self.cost_dict[key]
        a_l,a_r = node_a.child_L,node_a.child_R
        b_l,b_r = node_b.child_L,node_b.child_R
        cost_ll = self.comp_nodes(a_l,b_l) + self.comp_nodes(a_r,b_r)
        cost_lr = self.comp_nodes(a_l,b_r) + self.comp_nodes(a_r,b_l)
        cost_pl = self.comp_nodes(a_l,node_b) - self.prune_cost*a_r.leaves
        cost_pr = self.comp_nodes(a_r,node_b) - self.prune_cost*a_l.leaves
        cost_lp = self.comp_nodes(node_a,b_l) - self.prune_cost*b_r.leaves
        cost_rp = self.comp_nodes(node_a,b_r) - self.prune_cost*b_l.leaves
        costs = np.array([cost_ll,cost_lr,cost_pl,cost_pr,cost_lp,cost_rp])
        costs = np.array([cost_lp,cost_rp,cost_pl,cost_pr,cost_ll,cost_lr])
        max_ind = costs.argmax()
        max_cost = costs[max_ind]
        self.cost_dict[key] = max_cost
        self.bt_dict[key] = self.bt_types[max_ind]
        return max_cost

class Backtrack(object):
    def __init__(self,tree1,tree2,bt_dict):
        self.match = [(tree1,tree2)]
        self.prune_s = []
        self.prune_t = []
        self.bt_dict = bt_dict
    def backtrack(self,tree1,tree2):
        bt = self.bt_dict[(tree1.name,tree2.name)]
        if bt == BtType.NONE:
            return
        if bt == BtType.Match_L_L:
            self.match.append((tree1.child_L,tree2.child_L))
            self.match.append((tree1.child_R,tree2.child_R))
            self.backtrack(tree1.child_L,tree2.child_L)
            self.backtrack(tree1.child_R,tree2.child_R)
        elif bt == BtType.Match_L_R:
            self.match.append((tree1.child_L,tree2.child_R))
            self.match.append((tree1.child_R,tree2.child_L))
            self.backtrack(tree1.child_L,tree2.child_R)
            self.backtrack(tree1.child_R,tree2.child_L)
        elif bt == BtType.Prune_1_R:
            self.prune_s.append(tree1.child_R)
            self.backtrack(tree1.child_L,tree2)
        elif bt == BtType.Prune_1_L:
            self.prune_s.append(tree1.child_L)
            self.backtrack(tree1.child_R,tree2)
        elif bt == BtType.Prune_2_R:
            self.prune_s.append(tree2.child_R)
            self.backtrack(tree1,tree2.child_L)
        elif bt == BtType.Prune_2_L:
            self.prune_s.append(tree2.child_L)
            self.backtrack(tree1,tree2.child_R)
