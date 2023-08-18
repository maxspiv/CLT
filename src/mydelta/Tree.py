from enum import Enum, IntEnum
import re
import collections

NodeType = Enum('NodeType',['root','internal','terminal'])

class TN(object):
    def __init__(self,name:str,node_type:NodeType,parent=None,child_L=None,child_R=None):
        self.name = name
        self.node_type = node_type
        self.parent = parent
        self.child_L = child_L
        self.child_R = child_R
    def __str__(self):
        dd = {k:getattr(self,k) for k in dir(self) if not (k.startswith('_') or k.startswith('child_') or k == 'parent')}
        return str(dd)
    def __repr__(self):
        return self.__str__()
    def countLeaves(self):
        if self.node_type == NodeType.terminal:
            self.leaves = 1
            return 1
        else:
            self.leaves = self.child_L.countLeaves() + self.child_R.countLeaves()
            return self.leaves
class CLT(collections.UserDict):
    def __init__(self,tns:dict):
        super().__init__(tns)
        self.countLeaves()
        self.sort()
    def countLeaves(self):
        return self.data['root'].countLeaves()
    def sort(self):
        self.data = dict(sorted(self.data.items(),key = lambda v: v[1].leaves))


def nodes_from_file(fname:str):
    node_str = open(fname).readlines()
    nodes = []
    for node in node_str[1:]:
        nodes.append(node.upper().strip().split('\t'))
    return nodes
def create_CLT(nodes:list,pref=''):
    root = TN('root',NodeType.root)
    tree_nodes = {root.name:root}
    for node in nodes:
        parent = tree_nodes['root']
        node_str = node[0]
        name = pref
        while len(node_str) > 0:
            name += node_str[0]
            if name not in tree_nodes:
                tree_node = TN(name,NodeType.internal,parent=parent)
            else:
                tree_node = tree_nodes[name]
            if name[-1] == '0':
                parent.child_L = tree_node
            else:
                parent.child_R = tree_node
            tree_nodes[name] = tree_node
            node_str = node_str[1:]
            parent = tree_node
        tree_node.node_type = NodeType.terminal
        tree_node.cell_name = node[1]
        tree_node.cell_class = node[2]
    clt = CLT(tree_nodes)
    return clt
def cost_matrix(cost_fname):
    costs_str = open(cost_fname).readlines()
    costs = {}
    for cost_str in costs_str:
        cost = re.split(r'\t|\s',cost_str.upper().strip())
        costs[(cost[0],cost[1])] = float(cost[2])
    return costs
def create_all(nodes1,nodes2,pref1='q',pref2='s',cost_file=None):
    clt1 = create_CLT(nodes1,pref1)
    clt2 = create_CLT(nodes2,pref2)
    if cost_file is not None:
        cost = cost_matrix(cost_file)
    else:
        cost = None
    return clt1,clt2,cost
