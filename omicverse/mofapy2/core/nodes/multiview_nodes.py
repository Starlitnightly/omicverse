"""
Module to define multi-view nodes.
A multi-view node is simply a node that is defined for several views. For example, the weights W or the data Y, but not the latent variables Z.

Types of multi-view nodes:
- Multiview_Variational_Node: variational nodes present in all views
- Multiview_Constant_Node: multiview nodes that are just constant values
- Multiview_Mixed_Node: a node which is variational for some views (i.e. gaussian views) and constant for others (i.e. non-gaussian views)

All multiview nodes have the following main attributes:
- M: total number of views
- activeM: in some occasions a particular node is active in only a subset of views. For example, we could activate spike-and-slab in one view but not in the other.
- nodes: a list with the (single-view) nodes
"""

import scipy as s
import numpy as np

from .basic_nodes import Node
from .variational_nodes import Variational_Node

#TODO : check add Basic_Multiview_Mixed_Node(Node,Multiview_Constant_Node)

class Multiview_Node(Node):
    """General class for a multiview node"""
    def __init__(self, M, *nodes):
        """
        PARAMETERS
        ----------
        M: int
            total number of views
        nodes: list
            list of M nodes, which must be instances or children of the 'Node' class. If the node is not defined in view m, then index m is set to None.
        """
        self.M = M
        self.activeM = [ m for m, node in enumerate(nodes) if node is not None]
        self.nodes = nodes

    def addMarkovBlanket(self, **kwargs):
        """Method to define the Markov blanket"""
        # assert len(kwargs.values()) == len(self.activeM), "The markov blanket of a multiview node should be a dictionary where the key is the name of the node and the value is a list of nodes of length M"
        for k,v in kwargs.items():
            for m in self.activeM:
                self.nodes[m].addMarkovBlanket( **{ k: (v.getNodes()[m] if isinstance(v,Multiview_Node) else v) } )

        # for k,v in kwargs.items():
        #     for m in self.activeM:
        #         if hasattr(self.nodes[m], 'markov_blanket'):
        #             if k in self.nodes[m].markov_blanket.keys():
        #                 print("Error: " + str(k) + " is already in the markov blanket of " + str(self.nodes[m]))
        #             else:
        #                 if isinstance(v,Multiview_Node):
        #                     self.nodes[m].markov_blanket[k] = v.getNodes()[m]
        #                 else:
        #                     self.nodes[m].markov_blanket[k] = v
        #         else:
        #             self.nodes[m].addMarkovBlanket( **{ k: (v.getNodes()[m] if isinstance(v,Multiview_Node) else v) } )

    def getMarkovBlanket(self):
        print("Error: Multiview nodes do not have a markov blanket, use the single-view nodes")
        exit(1)

    def removeFactors(self, idx):
        """Method to remove factors from the node

        PARAMETERS
        ----------
        idx: ndarray
            indices of the factors to be removed
        """
        for m in self.activeM: self.nodes[m].removeFactors(idx)

    def getNodes(self):
        """Method to get the nodes"""
        return self.nodes

    def getExpectation(self):
        """Method to get the first moments (expectation)"""
        return [ self.nodes[m].getExpectation() for m in self.activeM ]

    def getExpectations(self):
        """Method to get all moments"""
        return [ self.nodes[m].getExpectations() for m in self.activeM ]

    def getParameters(self):
        """Method to get  the parameters"""
        return [ self.nodes[m].getParameters() for m in self.activeM ]

    def updateDim(self, axis, new_dim, m=None):
        """Method to update the dimensionality of the node

        PARAMETERS
        ----------
        axis: int
        new_dim: int
        m: iterable
            views to update
        """
        assert np.all(m in self.activeM), "Trying to update the dimensionality of a node that doesnt exist in a view"
        M = self.activeM if m is None else m
        for m in M: self.nodes[m].updateDim(axis,new_dim)

    def precompute(self, options):
        for m in self.activeM:
            self.nodes[m].precompute(options)

    def define_mini_batch(self, ix):
        for m in self.activeM:
            self.nodes[m].define_mini_batch(ix)

    def get_mini_batch(self):
        return [self.nodes[m].get_mini_batch() for m in self.activeM]


class Multiview_Variational_Node(Multiview_Node, Variational_Node):
    """General class for multiview variational nodes."""
    def __init__(self, M, *nodes):
        Multiview_Node.__init__(self, M, *nodes)
        for node in nodes: assert isinstance(node, Variational_Node)

    def update(self, ix=None, ro=1.):
        """ Method to update both parameters and expectations of the node"""
        for m in self.activeM:
            self.nodes[m].updateParameters(ix, ro)
            self.nodes[m].updateExpectations()

    def updateExpectations(self):
        """Method to update expectations using current estimates of the parameters"""
        for m in self.activeM: self.nodes[m].updateExpectations()
    def updateParameters(self, ix=None, ro=1.):
        """Method to update parameters using current estimates of the expectations"""
        for m in self.activeM: self.nodes[m].updateParameters(ix, ro)
    def calculateELBO(self, weights):
        """Method to calculate variational evidence lower bound"""
        lb = [ self.nodes[m].calculateELBO() * weights[m] for m in self.activeM ]
        return sum(lb)

class Multiview_Constant_Node(Multiview_Node):
    """General class for multiview local nodes"""
    def __init__(self, M, *nodes):
        Multiview_Node.__init__(self, M, *nodes)

    def getValues(self):
        """Method to return the values of the node"""
        return [ self.nodes[m].getValue() for m in self.activeM ]

class Multiview_Mixed_Node(Multiview_Constant_Node, Multiview_Variational_Node):
    """General Class for multiview nodes that contain both variational and constant nodes"""
    def __init__(self, M, *nodes):
        # M: number of views
        # nodes: list of M 'Node' instances
        Multiview_Node.__init__(self, M, *nodes)

    def update(self, ix=None, ro=1.):
        """Method to update values of the nodes"""
        for m in self.activeM: self.nodes[m].update(ix, ro)

    def calculateELBO(self, weights):
        """Method to calculate variational evidence lower bound
        The lower bound of a multiview node is the sum of the lower bound of its corresponding single view variational nodes
        """
        lb = 0
        for m in self.activeM:
            if isinstance(self.nodes[m],Variational_Node):
                lb += self.nodes[m].calculateELBO() * weights[m]
        return lb
