import graph_tool.all as gt
import numpy as np
from unnet.BarabasiGenerator import LinLogHomophilySampler, HomophilySamplerMulti, HomophilySamplerRejection

def barabassi_albert_network(n, m):
    return gt.price_network(n, m, directed=False)

def barabassi_albert_network_with_random_labels(n, m, minority_fraction=None, rng=None):
    """ Creates a barabasi albert graph and afterwards assigns random node labels
    """
    G = barabassi_albert_network(n,m)
    minority_size = int(minority_fraction * n)
        
    labels = np.zeros(n, dtype=bool)
    labels[:minority_size]=1
    if rng is None:
        np.random.shuffle(labels)
    else:
        raise NotImplementedError

    G.vertex_properties["minority"] = G.new_vertex_property('bool', vals=labels)
    return G


def homophily_barabasi_network(n, m , minority_fraction, homophily, epsilon, force_minority_first=False, multi_edges="none"):
    """ Creates a homophilic + preferential attachment graph
    """
    G=gt.Graph(directed=False)
    if multi_edges=="none":
        edges, labels = LinLogHomophilySampler(n, m , minority_fraction, homophily, epsilon=epsilon, force_minority_first=force_minority_first)
    elif multi_edges=="allow":
        edges, labels = HomophilySamplerMulti(n, m , minority_fraction, homophily, epsilon=epsilon, force_minority_first=force_minority_first)
    elif multi_edges=="rejection":
        edges, labels = HomophilySamplerRejection(n, m , minority_fraction, homophily, epsilon=epsilon, force_minority_first=force_minority_first)
    else:
        raise ValueError("multi_edges must be one of [none, allow, rejection]")
    G.add_edge_list(edges)
    G.vertex_properties["minority"]=G.new_vertex_property('bool', vals=labels)
    return G



class ParamsSetter:
    # pylint: disable=no-member
    def parameters_set(self, params):
        assert len(params) == len(self.parameters_mapping)
        for key, value in params.items():
            if key in self.parameters_mapping:
                setattr(self, key, value)
            else:
                raise KeyError(f"The key {key} is not valid for "+str(self.__class__))
    # pylint: enable=no-member



class BarabasiGenerator(ParamsSetter):
    def __init__(self):
        self.node_size = 0
        self.node_growth = 0
    
    def generate(self):
        pass
    
    @property
    def parameters_mapping(self):
        return {'node_size' : 'n',
         'node_growth' : 'm'}
    
    def execute(self,*args, **kwargs):
        return barabassi_albert_network_with_random_labels(self.node_size, 
        self.node_growth,0,0)


class HomophilyGenerator(ParamsSetter):
    """
    Generator class for homophilic random graph using BA preferential attachment model.

    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which 
    depends on the connectivity of sites and the homophily(similarities).
    homophily varies ranges from 0 to 1.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    minority_fraction : float
        fraction of instances in the minority group
    homophily: float
        value between 0 and 1. similarity between nodes
    multi_edges: str
        How to treat multi edges?
            "none" : use sampling without replacement to avoid multi edges WARNING: produces a slightly different distribution
            "rejection" : use rejection sampling to try and avoid multi edges (Fariba's approach)
            "allow" : allow multi edges to have a perfect BA model

    epsilon: float
        constant value that any node gets an edge #TODO

    Returns
    -------
    G : graph-tool.Graph
        undirected graph, the vertex property 'minority' indicates (1 belongs to minority, 0 does not) 

    Notes
    -----
        The initialization is a graph with with m nodes and no edges.

    """
    def __init__(self, multi_edges="none"):
        super().__init__()
        self.minority_fraction = 0.5
        self.homophily = 0.5
        self.n = 0
        self.m = 0
        self.epsilon = 10E-10
        self.multi_edges = multi_edges

    @property
    def parameters_mapping(self):
        return {
            'n' : 'n',
            'm' : 'm',
            'minority_fraction':'minority_fraction',
            'homophily' : 'homophily'}

    def execute(self,*args, **kwargs):
        return homophily_barabasi_network(n = self.n,
                                m = self.m,
                                minority_fraction = self.minority_fraction,
                                homophily = self.homophily,
                                epsilon=self.epsilon,
                                multi_edges=self.multi_edges)
