from unnet.utils import _get_filter_min_
import graph_tool.all as gt
import graph_tool.topology as topology


class ParamsSetter:
    # pylint: disable=no-member
    def parameters_set(self, params):
        assert len(params) == len(self.parameters_mapping)
        for key, value in params.items():
            if key in self.parameters_mapping:
                setattr(self, key, value)
            else:
                raise KeyError(f"The key {key} is not valid for " + str(self.__class__))

class NoSampler(ParamsSetter):
    """This class is a dummy class and does not apply any sampling"""
    def execute(self, G):
        return G

    def parameters_set(self, params):
        pass

    def parameters_mapping(self, params):
        {}

class IdentitySampler:
    """This class is a dummy class and does not apply any sampling"""
    def execute(self, G):
        yield ({}, G)


def identity(x):
    return x

class AdditionalParamWrapper:
    def __init__(self, cent, func, label):
        self.cent = cent
        self.func = func
        self.label = label
        self.value = None

    @property
    def parameters_mapping(self):
        return {self.label : self.label} + self.cent.parameters_mapping

    def execute(self, graph, *args, **kwargs):
        self.func(self.value, self.cent)
        return self.cent.execute(graph, *args, **kwargs)
    
    def parameters_set(self, params):
        new_params = {key : value for key, value in params.items() if key != self.label}
        self.cent.parameters_set(new_params)
        self.value = params[self.label]

class ChainSamplers:
    def __init__(self, samplers, names=None):
        self.samplers = samplers
        if names is not None:
            assert len(samplers) == len(names)
        self.current_sampler = None
        self.name = None
        self.names = names

    @property
    def parameters_mapping(self):

            return self.current_sampler.parameters_mapping

    def parameters_set(self, params):
        self.current_sampler.parameters_set(params)

    def execute(self, graph, *args, **kwargs):
        for i, sampler in enumerate(self.samplers):
            self.current_sampler = sampler
            if self.names is not None:
                self.name = self.names[i]
            for x, y in sampler.execute(graph, *args, **kwargs):
                if self.name is not None:
                    yield {**x, 'sampler_name' : self.name}, y
                else:
                    yield x
        
from functools import partial       
def change_function_params(function, value, cent):
    cent.function = partial(function, value)

import numpy as np
import numpy.random as rand
import graph_tool.all as gt
import random as rnd
import graph_tool.centrality as cent
import queue as q

#returns the neighbourhood of a node as a list of indices
def get_neighhbours(graph, index):
    return [int(v) for v in graph.vertex(index).out_neighbours()]
    #sur = []
    #for v in graph.vertex(index).out_neighbours():
        #sur.append(int(v))
    #return sur

#input: target graph and the amount of nodes to sample
def random_node_sampling(graph, desired_fraction):
    size = int(desired_fraction * graph.num_vertices())
    n = graph.num_vertices()
    arr = np.random.choice(np.arange(n), size=size, replace=False)
    v_filter = np.zeros(n, dtype=bool)
    v_filter[arr] = True
    return gt.GraphView(graph, vfilt = v_filter)

class NodeSampler(ParamsSetter):
    def __init__(self):
        self.desired_fraction = 0.0

    @property
    def parameters_mapping(self):
        return {
            'desired_fraction' : 'desired_fraction',}

    def execute(self,graph,*args, **kwargs):
        return random_node_sampling(graph, self.desired_fraction)


#input: target graph and the amount of edges to sample
def random_edge_sampling(graph, desired_fraction):
    size = int(desired_fraction * graph.num_edges())
    n = graph.num_edges()
    arr = np.random.choice(np.arange(n), size=size, replace=False)
    e_filter = np.zeros(n, dtype=bool)
    e_filter[arr] = True
    return gt.GraphView(graph, efilt = e_filter)


class EdgeSampler(ParamsSetter):
    def __init__(self):
        self.desired_fraction = 0.0

    @property
    def parameters_mapping(self):
        return {
            'desired_fraction' : 'desired_fraction',}

    def execute(self,graph,*args, **kwargs):
        return random_edge_sampling(graph, self.desired_fraction)

#input: target graph, desired fraction of nodes to, starting node(if not given, chosen randomly), desired fraction of nodes to keep after each iteration
#enable print commands to see chosen node and neighbours in each iteration
def random_walk_sampling(graph, desired_fraction, start=None):
    filter = np.zeros(graph.num_vertices(), dtype=bool)

    #if start not given pick randomly
    if start:
        cur_node = start
    else:
        cur_node = rand.choice(graph.get_vertices(),1)

    max = int(desired_fraction*graph.num_vertices())

    #counts iterations
    i = 1


    filter[cur_node] = 1
    neighbours = []
    for v in graph.vertex(cur_node).out_neighbours():
        if filter[int(v)] != 1:
            neighbours.append(int(v))


    while i <= max and len(neighbours) != 0:
        cur_node = rand.choice(neighbours,1)

        filter[cur_node] = 1
        neighbours.clear()
        for v in graph.vertex(cur_node).out_neighbours():
            if filter[int(v)] != 1:
                neighbours.append(int(v))
    return gt.GraphView(graph, vfilt = filter)


class BiasedEdgeSampler(ParamsSetter):
    """ Retains edges with probability specified in retain matrix

    E.g. a value 0.3 in position i,j in that matrix means, that 70% of edges from group i to j are dropped
    """
    def __init__(self):
        self.retain_matrix = np.zeros((0,0))

    @property
    def parameters_mapping(self):
        return {
            'retain_matrix' : 'retain_matrix',}

    def execute(self,graph,*args, **kwargs):
        return biased_edge_sampling(graph, self.retain_matrix)


#input: target graph and the amount of edges to sample
def biased_edge_sampling(graph, retain_matrix):
    edges = graph.get_edges()

    minorities = _get_filter_min_(graph)
    left_group=minorities[edges[:,0]].astype(np.uint8)
    right_group=minorities[edges[:,1]].astype(np.uint8)
    
    values = retain_matrix[left_group, right_group]

    e_filter = np.random.rand(len(values)) < values

    return gt.GraphView(graph, efilt = e_filter)




class SimilarityEdgeSampler(ParamsSetter):
    """
     sim_type can be any of the graph tool sym_types see https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.vertex_similarity
     function can be any function that maps the similarity values onto RETAIN probabilities, default identity
        use the function to apply a offset, scaling etc.
    """
    def __init__(self):
        self.sim_type = "jaccard"
        self.function = identity
        self.include_self = True

    @property
    def parameters_mapping(self):
        return {'sim_type' : 'sim_type'}

    def execute(self,graph,*args, **kwargs):
        G = similarity_sampling(graph, self.sim_type, self.function, include_self=True)    
        return G


#input: target graph
def similarity_sampling(graph, sim_type, function, include_self=False):
    similarities = get_similarities(graph, sim_type, include_self=include_self)
    
    values = function(similarities)
    #plt.hist(values[values>0], bins=20)
    #print("sum", np.sum(values>0))
    #plt.show()
    #print("frac of zero overlap ", np.sum(values==0)/len(values))
    e_filter = np.random.rand(len(values)) < values
    

    return gt.GraphView(graph, efilt = e_filter)



def get_similarities(graph_in, sim_type, include_self=False):
    if include_self:
        # create a copy and add self edges so that they are part of the neighborhood
        graph = gt.Graph(graph_in)
        arr=np.vstack([graph.get_vertices(),graph.get_vertices()]).T
        graph.add_edge_list(arr)
        edges = graph_in.get_edges()
        #edges = edges[np.logical_not(edges[:,0]==edges[:,1])]
        assert len(edges) == graph_in.num_edges(), f" {len(edges)}, {graph_in.num_edges()}"
    else:
        edges = graph_in.get_edges()
        graph = graph_in

    similarities = topology.vertex_similarity(graph, sim_type=sim_type, vertex_pairs=edges)
    return similarities



class SimilarityEdgeSampler2(ParamsSetter):
    """
     sim_type can be any of the graph tool sym_types see https://graph-tool.skewed.de/static/doc/topology.html#graph_tool.topology.vertex_similarity
     function can be any function that maps the similarity values onto RETAIN probabilities, default identity
        use the function to apply a offset, scaling etc.
    """
    def __init__(self):
        self.sim_type = "jaccard"
        self.function = identity
        self.retain_factor = 0.0
        self.include_self = True

    @property
    def parameters_mapping(self):
        return {'sim_type' : 'sim_type', 'retain_factor':'retain_factor'}

    def execute(self,graph,*args, **kwargs):
        G = similarity_sampling2(graph, self.sim_type, self.function, self.retain_factor, include_self=self.include_self)    
        print("sample_rate", G.num_edges()/graph.num_edges())
        return G



#input: target graph
def similarity_sampling2(graph, sim_type, function, retain_factor, include_self=False):
    similarities = get_similarities(graph, sim_type, include_self=include_self)
    
    values = function(similarities)
    #print("frac of zero overlap ", np.sum(values==0)/len(values))
    e_filter = keep_top_by_retain(values, retain_factor)
    
    return gt.GraphView(graph, efilt = e_filter)

def keep_top_by_retain(values, retain_factor):
    """ Returns a boolean array that attemps to keep the retain_factor top nodes"""
    
    idx = int((1 - retain_factor) * len(values))
    if idx==0:
        return np.ones(len(values),dtype=bool)
    sorted_values = np.sort(values)
    t = sorted_values[idx]
    e_filter = values >= t
    if np.sum(e_filter) == 0:
        return np.ones(len(values),dtype=bool)
    return e_filter

class CentralityCache():
    """Can be used to cache a centrality for a graph, useful for e.g. StructureEdgeSampler"""
    def __init__(self, centrality, cache_len=1):
        self.cache = []
        self.centrality = centrality
        self.cache_len=cache_len

    def get_values(self, graph):
        for (g, values) in self.cache:
            if g == graph:
                return values

        values = self.centrality.get_values(graph)
        if len(self.cache) == self.cache_len:
            self.cache.pop(0)
        self.cache.append((graph, values))
        return values

from functools import partial
class StructureEdgeSampler(ParamsSetter):
    """

    """
    def __init__(self, function):
        self.centrality = None
        self.function = function
        self.show_hist = False
        self.alpha = 0

    @property
    def parameters_mapping(self):
        return {'centrality' : 'centrality', 'alpha' : 'alpha'}

    def execute(self, graph,*args, **kwargs):
        G = structure_sampling(graph, self.centrality, partial(self.function, self.alpha), show_hist=self.show_hist)    
        return G


class StructureEdgeSampler2(ParamsSetter):
    """
        Discards edges based on ranking
    """
    def __init__(self, function=None):
        if function is None:
            function = log_agg_function
        self.centrality = None
        self.function = function
        self.show_hist = False
        self.retain_factor = 0

    @property
    def parameters_mapping(self):
        return {'centrality' : 'centrality', 'retain_factor' : 'retain_factor'}

    def execute(self, graph,*args, **kwargs):
        G = structure_sampling2(graph, self.centrality, self.function, retain_factor=self.retain_factor)    
        return G

def structure_sampling_values(graph, centrality, function):
    """ helper function to compute the centrality values for an edge"""
    cent = centrality.get_values(graph)
    edges = graph.get_edges()
    left_values =cent[edges[:,0]]
    right_values=cent[edges[:,1]]
    values = function(left_values, right_values)
    return values

def structure_sampling(graph, centrality, function, show_hist=False):
    """ Returns a graph view for the graph control the number of edges through the function
    centrality should have a function .get_values that returns the values for the centrality
    function should be a function that takes two vectors where elements at the same position correspond to the same edge
         also make sure function correctly scales values into [0,1] which correspond to survive chances
    retain factor is used to specify the desired amount of """
    values = structure_sampling_values(graph, centrality, function)
    if show_hist:
        import matplotlib.pyplot as plt
        plt.hist(values)
    e_filter = np.random.rand(len(values)) < values

    return gt.GraphView(graph, efilt = e_filter)

def structure_sampling2(graph, centrality, function, retain_factor):
    """ Returns a graph view for the graph with close to retain_factor edges
    centrality should have a function .get_values that returns the values for the centrality
    function should be a function that takes two vectors where elements at the same position correspond to the same edge
    retain factor is used to specify the desired amount of """
    values = structure_sampling_values(graph, centrality, function)
    order = np.argsort(values)
    
    e_filter = order > int((1 -retain_factor) * len(values))
    
    return gt.GraphView(graph, efilt = e_filter)



def log_agg_function(cent1, cent2):
    """aggregates two centrality measures using logarithms and min_max_scaling
    
    """
    #print(np.min(cent1*cent2))
    c1 = np.log(cent1)
    c1[c1<=0] = 0
    c2 = np.log(cent2)
    c2[c2<=0] = 0
    v=c1+c2
    v=(v - v.min()) / (v.max() - v.min())
    return v

def log_agg_function_alpha(alpha, cent1, cent2):
    """ Allows additional scaling of log_agg_function with a parameter alpha
    to roughly get retain factors for BA use
    retain factor 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  0.7, 0.8, 0.9
    alpha         1.7, 1.0, 0.7, 0.5, 0.4, 0.26, 0.2, 0.1, 0.015 """
    return log_agg_function(cent1, cent2) ** alpha




class RandomWalkSampler(ParamsSetter):
    def __init__(self):
        self.desired_fraction = 0.0

    @property
    def parameters_mapping(self):
        return {
            'desired_fraction' : 'desired_fraction',}

    def execute(self,graph,*args, **kwargs):
        return random_walk_sampling(graph, self.desired_fraction)

#input target graph, set if starting nodes, desired fractio of nodes, desired fraction of nodes to keep after each iteration
def snowball_sampling(graph, desired_fraction, retain_factor, start_set):

    filter = np.zeros(graph.num_vertices(), dtype=bool)
    max_vertices = int(graph.num_vertices() * desired_fraction)
    neighbourhood = np.zeros(0, dtype=int)
     #get neighbourhood of starting selected
    target_amount = 0
    #print("start set", [int(v) for v in start_set])
    total = 0
    while (total < max_vertices) and (len(start_set)>0):
        #print("cur_set", [int(v) for v in start_set])
        for v1 in start_set:
            if not filter[int(v1)]:
                filter[int(v1)] = 1
                total+=1
                new_neib = graph.get_out_neighbours(v1)
                #print("A", new_neib)
                new_neib = new_neib[~filter[new_neib]]
                #print("B", new_neib)
                neighbourhood = np.union1d(neighbourhood, new_neib)
                #print("C", new_neib)

        #print("neighbourhood", [int(v) for v in neighbourhood])
        if total == max_vertices or len(neighbourhood) == 0:
            break
        target_amount = min(int(len(neighbourhood) * retain_factor), max_vertices-total)
        #print("n", len(neighbourhood), target_amount, max_vertices-total)
        #print("target_amount", target_amount)
        start_set = np.random.choice(neighbourhood, target_amount, replace = False)
        neighbourhood = np.zeros(0, dtype=int)

    return gt.GraphView(graph, vfilt = filter)


class SnowballSampler(ParamsSetter):
    def __init__(self, start_func):
        self.desired_fraction = 0.0
        self.retain_factor = 1.0
        self.start_func = start_func

    @property
    def parameters_mapping(self):
        return {
            'desired_fraction' : 'desired_fraction',
            'retain_factor' : 'retain_factor',}

    def execute(self, graph,*args, **kwargs):
        return snowball_sampling(graph, self.desired_fraction, self.retain_factor, self.start_func(graph))



class EdgeAdditionFromDeletion:
    """ Adds edges to a graph where a model is used as "prior" From than models graph edges are deleted

    The default model is 'blockmodel-micro' in which we maintain 1) edges between groups and 2) the individual degree
    """
    def __init__(self, deleter, model='blockmodel-micro'):
        self.deleter = deleter
        self.model = model
        self.remove_parallel_edges = True

    @property
    def parameters_mapping(self):
        return self.deleter.parameters_mapping

    def execute(self,graph,*args, **kwargs):

        g = edge_addition_from_deletion(graph, self.deleter, model = self.model, remove_parallel_edges=self.remove_parallel_edges)

        return g
    
    def parameters_set(self, params):
        self.deleter.parameters_set(params)

    def __setattr__(self, name, value):
        if name in ["deleter", "model", "remove_parallel_edges"]:
            return super().__setattr__(name, value)
        else:
            return setattr(self.deleter, name, value)


def edge_addition_from_deletion(graph, deleter, model='blockmodel-micro', remove_parallel_edges=True):
    # create randomly rewired copy of input graph
    g_rewire=graph.copy()
    gt.random_rewire(g_rewire, model=model, block_membership = g_rewire.vp.minority)
    
    # delete edges from that graph
    g_sampled = deleter.execute(g_rewire)
    
    # Create new Graph that has edges from both the rewired and initial
    g_out = graph.copy()
    g_out.add_edge_list(g_sampled.get_edges())
    
    if remove_parallel_edges:
        gt.remove_parallel_edges(g_out)
    
    return g_out