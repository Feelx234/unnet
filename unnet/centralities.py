import numpy as np
import numpy.random as rand
import graph_tool.all as gt
import random as rnd
import graph_tool.centrality as cent
import graph_tool.search as search
import queue as q
from unnet.utils import project_with_node_view, _get_filter_min_



def _iter_to_list_(it):
    return [(int(i.source()), int(i.target())) for i in it]

def _ecc_get_dist_(source, target, list):
    dist = 0
    l = list
    while target != source:
        for v1 in l:
            if v1[1] == target:
                target = v1[0]
                dist = dist+1
    return dist


def node_eccentricity(Graph, Node):
    it = _iter_to_list_(search.bfs_iterator(Graph,Node))
    l = [(i[1],_ecc_get_dist_(Node,i[1],it)) for i in it]
    return max([j for i,j in l])

def graph_eccentricity(G):
    return [node_eccentricity(G,v) for v in G.get_vertices()]


def average_degree(graph):
    arr = graph.get_out_degrees(graph.get_vertices())/(graph.num_vertices()-1)
    return np.mean(arr), np.std(arr)

def color_aware_degree(graph):
    red_centrality = []
    blue_centrality = []
    for v in graph.vertices():
        if graph.vp.color[v] == 1:
            red_centrality.append(v)
        else:
            blue_centrality.append(v)
    red_centrality = graph.get_out_degrees(red_centrality)/(graph.num_vertices()-1)
    blue_centrality = graph.get_out_degrees(blue_centrality)/(graph.num_vertices()-1)

    return np.mean(blue_centrality), np.mean(red_centrality)



def mean_std_by_class(values, graph):
    min_filter =  _get_filter_min_(graph)
    mi = values[min_filter]
    ma = values[~ min_filter]
    return {
        'min_mean' : np.mean(mi),
        'min_std' : np.std(ma),
        'maj_mean' : np.mean(mi),
        'maj_std' : np.std(ma)}


def top_k_by_class(values, graph, k):
    if isinstance(values, gt.VertexPropertyMap):
        values = values.get_array()
    min_filter =  _get_filter_min_(graph)
    order = np.argsort(values)[::-1] # sort descendingly
    order = order[:k] # grab top k
    top_k = min_filter[order]
    min_k = np.sum(top_k)
    min_total = np.sum(min_filter)
    return {
        f'min_count_{k}' : min_k,
        f'maj_count_{k}' : k - min_k,
        f'min_total_{k}' : min_total,
        f'maj_total_{k}' : len(min_filter) - min_total,
        }

def top_cumsum_by_class(values, graph):
    if isinstance(values, gt.VertexPropertyMap):
        values = values.get_array()
    min_filter =  _get_filter_min_(graph)
    order = np.argsort(values)[::-1] # sort descendingly
    values_sorted = values[order]
    sorted_filter = min_filter[order]
    return {
        'min_cumsum' : np.cumsum(values_sorted * sorted_filter),
        'maj_cumsum' : np.cumsum(values_sorted * (~sorted_filter)),
        }

def top_places_by_class(values, graph):
    if isinstance(values, gt.VertexPropertyMap):
        values = values.get_array()
    min_filter =  _get_filter_min_(graph)
    order = np.argsort(values)[::-1] # sort descendingly
    sorted_filter = min_filter[order]
    return {
        'min_places' : np.cumsum(sorted_filter),
        'maj_places' : np.cumsum(~sorted_filter),
        }


def distr_by_class(values, graph, minlength):
    min_filter=_get_filter_min_(graph)

    min_centrality = values[min_filter]
    maj_centrality = values[~min_filter]
    if issubclass(values.dtype.type, np.floating):
        raise NotImplementedError("distr by class is not implemented for floating point values")
        #return {'distr_maj' : np.histogram(maj_centrality.astype(np.int), minlength=minlength)[0],
        #        'distr_min' : np.histogram(min_centrality.astype(np.int), minlength=minlength)[0]}
    else:
        return {'distr_maj' : np.bincount(maj_centrality.astype(np.int), minlength=minlength),
                'distr_min' : np.bincount(min_centrality.astype(np.int), minlength=minlength)}

def distr_cumsum_by_class(values, graph):
    min_filter=_get_filter_min_(graph)

    min_centrality = values[min_filter]
    maj_centrality = values[~min_filter]

    min_centrality.sort()
    maj_centrality.sort()

    return {'distr_cumsum_maj' : np.arange(len(maj_centrality), 0, -1)/len(maj_centrality),
            'distr_cumsum_maj_x' : maj_centrality,
            'distr_cumsum_min' : np.arange(len(min_centrality), 0, -1)/len(min_centrality),
            'distr_cumsum_min_x' : min_centrality}



class AbstractCentralityMeasure:
    def __init__(self, mode, **kwargs):
        assert len(mode)>0
        if isinstance(mode, str):
            mode=[mode]

        for val in mode:
            l=["average", "top_k", "distr", "top_cumsum", "top_places", "distr_cumsum", "raw", ""]
            if not val in l:
                raise ValueError("mode should be in " + str(l))

        self.mode = mode
        if "top_k" in mode:
            self.k = kwargs['k']
        if "minlength" in kwargs:
            self.minlength = kwargs["minlength"]

    def execute(self, graph):
        values = self.get_values(graph)
        results={}
        if "average" in self.mode:
            results = {**results, **self.prepend(mean_std_by_class(values, graph))}
        if "top_k" in self.mode:
            results = {**results, **self.prepend(top_k_by_class(values, graph, self.k))}
        if "distr" in self.mode:
            results = {**results, **self.prepend(distr_by_class(values, graph, getattr(self, 'minlength', 0)))}
        if "distr_cumsum" in self.mode:
            results = {**results, **self.prepend(distr_cumsum_by_class(values, graph))}
        if "top_cumsum" in self.mode:
            results = {**results, **self.prepend(top_cumsum_by_class(values, graph))}
        if "top_places" in self.mode:
            results = {**results, **self.prepend(top_places_by_class(values, graph))}
        if "raw" in self.mode:
            results = {**results , **self.prepend({'raw' : values}) }
        return results

    def prepend(self, d):
        prefix = self.name
        return {prefix+"_"+key:value for key, value in d.items()}

class EdgeCountByClass(AbstractCentralityMeasure):
    name="edge"
    def get_values(self, graph):
        m_filter = ~graph.vp.minority.get_array().astype(bool)
        #m_filter = ~_get_filter_min_(graph)

        edges = graph.get_edges()
        left_values =m_filter[edges[:,0]]
        right_values=m_filter[edges[:,1]]
        return np.bincount(left_values + 2 * right_values, minlength=3)

class MinorityFraction():
    def execute(self, graph):
        minority_fraction = _get_filter_min_(graph).sum() / graph.num_vertices()
        return {'minority_measured' : minority_fraction}

class NumEdges():
    def execute(self, graph):
        return {'num_edges' : graph.num_edges()}


class EccentricityByClass(AbstractCentralityMeasure):
    name = 'eccentricity'
    def get_values(self, graph):
        eccentricity = graph_eccentricity(graph)
        return project_with_node_view(eccentricity.get_array(), graph)

class BetweennessByClass(AbstractCentralityMeasure):
    name = 'betweenness'
    def get_values(self, graph):
        betweeness, _ = cent.betweenness(graph)
        return project_with_node_view(betweeness.get_array(), graph)

class PagerankByClass(AbstractCentralityMeasure):
    name = 'pagerank'
    def get_values(self, graph):
        return project_with_node_view(cent.pagerank(graph).get_array(), graph)

import time
class EigenvectorByClass(AbstractCentralityMeasure):
    def __init__(self, mode, **kwargs):
        self.epsilon = kwargs.get('epsilon', 1e-4)
        self.max_iter = kwargs.get('max_iter', 1e-4)

        super().__init__(mode, **kwargs)
    name = 'eigenvector'
    def get_values(self, graph):
        print("eigenvec")
        time.sleep(0.1)
        _, vector = cent.eigenvector(graph, epsilon=1e-4, max_iter=100)
        return project_with_node_view(vector.get_array(), graph)

class MinorityLabels():
    def execute(self, graph):
        minority_labels = _get_filter_min_(graph)
        return {'minority_labels' : minority_labels}

class DegreeByClass(AbstractCentralityMeasure):
    name = 'degree'
    def get_values(self, graph):
        return graph.get_out_degrees(graph.get_vertices())

class KatzByClass(AbstractCentralityMeasure):
    name = "katz"
    def get_values(self, graph):
        return project_with_node_view(cent.katz(graph).get_array(), graph)

class JoinMeasures():
    def __init__(self, measures):
        self.measures = measures

    def execute(self, graph):
        results={}
        for measure in self.measures:
            new_result = measure.execute(graph)
            results = {**results, **new_result}
        return results

class DegreeDistributionByClass(AbstractCentralityMeasure):

    def execute(self, graph):
        degrees=graph.get_out_degrees(np.arange(graph.num_vertices()))

        indices=graph.vp.minority.get_array().astype(bool)

        min_centrality = degrees[indices]
        maj_centrality = degrees[~indices]
        print(self.minlength)
        return {'degree_0' : np.bincount(maj_centrality.astype(np.int),minlength=self.minlength),
                'degree_1' : np.bincount(min_centrality.astype(np.int),minlength=self.minlength)}
