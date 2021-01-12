import csv
import numpy as np
import graph_tool.all as gt
from unnet.GEXFConverter import convert
from pathlib import Path

def __to_csv_nodes_hashed__(input):
    nodes = {}
    data = csv.reader(input, delimiter=',')
    for skip in range(1):
            next(data)
    for i,row in enumerate(data):
        if row[0] not in nodes:
            nodes[row[0]] = (i,row[1])
    return nodes

def __to_csv_edges_hashed__(input, nodes):
    edge_list = []
    data = csv.reader(input, delimiter=',')
    for skip in range(1):
            next(data)
    for row in data:
        edge_list.append((nodes[row[0]][0], nodes[row[1]][0]))
    return edge_list

def __to_csv_nodes__(input):
    nodes = {}
    data = csv.reader(input, delimiter=',')
    for skip in range(1):
            next(data)
    for row in data:
        nodes[int(row[0])] = int(row[1])
    return nodes

def __to_csv_edges__(input):
    edge_list = []
    data = csv.reader(input, delimiter=',')
    for skip in range(1):
            next(data)
    for row in data:
        edge_list.append((int(row[0]), int(row[1])))
    return edge_list

def __to_csv_edges_bt__(input):
    edge_list = []
    data = csv.reader(input, delimiter=',')
    for skip in range(1):
            next(data)
    for row in data:
        if int(row[2]) > 0: #check if data is a user
            edge_list.append((int(row[1]), int(row[2])))
        else:
            continue
    return edge_list



#brazil dataset must be under the name brazil.csv in local directory
def load_brazil_dataset():
    edge_list = set()
    with open('brazil.csv') as brazil:
        data = csv.reader(brazil, delimiter=';')
        #skip comments
        for skip in range(24):
            next(data)
        for row in data:
            edge_list.add((int(row[0]), int(row[1])))
        females = {int(x[0]) for x in edge_list}
        males   = {int(x[1]) for x in edge_list}

        g = gt.Graph(directed = False)
        g.add_edge_list(edge_list)
        v_min = g.new_vertex_property("int")
        g.vertex_properties["minority"] = v_min
        for v in females:
            g.vp.minority[v] = 1
        for v in males:
            g.vp.minority[v] = 0

        return g

def load_pok_dataset():

    nodes = {}
    edge_list = []
    if not Path('pok_nodes.csv').is_file():
        with open('pok.gexf') as pok:
            convert(pok, "pok")

    with open('pok_nodes.csv') as pok:
        nodes = __to_csv_nodes__(pok)

    with open('pok_edges.csv') as pok:
        edge_list = __to_csv_edges__(pok)

    g = gt.Graph(directed = False)
    v_min = g.new_vertex_property("int")
    g.add_edge_list(edge_list)
    g.vertex_properties["minority"] = v_min

    for key, value in nodes.items():
        if value == 2:
            g.vp.minority[key] = 1
        else:
            g.vp.minority[key] = 0
    return g


def load_aps_dataset():
    edge_list = []
    nodes = {}
    if not Path('aps_nodes.csv').is_file():
        with open('aps.gexf') as aps:
            convert(aps, "aps")

    with open('aps_nodes.csv') as aps:
        nodes = __to_csv_nodes_hashed__(aps)

    with open('aps_edges.csv') as aps:
        edge_list = __to_csv_edges_hashed__(aps, nodes)

    g = gt.Graph(directed = False)
    v_min = g.new_vertex_property("int")
    g.vertex_properties["minority"] = v_min
    g.add_edge_list(edge_list)
    for key, value in nodes.items():
        if value[1][-1] == "d":
            g.vp.minority[value[0]] = 0
        else:
            g.vp.minority[value[0]] = 1
    return g

def load_dblp_dataset():
    edge_list = []
    nodes = {}

    if not Path('dblp_nodes.csv').is_file():
        with open('dblp.gexf') as dblp:
            convert(dblp, "dblp")

    with open('dblp_nodes.csv') as dblp:
        nodes = __to_csv_nodes_hashed__(dblp)

    with open('dblp_edges.csv') as dblp:
        edge_list = __to_csv_edges_hashed__(dblp, nodes)

    g = gt.Graph(directed = False)
    v_min = g.new_vertex_property("int")
    g.vertex_properties["minority"] = v_min
    g.add_edge_list(edge_list)
    for key, value in nodes.items():
        if value[1] == "f":
            g.vp.minority[value[0]] = 1
        else:
            g.vp.minority[value[0]] = 0
    return g

def load_github_dataset():
    edge_list = []
    nodes = {}
    if not Path('github_nodes.csv').is_file():
        with open('github.gexf') as git:
            convert(git, "github")

    with open('github_nodes.csv') as git:
        nodes = __to_csv_nodes_hashed__(git)

    with open('github_edges.csv') as git:
        edge_list = __to_csv_edges_hashed__(git, nodes)
    arr = np.array(edge_list)
    #print(arr)
    #print(arr.shape)
    arr=arr[arr[:,1]<arr[:,0],:]
    #print(arr.shape)

    g = gt.Graph(directed = False)
    v_min = g.new_vertex_property("int")
    g.vertex_properties["minority"] = v_min
    g.add_edge_list(arr)
    no_gender = []
    for (node_id, value) in nodes.values():
        assert isinstance(node_id, int)
        if value == "female":
            g.vp.minority[node_id] = 1
        elif value == "male":
            g.vp.minority[node_id] = 0
        elif value == "None":
            no_gender.append(node_id)
        else:
            raise ValueError(value)
    for v in reversed(sorted(no_gender)):
        g.remove_vertex(v)
    return g

def load_copenhagen_dataset(edge_type):
    edge_list = []
    nodes = {}

    if not Path('copenhagen_nodes.csv').is_file():
        with open('copenhagen.gexf') as copenhagen:
            convert(copenhagen, "copenhagen")

    with open('copenhagen_nodes.csv') as copenhagen:
        nodes = __to_csv_nodes__(copenhagen)

    if edge_type == "facebook":
        with open('facebook_edges.csv') as edges:
            edge_list = __to_csv_edges__(edges)
    elif edge_type == "sms":
        with open('sms_edges.csv') as edges:
            edge_list = __to_csv_edges__(edges)
    elif edge_type == "calls":
        with open('calls_edges.csv') as edges:
            edge_list = __to_csv_edges__(edges)
    elif edge_type == "bt":
        with open('bt_edges.csv') as edges:
            edge_list = __to_csv_edges_bt__(edges)
    g = gt.Graph(directed = False)
    v_min = g.new_vertex_property("int")
    vertex_id = g.new_vertex_property("int")
    g.vertex_properties["minority"] = v_min
    g.vertex_properties["id"] = vertex_id
    g.add_edge_list(edge_list)
    gt.remove_parallel_edges(g)
    no_gender=[]
    for v in g.get_vertices():
        g.vp.id[v] = int(v)
        if int(v) not in nodes.keys():
            no_gender.append(int(v))
            continue
        elif nodes[int(v)] == 1:
            g.vp.minority[v] = 1
        elif nodes[int(v)] == 0:
            g.vp.minority[v] = 0
        else:
            raise ValueError(v)
    for v in reversed(sorted(no_gender)):
        g.remove_vertex(v)
    return g

