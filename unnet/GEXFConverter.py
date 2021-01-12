#Original Code: by selcuk guvel https://github.com/selcukguvel/gexf-to-csv
#Edited for un-net by berk kayirhan

# Converts the specified GEXF file to nodes.csv and edges.csv files.
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NODES_FILE = "nodes.csv"
EDGES_FILE = "edges.csv"

def findElement(graphElement,s):
    # Finds either nodesElement or edgesElement which str specifies to.
    element = None
    for child in graphElement:
        if s in child.tag:
            element = child
    if element == None:
        raise Exception('Please specify a valid GEXF file which contains "nodes" and "edges" tags.')
    return element

def writeNodes(nodesElement, name=None):
    # Writes to the nodes.csv file which has two columns: nodeID and nodeName.
    global NODES_FILE
    if name is not None:
        NODES_FILE = name+"_nodes.csv"
    with open(NODES_FILE,'w') as file:
        file.write("nodeID,group\n")
        def gen(nodesElement):
            for node in nodesElement:                              # node   = <node id= .. > .. </node>
                nodeAtr  = node.attrib
                nodeId   = nodeAtr["id"]
                for attvalues in node:                             # attvalues  = <attvalues> .. </attvalues>
                    for attvalue in attvalues:                     # attvalue   = <attvalue>  .. </attvalue>
                        nodeAttr = attvalue.attrib
                        group = nodeAttr["value"]
                yield nodeId + "," + group
        file.write("\n".join(gen(nodesElement)))

def writeEdges(edgesElement, name=None):
    # Writes to the edges.csv file which has three columns: SourceID, TargetID and weight.
    global EDGES_FILE
    if name is not None:
        EDGES_FILE = name+"_edges.csv"

    with open(EDGES_FILE,'w') as file:
        file.write("SourceID,TargetID\n")
        for edge in edgesElement:                              # edge   = <edge source= .. > .. </edge>
            edgeAtr = edge.attrib
            sourceId = edgeAtr["source"]
            targetId = edgeAtr["target"]
            file.write(sourceId + "," + targetId + "\n")

def convert(gexfFile, name=None):
    # Converts GEXF to the CSV.
    #if 'gexf' not in gexfFile:
    #    raise Exception('Please specify a valid GEXF file.')
    tree = ET.parse(gexfFile)
    root = tree.getroot()
    for child in root:
        if 'graph' in child.tag:
            graphElement = child
    nodesElement = findElement(graphElement,'nodes')           # nodesElement = <nodes> .. </nodes>
    edgesElement = findElement(graphElement,'edges')           # edgesElement = <edges> .. </edges>
    writeNodes(nodesElement, name)
    writeEdges(edgesElement, name)




    print(NODES_FILE + " and " + EDGES_FILE + " are created.")
    print(Path(NODES_FILE).absolute())
    print(Path(EDGES_FILE).absolute())
    print("Number of nodes: {}".format(len(nodesElement)))
    print("Number of edges: {}".format(len(edgesElement)))
