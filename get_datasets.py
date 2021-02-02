import subprocess

dblp_link = r"https://raw.githubusercontent.com/frbkrm/NtwPerceptionBias/master/datasets/DBLP_graph.gexf"
github_link = r"https://raw.githubusercontent.com/frbkrm/NtwPerceptionBias/master/datasets/github_mutual_follower_ntw.gexf"
aps_link = r"https://raw.githubusercontent.com/frbkrm/NtwPerceptionBias/master/datasets/sampled_APS_pacs052030.gexf"


import pathlib
print(pathlib.Path(__file__).parent.absolute())


parent = pathlib.Path(__file__).parent.absolute()

links = [dblp_link, github_link, aps_link]
names = ["dblp", "github", "aps"]
out_names = [str(parent/'notebooks'/name)+".gexf" for name in names]
print("downloading datasets")
print()
if True:
    for link, out_name in zip(links, out_names):
        command = "wget " + link +" -O " + out_name
        subprocess.call(command, shell=True)

print()
print("downloading complete")
print()

#import unnet
#from unnet.GEXFConverter.py import *

#out_name = out_names[0]
#with open(out_names, 'r'):


