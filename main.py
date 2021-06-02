#import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.cluster import average_clustering
from networkx.generators.community import LFR_benchmark_graph
from networkx.algorithms import community

import networkx.algorithms.community as nxcom
from matplotlib import pyplot as plt

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1

def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)            

def nodecolours(communities):
    print("Assigning colours to communities")    
    node_to_community = dict()
    index = 0
    for com in communities:
        for node in com:
            node_to_community[node] = index % 11
        index+=1
    community_to_color = {
        0 : 'tab:purple',
        1 : 'tab:orange',
        2 : 'tab:green',
        3 : 'tab:red',
        4 : 'tab:blue',
        5 : 'tab:brown',
        6 : 'tab:gray',
        7 : 'tab:olive',
        8 : 'tab:cyan',
        9 : 'tab:pink',
        10 : 'y',
    }
    #node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}
    node_color_array = [community_to_color[community_id] for node, community_id in node_to_community.items()]
    print("Assigning colours complete")
    return node_color_array

def draw(g, node_colours):
    print("Build image of graph")  
    pos =  nx.spectral_layout(g)
    #pos =  nx.spring_layout(g)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    plt.style.use('dark_background')

    communities = sorted(nxcom.greedy_modularity_communities(g), key=len, reverse=True)
    print("Found " + str(len(communities))+ " communities")

    # Set node and edge communities
    set_node_community(g, top_level_communities)
    set_edge_community(g)

    # Set community color for internal edges
    external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_colours = [get_color(g.nodes[v]['community']) for v in g.nodes]
    
    # external edges
    nx.draw_networkx(
        g, 
        pos=pos, 
        node_size=0, 
        edgelist=external, 
        edge_color="silver",
        node_color=node_colours,
        alpha=0.1, 
        with_labels=False)
    # internal edges
    nx.draw_networkx(
        g, 
        pos=pos, 
        edgelist=internal, 
        edge_color=internal_color,
        node_color=node_colours,
        alpha=0.05, 
        with_labels=False)


    print("Build image of graph complete")  

    plt.show()

# # create a modular graph    WERKT MAAR TE ZWAAR VOOR GN
n =     5000
k =     20
maxk =  60
mu =    .1#0.1
tau1 =  2.1#2.0
tau2 =  1.1#1.1
minc =  11
maxc =  50

# create a modular graph
# n =     300
# k =     20
# maxk =  160
# mu =    0.2#0.1
# tau1 =  3#2.0
# tau2 =  1.5#1.1
# minc =  90
# maxc =  70

for i in range(9):
    mu += 0.1
    print("Start LFR generation")
    g = LFR_benchmark_graph(n, tau1, tau2, mu, average_degree=k, max_degree=maxk, min_community=minc,  seed=10, max_iters=1000, tol=0.45) 
    print("Finnished LFR generation")
    print(g)
    top_level_communities = {frozenset(g.nodes[v]["community"]) for v in g}

    print("Found " + str(len(top_level_communities)) + " communities in the graph")

    node_color_array=nodecolours(top_level_communities)
    print(node_color_array)


    draw(g, node_color_array);



# print("Start Girvan Newman for communities")
# communities_generator = community.girvan_newman(g)
# top_level_communities_GN = next(communities_generator)
# print("Finnished Girvan Newman")
# print("Found " + str(len(top_level_communities_GN)) + " communities in the graph")
# print("Assigning colours to communities")

# print("Found " + str(len(top_level_communities_GN)) + " communities in the graph")

# node_color_array_GN=nodecolours(top_level_communities_GN)
# print(node_color_array_GN)

# for i in range(0,len(node_color_array)):
#     print(node_color_array[i] == node_color_array_GN[i])


# draw(g, node_color_array_GN);
