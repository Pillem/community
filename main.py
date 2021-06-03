#import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.cluster import average_clustering
from networkx.generators.community import LFR_benchmark_graph
import subprocess
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import networkx.algorithms.community as nxcom
from matplotlib import pyplot as plt
from numpy.core.shape_base import block

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1

def get_nodes_community_list(G):
    '''Get list orderd by nodes with the community label as value'''
    comms = []
    for v in G.nodes:
        # Add 1 to save 0 for external edges
        comms.append(G.nodes[v]['community'])
    return comms

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
    # print("Assigning colours to communities")    
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
    # print("Assigning colours complete")
    return node_color_array

def draw(g, comms, layout):
    node_colours = nodecolours(comms)
    # print("Build image of graph")  
    if(layout == 'spec'):
        pos =  nx.spectral_layout(g)
    else:
        pos =  nx.spring_layout(g)
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    plt.style.use('dark_background')

    # Set node and edge communities
    set_node_community(g, comms)
    set_edge_community(g)

    # Set community color for internal edges
    external = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in g.edges if g.edges[v, w]['community'] > 0]
    internal_color = ["black" for e in internal]
    node_colours = [get_color(g.nodes[v]['community']) for v in g.nodes]
    
    # # external edges
    # nx.draw_networkx(
    #     g, 
    #     pos=pos, 
    #     node_size=0, 
    #     edgelist=external, 
    #     edge_color="silver",
    #     node_color=node_colours,
    #     alpha=0.15, 
    #     with_labels=False)
    # # internal edges
    # nx.draw_networkx(
    #     g, 
    #     pos=pos, 
    #     edgelist=internal, 
    #     edge_color=internal_color,
    #     node_color=node_colours,
    #     alpha=0.09, 
    #     with_labels=False)

    # print("Build image of graph complete")  

    # plt.show(block=False)

def readCommunitiesFromFile(fileName):
    sets = []
    with open(fileName) as f:
        content = f.read().splitlines()
        for line in content:
            node, comm = line.split()
            if(int(comm) >= len(sets)):
                a = {node}
                sets.append(a)
            else:
                sets[int(comm)].add(node)
    return sets

def getLFR(args):
    # print("Start LFR generation")
    arguments = " -N " + str(args['N']) + " -k "     + str(args['k']) + " -maxk "  + str(args['maxk']) + " -mu "    + str(args['mu']) + " -t1 "    + str(args['t1']) + " -t2 "    + str(args['t2']) + " -minc "  + str(args['minc']) + " -maxc "  + str(args['maxc'])
    try:
        subprocess.run(["./benchmark" + arguments] , shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    g = nx.read_edgelist('./network.dat')
    # print("Finnished LFR generation")
    return g

def getCommunitiesLFR(g):
    return sorted(readCommunitiesFromFile('./community.dat'), key=len, reverse=True)

def getCommunitiesGN(g):
    # print("Start Girvan Newman for communities")
    communities_generator = nxcom.girvan_newman(g)
    comms = sorted(next(communities_generator), key=len, reverse=True)
    # print("Finnished Girvan Newman")
    return comms

def getCommunitiesCLAUSET(g):
    # print("Start Clauset greedy modularity optimisation for communities")
    comms = sorted(nxcom.greedy_modularity_communities(g), key=len, reverse=True)
    # print("Finnished Clauset greedy modularity optimisation for communities")
    return comms

def runSinglesMuIncrease(lfrArgs):
    for i in range(10):
        g = getLFR(lfrArgs)
        g_GN = g.copy()
        g_CLAUSET = g.copy()
        
        top_level_communities = getCommunitiesLFR(g)
        top_level_communities_GN = getCommunitiesGN(g_GN)
        top_level_communities_CLAUSET = getCommunitiesCLAUSET(g_CLAUSET)
        
        # print("Generated " + str(len(top_level_communities)) + " communities in the graph")
        # print("GN Found " + str(len(top_level_communities_GN)) + " communities in the graph")
        # print("CLAUSET Found " + str(len(top_level_communities_CLAUSET)) + " communities in the graph")
        
        draw(g, top_level_communities, 'spec')
        plt.figure(2)
        draw(g_GN, top_level_communities_GN, 'spec')
        plt.figure(3)
        draw(g_CLAUSET, top_level_communities_CLAUSET, 'spec')
        print("----------------------------------------------------------------------------")
        print("mu = " + str(lfrArgs['mu']))
        print("Normalized mutual information GN: " + str(nmi(get_nodes_community_list(g), get_nodes_community_list(g_GN))))
        print("Normalized mutual information CLAUSET: " + str(nmi(get_nodes_community_list(g), get_nodes_community_list(g_CLAUSET))))
        print("----------------------------------------------------------------------------")
        lfrArgs['mu'] += 0.1

def runCentAverage(args):
    args['nmi_GN'] =0
    args['nmi_CLAUSET'] =0
    for i in range(100):
        g = getLFR(args)
        g_GN = g.copy()
        g_CLAUSET = g.copy()
        
        top_level_communities = getCommunitiesLFR(g)
        # if(args['N'] < 5000):
            # top_level_communities_GN = getCommunitiesGN(g_GN)
        top_level_communities_CLAUSET = getCommunitiesCLAUSET(g_CLAUSET)
        
        # print("Generated " + str(len(top_level_communities)) + " communities in the graph")
        # print("GN Found " + str(len(top_level_communities_GN)) + " communities in the graph")
        # print("CLAUSET Found " + str(len(top_level_communities_CLAUSET)) + " communities in the graph")
        
        draw(g, top_level_communities, 'spec')
        # plt.figure(2)
        # draw(g_GN, top_level_communities_GN, 'spec')
        plt.figure(3)
        draw(g_CLAUSET, top_level_communities_CLAUSET, 'spec')

        # if(args['N'] < 5000):
        #     args['nmi_GN'] += nmi(get_nodes_community_list(g), get_nodes_community_list(g_GN))

        args['nmi_CLAUSET'] += nmi(get_nodes_community_list(g), get_nodes_community_list(g_CLAUSET))
    
    args['nmi_GN'] /= 100
    args['nmi_CLAUSET'] /= 100
    return args





# LFR graph generation parameters
lfrArgs = {
    'N':     200  , # variable 1000 or 5000
    'k':     20   ,   # static
    'maxk':  50   ,   # static
    'mu':    0.2  ,    # For 0 to 1 in steps
    't1':  2      ,   # static
    't2':  1      ,   # static

    'minc':  10   ,   # variable [10 and 50] or [20 and 100]
    'maxc':  50,
}


# runSinglesMuIncrease(lfrArgs)
resultStats = runCentAverage(lfrArgs)

print("----------------------------------------------------------------------------")
print("mu = " + str(lfrArgs['mu']))
print("Normalized mutual information GN: " + str(resultStats['nmi_GN']))
print("Normalized mutual information CLAUSET: " + str(resultStats['nmi_CLAUSET']))
print("----------------------------------------------------------------------------")