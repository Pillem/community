#import matplotlib.pyplot as plt
import networkx as nx
import subprocess
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import networkx.algorithms.community as nxcom
from matplotlib import pyplot as plt
from numpy.core.shape_base import block
from joblib import Parallel, delayed
import copy
import sys


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
        # print(v)
        # print(G.nodes[v])
        if(G.nodes[v] == {}):
            comms.append(0)
        else:
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

def draw(g, comms, layout, showImage = False, singlePos = None):
    node_colours = nodecolours(comms)
    # print("Build image of graph")  
    if(singlePos is not None):
        pos = singlePos
    else:
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
    
    if(showImage):
        # external edges
        nx.draw_networkx(
            g, 
            pos=pos, 
            node_size=0, 
            edgelist=external, 
            edge_color="silver",
            node_color=node_colours,
            alpha=0.15, 
            with_labels=False)
        # internal edges
        nx.draw_networkx(
            g, 
            pos=pos, 
            edgelist=internal, 
            edge_color=internal_color,
            node_color=node_colours,
            alpha=0.09, 
            with_labels=False)
        plt.show(block=False)

    # print("Build image of graph complete")  

    

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
    arguments = " -N " + str(args['N']) + " -k "     + str(args['k']) + " -maxk "  + str(args['maxk']) + " -mu "    + str(args['mu']) + " -t1 "    + str(args['t1']) + " -t2 "    + str(args['t2']) + " -minc "  + str(args['minc']) + " -maxc "  + str(args['maxc']) + " -powerlaw " + str(args['powerlaw'])
    try:
        subprocess.run(["./benchmark" + arguments] , shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    g = nx.read_edgelist('./network.dat')
    c = sorted(readCommunitiesFromFile('./community.dat'), key=len, reverse=True)
    # print("Finnished LFR generation")
    return [g, c]

def getHundredLFR(args):
    return [getLFR(args) for i in range(100)] 

def getCommunitiesLFR(g):
    return sorted(readCommunitiesFromFile('./community.dat'), key=len, reverse=True)

def getCommunitiesGN(g):
    # print("Start Girvan Newman for communities")
    communities_generator = nxcom.girvan_newman(g)
    
    comms = next(communities_generator)
    further = next(communities_generator)
    oldScore = nxcom.modularity(g, comms)
    newScore = nxcom.modularity(g, further)
    while (newScore > oldScore):
        comms = further
        oldScore = newScore
        further = next(communities_generator)
        newScore = nxcom.modularity(g, further)
        # print(oldScore)
    comms = sorted(comms, key=len, reverse=True)
    # print("Finnished Girvan Newman")
    return comms

def getCommunitiesLABEL_PROPAGATION(g):
    # print("Start Clauset greedy modularity optimisation for communities")
    comms = sorted(nxcom.label_propagation_communities(g), key=len, reverse=True)
    # print("Finnished Clauset greedy modularity optimisation for communities")
    return comms

def getCommunitiesCLAUSET(g):
    # print("Start Clauset greedy modularity optimisation for communities")
    comms = sorted(nxcom.greedy_modularity_communities(g), key=len, reverse=True)
    # print("Finnished Clauset greedy modularity optimisation for communities")
    return comms

def runSingleConfigTest(lfrArgs):
    lfrInfo = getLFR(lfrArgs) # Original author code
    g = lfrInfo[0]
    g_LABEL_PROPAGATION = g.copy()
    g_CLAUSET = g.copy()
    
    top_level_communities = lfrInfo[1]#getCommunitiesLFR(g)
    top_level_communities_LABEL_PROPAGATION = getCommunitiesLABEL_PROPAGATION(g_LABEL_PROPAGATION)
    top_level_communities_CLAUSET = getCommunitiesCLAUSET(g_CLAUSET)
    
    # print("Generated " + str(len(top_level_communities)) + " communities in the graph")
    # print("GN Found " + str(len(top_level_communities_GN)) + " communities in the graph")
    # print("CLAUSET Found " + str(len(top_level_communities_CLAUSET)) + " communities in the graph")
    
    pos = nx.spectral_layout(g)
    draw(g, top_level_communities, 'spec', True, singlePos= pos)
    plt.figure(2)
    draw(g_LABEL_PROPAGATION, top_level_communities_LABEL_PROPAGATION, 'spec', True, singlePos= pos)
    plt.figure(3)
    draw(g_CLAUSET, top_level_communities_CLAUSET, 'spec', True, singlePos= pos)
    print("----------------------------------------------------------------------------")
    print("mu = " + str(lfrArgs['mu']))
    print("Normalized mutual information LABEL_PROPAGATION: " + str(nmi(get_nodes_community_list(g), get_nodes_community_list(g_LABEL_PROPAGATION))))
    print("Normalized mutual information CLAUSET: " + str(nmi(get_nodes_community_list(g), get_nodes_community_list(g_CLAUSET))))
    print("----------------------------------------------------------------------------")
    plt.show()
    
def runSingleIteration(args, lfrInfo):
    g = lfrInfo[0]
    g_LABEL_PROPAGATION = g.copy()
    g_CLAUSET = g.copy()
    
    lfrInfo.append(0)         # 'nmi_LABEL_PROPAGATION' result sum values
    lfrInfo.append(0)         # 'nmi_CLAUSET' result sum value

    top_level_communities = lfrInfo[1] #getCommunitiesLFR(g)
    top_level_communities_LABEL_PROPAGATION = getCommunitiesLABEL_PROPAGATION(g_LABEL_PROPAGATION) 
    top_level_communities_CLAUSET = getCommunitiesCLAUSET(g_CLAUSET)
    
    # print("Generated " + str(len(top_level_communities)) + " communities in the graph")
    # print("GN Found " + str(len(top_level_communities_GN)) + " communities in the graph")
    # print("CLAUSET Found " + str(len(top_level_communities_CLAUSET)) + " communities in the graph")
    
    draw(g, top_level_communities, 'spec')
    plt.figure(2)
    draw(g_LABEL_PROPAGATION, top_level_communities_LABEL_PROPAGATION, 'spec')
    plt.figure(3)
    draw(g_CLAUSET, top_level_communities_CLAUSET, 'spec')

    lfrInfo[2] += nmi(get_nodes_community_list(g), get_nodes_community_list(g_LABEL_PROPAGATION))
    lfrInfo[3] += nmi(get_nodes_community_list(g), get_nodes_community_list(g_CLAUSET))

    return lfrInfo

def runCentAverage(args):
    args['nmi_LABEL_PROPAGATION'] =0
    args['nmi_CLAUSET'] =0
    multipleLFR = getHundredLFR(args)

    # Multithreaded operation
    multipleLFR = Parallel(n_jobs=16)(delayed(runSingleIteration)(args, multipleLFR[i]) for i in range(100))
    
    # for i in range(100):
    #     multipleLFR[i] = runSingleIteration(args, multipleLFR[i])

    # Combine 100 values of gn and clauset and divide by 100 to get average
    for i in range(100):
        args['nmi_LABEL_PROPAGATION'] += multipleLFR[i][2]
        args['nmi_CLAUSET'] += multipleLFR[i][3]
    args['nmi_LABEL_PROPAGATION'] /= 100
    args['nmi_CLAUSET'] /= 100
    return args

def runEachMuCentTimes(args, name):
    print()
    print("----------------------------------------------------------------------------")
    print('Configuration: ' + str(name)) 

    currentArgs = copy.deepcopy(args)

    for i in range(5): # Run S SN for each mu
        resultStats = runCentAverage(currentArgs)
        
        
        # print()
        # print("mu = " + str(args['mu']))
        # print("Normalized mutual information LABEL_PROPAGATION: " + str(resultStats['nmi_LABEL_PROPAGATION']))
        # print("Normalized mutual information CLAUSET: " + str(resultStats['nmi_CLAUSET']))
        print(str(currentArgs['mu']) + " , " + 
            str(resultStats['nmi_LABEL_PROPAGATION']) + " , " + 
            str(resultStats['nmi_CLAUSET'])
        )
        sys.stdout.flush()
        currentArgs['mu']+=0.1
    print("----------------------------------------------------------------------------")
    print()

def runAllTestsLFR(argConfigs, configNames):
    print('Measuring Normalized Mutual information of communities found on LFR graph with label propagation and Clouset et al. algorithms')
    print('Testing for 4 LFR configurations for 10 intervals of Mu')
    print('Each test result is averaged over 100 trials')
    print('mu , nmi_LABEL_PROPAGATION , nmi_CLAUSET')
    print()
    for i in range(4):
        runEachMuCentTimes(argConfigs[i], configNames[i])

def runAllTestsBA(argConfigs, configNames):
    argConfigs[0]['powerlaw'] = 0         # _S_S
    argConfigs[1]['powerlaw'] = 0         # _S_B
    argConfigs[2]['powerlaw'] = 0         # _S_B
    argConfigs[3]['powerlaw'] = 0         # _B_S

    print('Measuring Normalized Mutual information of communities found on Barabási-Albert graph with label propagation and Clouset et al. algorithms')
    print('Testing for 4 LFR configurations for 10 intervals of Mu')
    print('Each test result is averaged over 100 trials')
    print('mu , nmi_LABEL_PROPAGATION , nmi_CLAUSET')
    print()
    for i in range(4):
        runEachMuCentTimes(argConfigs[i].copy(), configNames[i])

# LFR graph generation parameters for small communities and small amount of nodes
# naming done with lfrArgs_X_Y where:
#       X = S for small comms (10-50) and B for big comms (20-100)
#       Y = S for small amount of nodes (1000) and B for large amount of nodes (5000)

lfrArgs_S_S = {
    'N':     1000  , # variable 1000 or 5000
    'k':     20   ,   # static
    'maxk':  50   ,   # static
    'mu':    0.1  ,    # For 0 to 1 in steps
    't1':  2      ,   # static
    't2':  1      ,   # static

    'minc':  10   ,   # variable [10 and 50] or [20 and 100]
    'maxc':  50,
    'powerlaw': 1,
}
argConfigs = [lfrArgs_S_S,lfrArgs_S_S.copy(),lfrArgs_S_S.copy(),lfrArgs_S_S.copy()] 

argConfigs[0]['N'] = 1000                                                           # _S_S

argConfigs[1]['N'] = 5000                                                           # _S_B

argConfigs[2]['minc'] = 20                                                          # _B_S
argConfigs[2]['maxc'] = 100                                                          

argConfigs[3]['minc'] = 20                                                          # _B_B
argConfigs[3]['maxc'] = 100                             
argConfigs[3]['N'] = 5000

configNames = ['_S_S (minc=10,maxc=50,N=1000)','_S_B (minc=10,maxc=10,N=5000)','_B_S (minc=20,maxc=100, N=1000)','_B_B (minc=20,maxc=20,N=5000)']



# argConfigs[0]['powerlaw'] = 0
# argConfigs[0]['n'] = 5000
# for i in range(6):
#     argConfigs[0]['mu'] +=.1
#     runSingleConfigTest(argConfigs[0])
# plt.show()

runAllTestsLFR(argConfigs.copy(), configNames)
runAllTestsBA(argConfigs.copy(), configNames)