"""smp_graphs.experiment.py - sensorimotor experiments as computation graphs (smp)

2017 Oswald Berthold

Experiment: basic experiment shell for
 - running a graph
 - loading and drawing a graph (networkx)
"""

import argparse, os, re, sys
import time

from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# for config reading
from numpy import array

from smp_base.plot import set_interactive, makefig

from smp_graphs.block import Block2
from smp_graphs.utils import print_dict
from smp_graphs.common import conf_header, conf_footer
from smp_graphs.common import get_config_raw
from smp_graphs.graph import nxgraph_plot, recursive_draw, nxgraph_flatten, nxgraph_add_edges

################################################################################
# utils, TODO: move to utils.py
def get_args():
    """Experiment.py.get_args

    Define argparse commandline arguments
    """
    # define defaults
    default_conf     = "conf/default.py"
    default_numsteps = None # 10
    # create parser
    parser = argparse.ArgumentParser()
    # add required arguments
    parser.add_argument("-c", "--conf",     type=str, default=default_conf,     help="Configuration file [%s]" % default_conf)
    parser.add_argument("-dr", "--do-ros",  dest="ros", action="store_true",    default = None, help = "Do / enable ROS?")
    parser.add_argument("-nr", "--no-ros",  dest="ros", action="store_false",   default = None, help = "No / disable ROS?")
    parser.add_argument("-m", "--mode",     type=str, default="run",            help="Which subprogram to run [run], one of [run, graphviz]")
    parser.add_argument("-n", "--numsteps", type=int, default=default_numsteps, help="Number of outer loop steps [%s]" % default_numsteps)
    parser.add_argument("-s", "--randseed",     type=int, default=None,             help="Random seed [None], seed is taken from config file")
    parser.add_argument("-pg", "--plotgraph", dest="plotgraph", action="store_true", default = False, help = "Plot smp graph")
    # parser.add_argument("-sp", "--saveplot", type=int, default=None,             help="Random seed [None], seed is taken from config file")
    # 
    # parse arguments
    args = parser.parse_args()
    # return arguments
    return args

def set_config_defaults(conf):
    """Experiment.py.set_config_defaults

    Set configuration defaults if they are missing
    """
    if not conf['params'].has_key("numsteps"):
        conf['params']['numsteps'] = 100
    return conf

def set_config_commandline_args(conf, args):
    """Experiment.py.set_config_commandline_args

    Set configuration params from commandline, used to override config file setting for quick tests
    """
    # for commandline_arg in conf['params'].has_key("numsteps"):
    #     conf['params']['numsteps'] = 100
    gparams = ['numsteps', 'randseed', 'ros']
    for clarg in gparams:
        if getattr(args, clarg) is not None:
            conf['params'][clarg] = getattr(args, clarg)
    return conf

def make_expr_id_configfile(name = "experiment", configfile = "conf/default2.py"):
    """Experiment.py.make_expr_id_configfile

    Make experiment signature from name and timestamp
    """
    # split configuration path
    confs = configfile.split("/")
    # get last element config filename
    confs = confs[-1].split(".")[0]
    # format and return
    return "%s_%s_%s" % (name, confs, make_expr_sig())

def make_expr_id(name = "experiment"):
    """Experiment.py.make_expr_id

    Dummy callback
    """
    pass

def make_expr_sig(args =  None):
    """Experiment.py.make_expr_sig

    Return formatted timestamp
    """
    return time.strftime("%Y%m%d_%H%M%S")

def md5(obj):
    import hashlib
    # print "self.conf", str(self.conf)
    # if type(obj) is not str:
    #     obj = str(obj)
    m = hashlib.md5(obj)
    return m

def make_expr_md5(obj):
    return md5(str(obj))

def set_random_seed(args):
    """set_random_seed
    
    Extract randseed parameter from args.conf, override with args.randseed if set and seed the numpy prng.

    Arguments:
    - args: argparse Namespace

    Returns:
    - randseed: the seed
    """
    assert hasattr(args, 'conf')
    randseed = 0
    
    conf = get_config_raw(args.conf, confvar = 'conf', fexec = False)

    pattern = re.compile('(randseed *= *[0-9]*)')
    # pattern = re.compile('.*(randseed).*')
    # print "pattern", pattern
    m = pattern.search(conf)
    # print "m[:] = %s" % (m.groups(), )
    # print "m[0] = %s" % (m.group(0), )
    # print "lv = %s" % (lv, )
    # m = re.search(r'(.*)(randseed *= *[0-9]*)', conf)
    # conf_ = re.sub(r'\n', r' ', conf)
    # conf_ = re.sub(r' +', r' ', conf_)
    # print "m", conf_
    # m = re.search(r'.*(randseed).*', conf)
    # print "m", m.group(1)
    if m is not None:
        code = compile(m.group(0), "<string>", "exec")
        gv = {}
        lv = {}
        exec(code, gv, lv)
        randseed = lv['randseed']
        # print "args.conf randseed match %s" % (randseed, )

    if hasattr(args, 'randseed') and args.randseed is not None:
        randseed = args.randseed

    # print "m", m
    # print "conf", conf
    # print "randseed", randseed
        
    np.random.seed(randseed)
    return randseed

class Experiment(object):
    """Experiment class

    Arguments:
    - args: argparse configuration namespace (key, value)

    Load a config from the file given in args.conf, initialize nxgraph from conf, run the graph
    """

    # global config file parameters
    gparams = ['ros', 'numsteps', 'recurrent', 'debug', 'dim', 'dt', 'showplot', 'saveplot', 'randseed']
    
    def __init__(self, args):
        """Experiment.__init__

        Experiment init

        Arguments:
        - args: argparse configuration namespace (key, value) containing args.conf
        """
        # get global func pointer
        global make_expr_id
        # point at other func, global make_expr_id is used in common (FIXME please)
        make_expr_id = partial(make_expr_id_configfile, name = 'smpx', configfile = args.conf)

        # set random seed _before_ compiling conf
        set_random_seed(args)

        # get configuration from file # , this implicitly sets the id via global make_expr_id which is crap
        self.conf = get_config_raw(args.conf)
        # print "conf.params.id", self.conf['params']['id']
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)
        # fill in missing defaults
        self.conf = set_config_defaults(self.conf)
        # update conf from commandline arguments
        self.conf = set_config_commandline_args(self.conf, args)
        # initialize ROS if needed
        if self.conf['params']['ros']:
            import rospy
            rospy.init_node("smp_graph")

        # store all conf entries in self
        # print "%s-%s.init\n" % (self.__class__.__name__, None),
        for k in self.conf.keys():
            setattr(self, k, self.conf[k])
            # selfattr = getattr(self, k)
            # if type(selfattr) is dict:
            #     print "        self.%s = %s\n" % (k, print_dict(selfattr))
            # else:
            #     print "        self.%s = %s\n" % (k, selfattr)
        """
        Hash functions
        - hashlib md5/sha
        - locally sensitive hashes, lshash. this is not independent of input size, would need maxsize kludge
        """

        # update experiments database with the current expr
        m = self.update_experiments_store()
            
        # store md5 in params _after_ we computed the md5 hash
        self.conf['params']['md5'] = m.hexdigest()

        # instantiate topblock
        self.topblock = Block2(conf = self.conf)

        # plotting
        self.plotgraph_flag = args.plotgraph

    def update_experiments_store(self):
        """Experiment.update_experiments_store

        Update the global store of experiments with the current one.

        The idea is to take a hash of the configuration and store the
        experiment's results with its hash as a key. If the experiment
        is rerun with the same config, only the logfile is loaded
        instead of recomputing everything.

        Storage options:
         1. a dict with pickle, fail
         2. tinydb, fail
         3. storage: hdf5 via pandas dataframe, works, current
         4. maybe upgrade to nosql / distributed a la mongodb, couchdb, or current favorite elasticsearch
        """

        f = open('conf-%s.txt' % (time.strftime('%H%M%S')), 'wa')
        f.write(str(self.conf))
        f.flush()
    
        # hash the experiment
        m = make_expr_md5(self.conf)
        self.conf['params']['id'] = make_expr_id() + "-" + m.hexdigest()

        experiments_store = 'data/experiments_store.h5'
        columns = ['md5', 'block', 'params']
        values = [[m.hexdigest(), str(self.conf['block']), str(self.conf['params'])]]
        print "%s.update_experiments_store values = %s" % (self.__class__.__name__, values)
        # values = [[m.hexdigest(), self.conf['block'], self.conf['params']]]
    
        if os.path.exists(experiments_store):
            try:
                self.experiments = pd.read_hdf(experiments_store, key = 'experiments')
                print "Experiment.update_experiments_store loaded experiments_store = %s with shape = %s" % (experiments_store, self.experiments)
                # search for hash
            except Exception, e:
                print "Loading store %s failed with %s" % (experiments_store, e)
                sys.exit(1)
        else:
            # store doesn't exist, create an empty one
            self.experiments = pd.DataFrame(columns = columns)
        self.cache = self.experiments[:][self.experiments['md5'] == m.hexdigest()]
        print "Experiment.update_experiments_store found cached results = %s" % (self.cache, )

        # temp dataframe
        df = pd.DataFrame(values, columns = columns, index = [self.experiments.shape[0]])

        dfs = [self.experiments, df]

        print "dfs", dfs
        
        # concatenated onto main df
        self.experiments = pd.concat(dfs)

        # write store 
        self.experiments.to_hdf('data/experiments_store.h5', key = 'experiments')

        # return the hash
        return m
        
    def plotgraph(self):
        """Experiment.plotgraph

        Show a visualization of the graph of the experiment
        """
        graph_fig = makefig(
            rows = 1, cols = 3, wspace = 0.1, hspace = 0.0,
            axesspec = [(0, 0), (0, slice(1, None))], title = "Nxgraph and Bus")
        
        # nxgraph_plot(self.topblock.nxgraph, ax = graph_fig.axes[0])
        # flatten for drawing, quick hack
        G = nxgraph_flatten(self.topblock.nxgraph)
        # for node,noded in G.nodes_iter(data=True):
        #     print "node", node, G.node[node], noded
        G = nxgraph_add_edges(G)
        # for edge in G.edges_iter():
        #     print "edge", edge
        nxgraph_plot(G, ax = graph_fig.axes[0], layout_type = "spring", node_size = 300)
        # recursive_draw(self.topblock.nxgraph, ax = graph_fig.axes[0], node_size = 300, currentscalefactor = 0.1)
        self.topblock.bus.plot(graph_fig.axes[1])
        if self.conf['params']['saveplot']:
            filename = "data/%s_%s.%s" % (self.topblock.id, "graph_bus", 'jpg')
            graph_fig.savefig(filename, dpi=300, bbox_inches="tight")
        # print self.conf['params']
            
    def run(self):
        """Experiment.run

        Run the experiment by running the graph.
        """
        print '#' * 80
        print "Init done, running %s" % (self.topblock.nxgraph.name, )
        print "    Graph: %s" % (self.topblock.nxgraph.nodes(), )
        print "      Bus: %s" % (self.topblock.bus.keys(),)
        print " numsteps: {0}/{1}".format(self.params['numsteps'], self.topblock.numsteps)

        # TODO: try run
        #       except go interactive
        # import pdb
        # topblock_x = self.topblock.step(x = None)
        for i in xrange(self.params['numsteps']):
            # try:
            topblock_x = self.topblock.step(x = None)
            # except:
            # pdb.set_trace()
            # FIXME: progress bar / display        
            
        print "final return value topblock.x = %s" % (topblock_x)

        # # write store
        # self.experiments.to_hdf('data/experiments_store.h5', 'data')
        
        # plot the computation graph and the bus
        set_interactive(True)
        if self.plotgraph_flag:
            self.plotgraph()

        if self.conf['params']['showplot']:
            set_interactive(False)
            plt.show()

import networkx as nx
import re

class Graphviz(object):
    """Graphviz class

    Load a runtime config into a networkx graph and plot it
    """
    def __init__(self, args):
        """Graphviz.__init__

        Initialize a Graphviz instance

        Arguments:
        - args: argparse configuration namespace (key, value)
        """
        # load graph config
        self.conf = get_config_raw(args.conf)
        assert self.conf is not None, "%s.init: Couldn't read config file %s" % (self.__class__.__name__, args.conf)

        # set the layout
        self.layouts = ["spring", "shell", "pygraphviz", "random"]
        self.layout  = self.layouts[2]

    def run(self):
        """Graphviz.run

        Run method
        """
        # create nx graph
        G = nx.MultiDiGraph()

        # FIXME: make the node and edge finding stuff into recursive functions
        #        to accomodate nesting and loops at arbitrary levels
        
        # pass 1: add the nodes
        for k, v in self.conf['params']['graph'].items():
            print "k", k, "v", v
            blockname = re.sub(r"<smp_graphs.block.*\.(.*) object.*", "\\1", v['block'])
            G.add_node(k, block = blockname)
            if v['params'].has_key('graph'): # hierarchical block containing subgraph
                for subk, subv in v['params']['graph'].items():
                    # print "sub", subk, subv
                    blockname = re.sub(r"<smp_graphs.block.*\.(.*) object.*", "\\1", subv['block'])
                    G.add_node(subk, block = blockname)
            elif v['params'].has_key('loopblock') and v['params'].has_key('blocksize'):
                if len(v['params']['loopblock']) < 1: continue
                # for subk, subv in v['params']['loopblock'].items():
                # print "sub", subk, subv
                # print k, print_dict(v['params'])
                lblock = v['params']['loopblock']
                # print "lblock", lblock, v['params']['blocksize']
                blockname = re.sub(r"<class 'smp_graphs.block.*\.(.*)'>", "\\1", lblock['block'])
                # print "block.id", lblock['params']['id']
                for i in range(v['params']['blocksize']):
                    k_from = lblock['params']['id'] + "/%d" % (i,)
                    G.add_node(k_from, block = blockname)
                    G.add_edge(k, k_from)
                    
            # print "k", k
            # print "v", v
            
        # pass 2: add the edges
        for k, v in self.conf['params']['graph'].items():
            # print "v['params']", v['params']
            # loop edges
            if v['params'].has_key('loopblock') and len(v['params']['loopblock']) == 0:

                # print "G", G[k]
                k_from = k.split("_")[0]
                G.add_edge(k_from, k)
            
            # input edges
            if not v['params'].has_key('inputs'): continue
            for inputkey, inputval in v['params']['inputs'].items():
                print "ink", inputkey
                print "inv", inputval
                if not inputval.has_key('bus'): continue
                # get the buskey for that input
                if inputval['bus'] not in ['None']:
                    k_from, v_to = inputval['bus'].split('/')
                    G.add_edge(k_from, k)

        # FIXME: add _loop_ and _containment_ edges with different color
        # print print_dict(pdict = self.conf[7:])

        # pass 3: create the layout

        nxgraph_plot(G, layout_type = self.layout)
        
        # layout = nxgraph_get_layout(G, self.layout)
                    
        # print G.nodes(data = True)
        # labels = {'%s' % node[0]: '%s' % node[1]['block'] for node in G.nodes(data = True)}
        # print "labels = %s" % labels
        # # nx.draw(G)
        # # nx.draw_networkx_labels(G)
        # # nx.draw_networkx(G, pos = layout, node_color = 'g', node_shape = '8')
        # nx.draw_networkx_nodes(G, pos = layout, node_color = 'g', node_shape = '8')
        # nx.draw_networkx_labels(G, pos = layout, labels = labels, font_color = 'r', font_size = 8, )
        # # print G.nodes()
        # e1 = [] # std edges
        # e2 = [] # loop edges
        # for edge in G.edges():
        #     # print edge
        #     if re.search("[_/]", edge[1]):
        #         e2.append(edge)
        #     else:
        #         e1.append(edge)

        # nx.draw_networkx_edges(G, pos = layout, edgelist = e1, edge_color = "g", width = 2)
        # nx.draw_networkx_edges(G, pos = layout, edgelist = e2, edge_color = "k")
        plt.show()
