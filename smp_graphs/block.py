"""smp_graphs - smp sensorimotor experiments as computation graphs

block: basic block of computation

2017 Oswald Berthold

"""

import uuid, sys
from collections import OrderedDict
import pickle

import numpy as np

import smp_graphs.logging as log

BLOCKSIZE_MAX = 10000

# some decorators
class decInit():
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            f(exec_self, *args, **kwargs)
            # write dynamic changes back to config
            if type(kwargs['conf']) == dict: # and kwargs['conf'].has_key('graph'):
                for k,v in kwargs['conf'].items():
                    # print "%s k = %s, v = %s" % (exec_self.__class__.__name__, k, v)
                    kwargs['conf'][k] = exec_self.__dict__[k]
                    # print "xxx", k, v
                # print "%s, conf = %s" % (exec_self.__class__.__name__, kwargs['conf'])
        return wrap
            

class decStep():
    """step decorator"""
    def __call__(self, f):
        def wrap(exec_self, *args, **kwargs):
            # get any input or set to default
            if len(args) > 0:
                x = np.atleast_2d(args[0])
            elif kwargs.has_key('x'):
                x = np.atleast_2d(kwargs['x'])
            else:
                x = None

            # input argument above might be obsolete due to busses
            # get the input from the bus
            if exec_self.idim is not None:
                stack = [exec_self.bus[k] for k in exec_self.inputs]
                # print "stack", stack
                x = np.vstack(stack)

            # print "decStep: x.shape = %s" % (x.shape,)
            
            # write to input buffer
            if x.shape == (exec_self.idim, 1):
                if exec_self.ibufsize > 1:
                    exec_self.bufs["ibuf"] = np.roll(exec_self.bufs["ibuf"], shift = -1, axis = 1)
                exec_self.bufs["ibuf"][:,-1,np.newaxis] = x

            # call the function
            f_out = f(exec_self, x)
            
            # count calls
            exec_self.cnt += 1
            exec_self.ibufidx = exec_self.cnt % exec_self.ibufsize
            
            return f_out
        # return the new func
        return wrap
    
class Block(object):
    """smp_graphs block base class

handles both primitive and composite blocks
 - a primitive block directly implements a computation in it's step function
 - a composite block consists of a list of other blocks, e.g. the top level experiment, or a robot
    """
    
    defaults = {
        'id': None,
        'idim': None,
        'odim': None,
        'ibufsize': 1,
        'blocksize': 1,
        # 'obufsize': 1,
        'logging': True, # normal logging
        # 'savedata': True,
        'topblock': False,
        'ros': False,
        'debug': False,
        'inputs': [],
        'outputs': {'x': [1]} # name, dim
    }

    # @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        # fetch default block config
        for k,v in self.defaults.items():
            self.__dict__[k] = v
            
        self.debug_print("%s.__init__: conf = %s", (self.__class__.__name__, conf))
        # fetch configuration arguments if block is primitive and conf is a dict 
        if type(conf) == dict:
            for k,v in conf.items():
                self.__dict__[k] = v

        # auto-generate id if None supplied
        if self.id is None:
            # self.id = self.__class__.__name__ + "_%s" % uuid.uuid4()
            self.id = self.__class__.__name__ + "_%s", (uuid.uuid1().int>>64)

        # count steps
        self.cnt = 0

        # minimum blocksize downstairs
        self.blocksize_min = BLOCKSIZE_MAX

        # initialize local buffers
        self.init_bufs()

        # global bus (SMT)
        self.bus = bus
        # block's nodes
        self.nodes = None
        
        # init sub-nodes if this is a composite node
        if type(conf) is dict and conf.has_key('graph'):
            self.nodes = OrderedDict()
            # topblock is special, connecting experiment with the outside
            if self.topblock:
                # initialize global signal bus
                self.bus = {}
                # initialize global logging
                log.log_pd_init(conf)
            # for node_key, node_val in conf.items():
            for i, node in enumerate(conf['graph'].items()):
                nk, nv = node
                # print "key: %s, val = %s", (node_key, node_val)
                self.debug_print("node[%d] = %s(%s)", (i, nv['block'], nv['params']))
                # s = "%s(%s)", (node_val["block"], node_val["params"])
                # self.nodes[node_key] = node_val['block'](node_val['params'])
                nodekey = nv['params']['id'] # self.id # 'n%04d' % i
                self.debug_print("nodekey = %s", (nodekey))
                # self.nodes[nodekey] = Block(block = nv['block'], conf = nv['params'])
                # create node
                self.nodes[nodekey] = nv['block'](
                    block = nv['block'],
                    conf = nv['params'],
                    bus = self.bus)

                # blocksize check
                if self.nodes[nodekey].blocksize < self.blocksize_min:
                    self.blocksize_min = self.nodes[nodekey].blocksize
                    
                # initialize block logging
                for k, v in self.nodes[nodekey].outputs.items():
                    log.log_pd_init_block(
                        tbl_name = "%s/%s" % (self.nodes[nodekey].id, k),
                        tbl_dim = v[0], # odim
                        tbl_columns = ["out_%d" % col for col in range(v[0])],
                                        numsteps = self.numsteps)
            # print "%s.init: conf = %s"  %(self.__class__.__name__, conf)

            # done, all block added to top block, now reiterate
            for i, node in enumerate(self.nodes.items()):
                print "%s.init nodes.item = %s" % (self.__class__.__name__, node)
                nk, nv = node
                # FIXME: make one min_blocksize bus group for each node output
                for outkey, outparams in self.nodes[nk].outputs.items():
                    nodeoutkey = "%s/%s" % (nk, outkey)
                    print "bus %s, outkey %s, odim = %d" % (nk, nodeoutkey, outparams[0])
                    self.bus[nodeoutkey] = np.zeros((self.nodes[nk].odim, 1))
            
        # atomic block
        elif type(conf) is dict and block is not None:
            self.debug_print("block is %s, nothing to do", (block))
            # self.nodes['n0000'] = self
            # self.step = step_funcs[block]

    def init_bufs(self):
        self.bufs = {}
        # check for blocksize argument, ibuf needs to be at least of size blocksize
        self.ibufsize = max(self.ibufsize, self.blocksize)
        self.ibufidx = self.cnt
        # current index
        if not self.idim is None:
            self.bufs["ibuf"] = np.zeros((self.idim, self.ibufsize))
            # self.bufs["obuf"] = np.zeros((self.odim, self.obufsize))
            
    @decStep()
    def step(self, x = None):
        self.debug_print("%s-%s.step: x = %s", (self.__class__.__name__, self.id, x))
        # iterate all nodes and step them
        if self.nodes is not None:
            for k,v in self.nodes.items():
                x_ = v.step(x = x)
        else:
            # default action: copy input to output / identity
            x_ = x

        if self.topblock:
            # do logging
            for k, v in self.bus.items():
                self.debug_print("%s.step: bus k = %s, v = %s, %s", (self.__class__.__name__, k, v.shape))
                # print "%s id" % (self.__class__.__name__), self.id, "k", k, v.shape, v
                log.log_pd(nodeid = k, data = v)

        # store log
        if (self.cnt+1) % 100 == 0:
            log.log_pd_store()

        return(x_)

    def save(self, filename):
        """save this block into a pickle"""
        pass

    def load(self, filename):
        """load block from a pickle"""
        pass

    def debug_print(self, fmtstring, data):
        """only print if debug is enabled for this block"""
        if self.debug:
            print fmtstring % data

################################################################################
# Simple blocks for testing

class ConstBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        self.x = np.ones((self.odim, 1)) * self.const

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # loop over outputs dict and copy them to a slot in the bus
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)
        return self.x

class UniformRandomBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s, inputs = %s", (self.__class__.__name__, x, self.bus, self.inputs))
        self.hi = x
        self.x = np.random.uniform(self.lo, self.hi, (self.odim, 1))
        # loop over outputs dict and copy them to a slot in the bus
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            self.bus[buskey] = getattr(self, k)
        # self.bus[self.id] = self.x
        return self.x

# File reading

def read_puppy_hk_pickles(lfile, key = None):
    """read pickled log dicts from andi's puppy experiments"""
    d = pickle.load(open(lfile, 'rb'))
    # print "d.keys", d.keys()
    # data = d["y"][:,0,0] # , d["y"]
    rate = 20
    offset = 0
    # data = np.atleast_2d(data).T
    # print "wavdata", data.shape
    data = d
    return (data, rate, offset)
    
class FileBlock(Block):
    @decInit()
    def __init__(self, block = None, conf = None, bus = None):
        Block.__init__(self, block = block, conf = conf, bus = bus)
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        lfile = conf['file'][0]
        # puppy homeokinesis (andi)
        if lfile.startswith('data/pickles_puppy') and lfile.endswith('.pickle'):
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])

        # init states
        for key, v in self.outputs.items(): # ['x', 'y']:
            # print "key", self.data[key]
            setattr(self, key, np.zeros((self.data[key].shape[1], 1)))
            # self.x = np.zeros((self.odim, 1))
        
        # set odim from file
        self.odim = self.x.shape[1] # None # self.data.shape[1]

    @decStep()
    def step(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x))
        for k, v in self.outputs.items():
            buskey = "%s/%s" % (self.id, k)
            # self.bus[buskey] = getattr(self, k)
            if k.endswith('x'):
                print "endswith"
                self.bus[buskey] = self.data[k][[self.cnt],:,0].T
            else:
                self.bus[buskey] = self.data[k][[self.cnt]].T
        # self.bus[buskey] = self.x
        return self.x
