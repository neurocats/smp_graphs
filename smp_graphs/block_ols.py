import numpy as np
import pandas as pd

from smp_graphs.block import Block2, decInit, decStep

# FIXME: this should go into systems    
class FileBlock2(Block2):
    """!@brief File block: read some log or data file and output blocksize lines each step"""
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # ad hoc default
        if not conf['params'].has_key('type'): conf['params']['type'] = 'puppy'
        # multiple files: concat? block manipulation blocks?
        self.file = []
        # auto odim
        # if self.odim == 'auto':
        # print conf
        lfile = conf['params']['file'][0]
        # puppy homeokinesis (andi)
        if conf['params']['type'] == 'puppy':
            (self.data, self.rate, self.offset) = read_puppy_hk_pickles(lfile)
            # setattr(self, 'x', self.data['x'])
            # setattr(self, 'x', self.data['x'])
            self.step = self.step_puppy
        # selflogconfs
        elif conf['params']['type'] == 'selflogconf':
            self.store = pd.HDFStore(lfile)
            self.step = self.step_selflogconf
        elif conf['params']['type'] == 'selflog':
            self.store = pd.HDFStore(lfile)
            # clean up dummy entry
            del conf['params']['outputs']['log']
            # loop log tables
            for k in self.store.keys():
                print "%s" % self.__class__.__name__, k, self.store[k].shape
                conf['params']['outputs'][k] = [self.store[k].T.shape]
                conf['params']['blocksize'] = self.store[k].shape[0]
            self.step = self.step_selflog

        # init states
        for k, v in conf['params']['outputs'].items(): # ['x', 'y']:
            # print "key", self.data[key]
            # setattr(self, k, np.zeros((self.data[k].shape[1], 1)))
            # print "v[0]", v[0]
            if v[0] is None:
                # self.outputs[k][0] = (self.data[k].shape[1], 1)
                # print "data", self.data[k].shape[1]
                # print "out dim", conf['params']['outputs'][k]
                conf['params']['outputs'][k][0] = (self.data[k].shape[1], conf['params']['blocksize'])
                # print "out dim", conf['params']['outputs'][k]
            # self.x = np.zeros((self.odim, 1))
        
        Block2.__init__(self, conf = conf, paren = paren, top = top)
        
        # set odim from file
        # self.odim = self.x.shape[1] # None # self.data.shape[1]

    @decStep()
    def step(self, x = None):
        pass
    
    @decStep()
    def step_puppy(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        # self.x = np.atleast_2d(self.data[[self.cnt]]).T #?
        self.debug_print("self.x = %s", (self.x,))
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            for k, v in self.outputs.items():                
                sl = slice(self.cnt-self.blocksize, self.cnt)
                setattr(self, k, self.data[k][sl].T)
                # setattr(self, k, self.data[k][[self.cnt]].T)
            
    @decStep()
    def step_selflog(self, x = None):
        if (self.cnt % self.blocksize) == 0:
            for k, v in self.outputs.items():
                # if k.startswith('conf'):
                print "step: cnt = %d key = %s, log.sh = %s" % (self.cnt, k, self.store[k].shape)
                # print self.store[k].values
                setattr(self, k, self.store[k][self.cnt-self.blocksize:self.cnt].values.T)
                # print self.store[k][self.cnt-self.blocksize:self.cnt].values.T
                # print k
        # for k in self.__dict__.keys(): #self.bus.keys():
            # k = k.replace("/", "_")
            # print "k", k
                        
    @decStep()
    def step_selflogconf(self, x = None):
        self.debug_print("%s.step: x = %s, bus = %s", (self.__class__.__name__, x, self.bus))
        if (self.cnt % self.blocksize) == 0:
            for k, v in self.outputs.items():
                if k.startswith('conf'):
                    print "%s = %s\n" % (k, self.store.get_storer(k).attrs.conf,)
