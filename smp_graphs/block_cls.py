"""smp_graphs systems blocks

a system block is a wrapper for a system from smp_sys

TODO
 - replace proprio / extero by a general list of modalities
 - general formulation of both pm + sa with only individual step functions
"""

import numpy as np

from smp_graphs.block import decStep, PrimBlock2
from smp_sys.systems import PointmassSys
from smp_sys.systems import SimplearmSys
# BHA / mathias
# stdr
# sphero
# 

class SysBlock2(PrimBlock2):
    """!@brief Basic system block"""
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)


class PointmassBlock2(SysBlock2):
    """!@brief Pointmass system block, very thin wrapper around smp_sys.systems.PointmassSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = PointmassSys(conf['params'])
        # latent output variables defined by pointmass system
        self.x = {
            's_proprio': np.zeros((self.sysdim,   self.blocksize)),
            's_extero':  np.zeros((self.sysdim,   self.blocksize)),
            's_all':     np.zeros((self.statedim, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            self.x = self.system.step(self.u)
            # print "self.u", self.u
            # real output variables defined by config
            # for k in ['s_proprio', 's_extero', 's_all']:
            for k in self.outputs.keys():
                k_ = getattr(self, k)
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)

                
class SimplearmBlock2(SysBlock2):
    """!@brief Simplearm system block, very thin wrapper around smp_sys.systems.SimplearmSys"""
    def __init__(self, conf = {}, paren = None, top = None):
        SysBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.debug_print("init: conf = %s", (conf,))
        self.system = SimplearmSys(conf['params'])
        # latent output variables defined by simplearm system
        self.x = {
            's_proprio': np.zeros((self.sysdim,   self.blocksize)),
            's_extero':  np.zeros((self.dim_s_extero, self.blocksize)),
            's_all':     np.zeros((self.statedim, self.blocksize)),
        }
        # copy those into self attributes
        for k in ['s_proprio', 's_extero', 's_all']:
            setattr(self, k, self.x[k])
            # print "%s.init[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
        

    @decStep()
    def step(self, x = None):
        for i in range(self.blocksize):
            self.u = self.inputs['u']['val'][:,[i]]
            self.x = self.system.step(self.u)
            # print "self.u", self.u
            # real output variables defined by config
            # for k in ['s_proprio', 's_extero', 's_all']:
            for k in self.outputs.keys():
                k_ = getattr(self, k)
                k_[:,[i]] = self.x[k]
                # setattr(self, k, self.x[k])
                # print "%s.step[%d]: x = %s/%s" % (self.cname, self.cnt, self.x, self.system.x)
                
