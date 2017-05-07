
import matplotlib.pyplot as plt
import numpy as np

from smp_graphs.block import decStep, decInit
from smp_graphs.block import PrimBlock2

from smp_base.plot import makefig, timeseries, histogram

################################################################################
# Plotting blocks

class TimeseriesPlotBlock2(PrimBlock2):
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)

    @decStep()
    def step(self, x = None):
        if len(self.inputs) < 1: return
        # print "plotblock conf", self.ibuf
        # print "plotblock real", self.inputs['d1'][0].shape
        if (self.cnt % self.blocksize) == 0: # (self.blocksize - 1):
            # self.debug_print("step ibuf = %s, in(%s).shape = %s", (self.ibuf, ink, inv[0]))
            # plt.plot(self.bufs['ibuf'].T)
            # plt.show()
            
            rows = len(self.subplots)
            cols = len(self.subplots[0])
            # create figure
            fig = makefig(rows = rows, cols = cols)
            # self.debug_print("fig.axes = %s", (fig.axes, ))

            # loop over configured subplots
            for i, subplot in enumerate(self.subplots):
                for j, subplotconf in enumerate(subplot):
                    idx = (i*cols)+j
                    # self.debug_print("%s.step idx = %d, conf = %s, data = %s", (
                    #     self.__class__.__name__, idx,
                    #     subplotconf, self.inputs[subplotconf['input']][0]))
                    if type(subplotconf['input']) is str:
                        t = np.linspace(0, self.blocksize-1, self.blocksize)
                        plotdata = self.inputs[subplotconf['input']][0].T
                    elif type(subplotconf['input']) is list:
                        t = self.inputs[subplotconf['input'][0]][0].T
                        plotdata = self.inputs[subplotconf['input'][1]][0].T
                    # fix nans
                    plotdata[np.isnan(plotdata)] = -1.0
                    if hasattr(subplotconf['plot'], 'func_name'):
                        # plain function
                        plottype = subplotconf['plot'].func_name
                    elif hasattr(subplotconf['plot'], 'func'):
                        # partial'ized func
                        plottype = subplotconf['plot'].func.func_name
                    else:
                        # unknown func type
                        plottype = "unk type"

                    if type(subplotconf['input']) is list:
                        plotvar = ""
                        # FIXME: if len == 2 it is x over y, if len > 2 concatenation
                        for k, inputvar in enumerate(subplotconf['input']):
                            tmpinput = self.inputs[inputvar][2]
                            plotvar += str(tmpinput)
                            if k != (len(subplotconf['input']) - 1):
                                plotvar += " revo "
                    else:
                        plotvar = self.inputs[subplotconf['input']][2]
                        
                    print "plotvar", plotvar
                        
                    # plot the plotdata
                    subplotconf['plot'](
                        fig.axes[idx],
                        data = plotdata, ordinate = t)
                    # metadata
                    fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                    # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)

                    
                    
                    # timeseries(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
                    # histogram(fig.axes[idx], self.bufs['ibuf'][plotcol[0]:plotcol[1]].T)
            fig.show()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))
            
