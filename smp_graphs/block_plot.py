# import itertools

from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from  matplotlib import rcParams
import numpy as np
import pandas as pd
# FIXME: soft import
import seaborn as sns

# perceptually uniform colormaps
import colorcet as cc

from smp_base.common     import get_module_logger
from smp_base.plot_utils import custom_legend, put_legend_out_right, put_legend_out_top
from smp_base.dimstack   import dimensional_stacking, digitize_pointcloud
from smp_base.plot       import makefig, timeseries, histogram, plot_img, plotfuncs, uniform_divergence
from smp_base.plot       import get_colorcycler, fig_interaction
from smp_base.plot       import ax_invert, ax_set_aspect

from smp_graphs.common import code_compile_and_run
from smp_graphs.block import decStep, decInit, block_cmaps, get_input
from smp_graphs.block import PrimBlock2
from smp_graphs.utils import myt, mytupleroll
import smp_graphs.utils_logging as log
from functools import reduce

################################################################################
# Plotting blocks
# pull stuff from
# smp/im/im_quadrotor_plot
# ...

# FIXME: do some clean up here
#  - unify subplot spec and options handling
#  - clarify preprocessing inside / outside plotblock
#  - general matrix / systematic combinations plotting for n-dimensional data
#   - from scatter_matrix to modality-timedelay matrix
#   - modality-timedelay matrix is: modalities on x, timedelay on y
#   - modality-timedelay matrix is: different dependency measures xcorr, expansion-xcorr, mi, rp, kldiv, ...
#   - information decomposition matrix (ica?)
# 
rcParams['figure.titlesize'] = 11

axes_spines = False
# smp_graphs style
rcParams['axes.grid'] = False
rcParams['axes.spines.bottom'] = axes_spines
rcParams['axes.spines.top'] = axes_spines
rcParams['axes.spines.left'] = axes_spines
rcParams['axes.spines.right'] = axes_spines
rcParams['axes.facecolor'] = 'none'
# rcParams['axes.labelcolor'] = .15
# rcParams['axes.labelpad'] = 4.0
rcParams['axes.titlesize'] = 10.0
rcParams['axes.labelsize'] = 8.0
rcParams['axes.labelweight'] = 'normal'
rcParams['legend.framealpha'] = 0.3
rcParams['legend.fontsize'] = 9.0
rcParams['legend.labelspacing'] = 0.5
rcParams['xtick.labelsize'] = 8.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.labelsize'] = 8.0
rcParams['ytick.direction'] = 'out'
# subplots
rcParams['figure.subplot.bottom'] = 0.12 # 0.11
rcParams['figure.subplot.left'] = 0.1 # 0.125
rcParams['figure.subplot.right'] = 0.9
rcParams['figure.subplot.top'] = 0.88 # 0.88

# f = open("rcparams.txt", "w")
# f.write("rcParams = %s" % (rcParams, ))
# f.close()

from logging import DEBUG as logging_DEBUG
import logging
logger = get_module_logger(modulename = 'block_plot', loglevel = logging_DEBUG)

def subplot_input_fix(input_spec):
    """subplot configuration convenience function
    
    Convert subplot configuration items into a list if they are singular types like numbers, strs and tuples
    """
    # assert input an array 
    if type(input_spec) is str or type(input_spec) is tuple:
        return [input_spec]
    else:
        return input_spec


class AnalysisBlock2(PrimBlock2):
    defaults = {
        'nocache': True,
        'saveplot': False,
        'savetype': 'jpg',
        'block_group': 'measure',
        'desc': 'Some kind of analysis'
        }
    def __init__(self, conf = {}, paren = None, top = None):
        # use inputs from logfile even in no-cached epxeriment
        self.inputs_log = None
        # saving plots
        self.saveplot = False
        self.savetype = "jpg"

        defaults = {}
        # defaults.update(Block2.defaults)
        defaults.update(PrimBlock2.defaults, **self.defaults)
        self.defaults = defaults
        
        PrimBlock2.__init__(self, conf = conf, paren = paren, top = top)
        # print "AnalysisBlock2.init", conf['params']['saveplot'], self.conf['params']['saveplot']
        # print "AnalysisBlock2.init saveplot =", self.saveplot

        # default title?
        if not hasattr(self, 'title'):
            # self.title = "%s - %s-%s" % (self.top.id, self.cname, self.id)
            # self.title = "%s of %s" % (self.cname, self.top.id[:20], )
            self.title = "%s of %s\nnumsteps = %d, caching = %s" % (self.id, self.top.id, self.top.numsteps, self.top.docache)
        
    def save(self):
        """Save the analysis, redirect to corresponding class method, passing the instance
        """
        if isinstance(self, FigPlotBlock2) or isinstance(self, SnsMatrixPlotBlock2):
            FigPlotBlock2.savefig(self)

    def check_plot_input(self, ink, args):
        i, j, k = args[:3]
        if ink not in self.inputs:
            # self._debug("    triggered: bus[%s] = %s, buskeys = %s" % (buskey, xself.bus[v['buskey']], bus.keys()))
            self._warning('plot_subplot pass 1 subplotconf[%d,%d] input[%d] = %s doesn\'t exist in self.inputs %s' % (
                i, j, k, ink, list(self.inputs.keys())))
            return False
        return self.inputs[ink]

    def check_plot_type(self, conf, defaults = {}):
        """subplot plotfunc configuration type fix function part 1: raw conf

        Get subplot configuration item 'plot' and make sure it is a list of function pointers

        Returns:
         - list of plotfunc pointers
        """
        # merge defaults with conf
        defaults.update(conf)
        conf = defaults

        # check 'plot' type
        if type(conf['plot']) is list:
            # check if str or func for each single element
            conf_plot = [self.check_plot_type_single(f) for f in conf['plot']]
            # conf_plot = conf['plot'] # [j]
            # assert conf_plot is not type(str), "FIXME: plot callbacks is array of strings, eval strings"
        elif type(conf['plot']) is str:
            # conf_plot = self.eval_conf_str(conf['plot'])
            rkey = 'fp'
            conf_plot = code_compile_and_run(code = '%s = %s' % (rkey, conf['plot']), gv = plotfuncs, lv = {}, return_keys = [rkey])
            if type(conf_plot) is list:
                conf_plot = self.check_plot_type(conf, defaults)
            else:
                conf_plot = [conf_plot]
        else:
            conf_plot = [conf['plot']]
        return conf_plot

    def check_plot_type_single(self, f):
        """subplot plotfunc configuration type fix function part 2: single list item

        Get subplot configuration item 'plot' and, if necessary, type-fix the value by translating strings to functions.

        Returns:
         - single function pointer
        """
        # convert a str to a func by compiling it
        if type(f) is str:
            # return self.eval_conf_str(f)
            rkey = 'fp'
            return code_compile_and_run(
                code = '%s = %s' % (rkey, f),
                gv = plotfuncs,
                lv = {},
                return_keys = [rkey]) # [rkey]
        else:
            return f

    def get_title_from_plot_type(self, plotfunc_conf):
        title = ""
        for plotfunc in plotfunc_conf:
            # get the plot type from the plotfunc type
            if hasattr(plotfunc, 'func_name'):
                # plain function
                plottype = plotfunc.__name__
            elif hasattr(plotfunc, 'func'):
                # partial'ized func
                plottype = plotfunc.func.__name__
            else:
                # unknown func type
                plottype = timeseries # "unk plottype"

            # append plot type to title
            title += " " + plottype
        return title
        

class BaseplotBlock2(AnalysisBlock2):
    """Plotting base class
    
    Common features for all plots.

    Variants:
     - :class:`FigPlotBlock2' is :mod:`matplotlib` figure based plot block
     - :class:`SnsMatrixPlotBlock2` is a :mod:`seaborn` based plot which do not cooperate with external figure handles

    Plot block_group is both measure *and* output [wins]
    """
    defaults = {
        'block_group': ['output', 'measure'],
    }
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # update child class 'self' defaults
        # self.defaults.update(BaseplotBlock2.defaults)
        defaults = {}
        # defaults.update(Block2.defaults)
        defaults.update(AnalysisBlock2.defaults, **self.defaults)
        self.defaults = defaults
        # super init
        AnalysisBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def prepare_saveplot(self):
        """if saveplot set, compute filename and register top.outputs of type fig
        """
        if self.saveplot:
            self.filename = '%s_%s.%s' % (self.top.datafile_expr, self.id, self.savetype)
            self.top.outputs['%s' % (self.id, )] = {
                'type': 'fig',
                'filename': self.filename,
                'label': self.top.id,
                'id': self.id,
                'desc': self.desc,
                'width': 1.0,
            }
    
class FigPlotBlock2(BaseplotBlock2):
    """FigPlotBlock2 class

    PlotBlock base class for matplotlib figure-based plots. Creates
    the figure and a gridspec on init, uses fig.axes in the step
    function
    
    Args:
     - blocksize(int): the blocksize
     - subplots(list): an array of arrays, each cell of that matrix contains one subplot configuration dict
     - subplotconf(dict): dict with entries *inputs*, a list of input keys, *plot*, a plot function pointer
    """
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # defaults
        self.wspace = 0.0
        self.hspace = 0.0
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        # configure figure and plot axes
        self.fig_rows = len(self.subplots)
        self.fig_cols = len(self.subplots[0])

        # create figure
        self.fig = makefig(
            rows = self.fig_rows, cols = self.fig_cols,
            wspace = self.wspace, hspace = self.hspace,
            title = self.title)
        # self.fig.tight_layout(pad = 1.0)
        # self.debug_print("fig.axes = %s", (self.fig.axes, ))

        self.prepare_saveplot()
        
        # FIXME: too special
        self.isprimitive = False
        
    @staticmethod
    def savefig(plotinst):
        """Save the figure 'fig' using configurable options

        Args:
         - plotinst(BaseplotBlock2): a plot block instance

        Returns:
         - None
        """
        # subplotstr = ''
        # if len(plotinst.subplots) > 0 and len(plotinst.subplots[0]) > 0 and plotinst.subplots[0][0].has_key('input'):
        #     subplotstr = "_".join(np.array(
        #         [[
        #             "r%d_c%d_%s" % (r, c, "_".join(subplot_input_fix(sbc['input'])),) for c,sbc in enumerate(sbr)
        #             ] for r, sbr in enumerate(plotinst.subplots)
        #         ]).flatten())

        # get filename from instance
        filename = plotinst.filename
        
        plotinst._debug("%s-%s.save filename = %s" % (plotinst.cname, plotinst.id, filename))

        if not hasattr(plotinst, 'savesize'):
            savescale = 3
            plotinst.savesize = (
                min(plotinst.fig_cols * 2.5 * savescale, 24),
                min(plotinst.fig_rows * 1.0 * savescale, 12))

        plotinst._debug("savesize w/h = %f/%f, fig_cols/fig_rows = %s/%s" % (plotinst.savesize[0], plotinst.savesize[1], plotinst.fig_cols, plotinst.fig_rows))
        plotinst.fig.set_size_inches(plotinst.savesize)

        # write the figure to file
        try:
            plotinst._info("%s-%s.save saving plot %s to filename = %s" % (plotinst.cname, plotinst.id, plotinst.title, filename))
            plotinst.fig.savefig(filename, dpi=300, bbox_inches="tight")
            # if plotinst.top.
            plotinst.top.outputs['latex']['figures'][plotinst.id] = {
                'filename': filename,
                'label': plotinst.top.id,
                'id': plotinst.id,
                'desc': plotinst.desc}
            # plotinst.fig.savefig(filename, dpi=300)
        except Exception as e:
            logger.error("%s.save saving failed with %s" % ('FigPlotBlock2', e))
            
    @decStep()
    def step(self, x = None):
        """Call the :func:`plot_subplots` function

        Makes sure
         - that there is some data to plot
         - that the data is loaded from the :data:`log_store` instead
           of the :class:`Bus` inputs if the :data:`inputs_log` is
           set.
        """
        
        # have inputs at all?
        if len(self.inputs) < 1: return

        # make sure that data has been generated
        if (self.cnt % self.blocksize) in self.blockphase: # or (xself.cnt % xself.rate) == 0:

            # HACK: override block inputs with log.log_store
            if self.inputs_log is not None:
                print("Using inputs from log.log_store = %s with keys = %s instead of bus" % (log.log_store.filename, list(log.log_store.keys()), ))
                # commit data
                log.log_pd_store()
                # iterate input items
                for ink, inv in list(self.inputs.items()):
                    bus = '/%s' % (inv['bus'], )
                    # print "ink", ink, "inv", inv['bus'], inv['shape'], inv['val'].shape
                    # check if a log exists
                    if bus in list(log.log_store.keys()):
                        # print "overriding bus", bus, "with log", log.log_store[bus].shape
                        # copy log data to input value
                        inv['val'] = log.log_store[bus].values.copy().T # reshape(inv['shape'])
            
            # for ink, inv in self.inputs.items():
            #     self.debug_print("[%s]step in[%s].shape = %s", (self.id, ink, inv['shape']))

            # run the plots
            plots = self.plot_subplots()
            
            # set figure title and show the fig
            # self.fig.suptitle("%s: %s-%s" % (self.top.id, self.cname, self.id))
            self.fig.show()

            # if self.saveplot:
            #     self.save_fig()
        else:
            self.debug_print("%s.step", (self.__class__.__name__,))

    def plot_subplots(self):
        """FigPlotBlock2.plot_subplots

        This is a stub and has to be implement by children classes.
        """
        print("%s-%s.plot_subplots(): implement me" % (self.cname, self.id,))

class PlotBlock2(FigPlotBlock2):
    """PlotBlock2 class
    
    Block for plotting timeseries and histograms
    """
    # PlotBlock2.defaults
    defaults = {
        # 'inputs': {
        #     'x': {'bus': 'x'}, # FIXME: how can this be known? id-1?
        # },
        'blocksize': 1,
        'xlim_share': True,
        'ylim_share': True,
        # 'subplots': [[{'input': ['x'], 'plot': timeseries}]],
        'subplots': [[{}]],
    }

    defaults_subplotconf = {
        'input': [],
        'xlabel': None,
        'ylabel': None,
        'loc': 'left',
        # 'xticks': False,
        # 'yticks': False,
    }
        
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        """loop over configured subplots and plot the data according to the configuration

        The function does not take any arguments. Instead, the args
        are taken from the :data:`subplots` member.

        subplots is a list of lists, specifying a the subplot
        grid. `subplots[:]` are rows and `subplots[:][:]` are the
        columns.

        Each subplot entry is a dictionary with the following keys:
         - input: list of block.inputs label keys
         - plot: function pointer for a plotting function like
           :func:`timeseries`, :func:`histogram`, ...
         - ndslice: a multidimensional slice selecting data from tensor input
         - shape: the shape of the data after nd-slicing
         - xslice: just the x-axis slice, usually time

        Arguments:
         - None

        Returns:
         - None
        """
        # self._debug("%s plot_subplots self.inputs = %s", self.cname, self.inputs)

        # subplots pass 0: remember ax limits
        sb_rows = len(self.subplots)
        sb_cols = len(self.subplots[0])
        
        rows_ylim_max = [(1e9, -1e9) for _ in range(sb_rows)]
        cols_xlim_max = [(1e9, -1e9) for _ in range(sb_cols)]

        # set default plot size when we know the subplot geometry
        default_plot_scale = 3
        default_plot_size = (sb_cols * 2.5 * default_plot_scale, sb_rows * 1 * default_plot_scale)
        self.fig.set_size_inches(default_plot_size)

        # subplots pass 1: the hard work, iterate over subplot config and build the plot
        for i, subplot in enumerate(self.subplots):   # rows are lists of dicts
            for j, subplotconf_ in enumerate(subplot): # columns are dicts
                if type(subplotconf_) is dict:
                    subplotconf = {}
                    subplotconf.update(self.defaults_subplotconf)
                    subplotconf.update(subplotconf_)
                    subplotconf_.update(subplotconf)
                    
                # assert subplotconf.has_key('input'), "PlotBlock2 needs 'input' key in the plot spec = %s" % (subplotconf,)
                # assert subplotconf.has_key('plot'), "PlotBlock2 needs 'plot' key in the plot spec = %s" % (subplotconf,)
                # empty gridspec cell
                if subplotconf is None or len(subplotconf) < 1:
                    self._warning('plot_subplots pass 1 subplot[%d,%d] no plot configured, subplotconf = %s' % (i, j, subplotconf))
                    return

                # convert conf items into list if they aren't (convenience)
                for input_spec_key in ['input', 'ndslice', 'shape']:
                    if input_spec_key in subplotconf:
                        subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                        # print "    id: %s, subplotconf[%s] = %s" % (self.id, input_spec_key, subplotconf[input_spec_key])

                # linear axes index from subplot row_i * col_j
                idx = (i*self.fig_cols)+j
                    
                # remember axes and their labels created during pass 1 e.g. by twin[xy]()
                axs = {
                    'main': {
                        'ax': self.fig.axes[idx],
                        'labels': []
                    }
                }

                # remember input data processed for plot input in plotdata
                plotdata = OrderedDict()
                # remember input data processed for plot input in plotdatad, dict storing more info on the plot
                plotdatad = OrderedDict()
                # remember distinct input variables
                plotvar = ' '
                # create a plot title
                title = ''
                if 'title' in subplotconf: title += subplotconf['title']

                # get this subplot's plotfunc configuration and make sure its a list
                plotfunc_conf = self.check_plot_type(subplotconf)
                # print "%s-%s plotfunc_conf = %s" % (self.cname, self.id, plotfunc_conf)
                assert type(plotfunc_conf) is list, "plotfunc_conf must be a list, not %s" % (type(plotfunc_conf), )

                # add plotfunc type to default title
                if title == '':
                    title += self.get_title_from_plot_type(plotfunc_conf)

                # generate labels all at once
                # l = [['%s[%d]' % (ink, invd) for invd in range(inv.shape[1])] for ink, inv in plotdata.items()]
                # self._debug("pre labels l = %s, items = %s" % (l, plotdata.items(), ))
                # labels = reduce(lambda x, y: x + y, l)
                labels = []

                ################################################################################
                # loop over subplot 'input'
                for k, ink in enumerate(subplotconf['input']):
                    
                    # FIXME: array'ize this loop
                    # vars: input, ndslice, shape, xslice, ...
                    input_ink = self.check_plot_input(ink, [i, j, k])
                    if not input_ink: continue
                    
                    # get numsteps of data for the input
                    if 'shape' not in input_ink:
                        input_ink['shape'] = input_ink['val'].shape
                    plotlen = input_ink['shape'][-1] # numsteps at shape[-1]

                    # set default slice
                    xslice = slice(0, plotlen)
                    # compute final shape of plot data, custom transpose from horiz time to row time
                    plotshape = mytupleroll(input_ink['shape'])
                    
                    # print "%s.subplots defaults: plotlen = %d, xslice = %s, plotshape = %s" % (self.cname, plotlen, xslice, plotshape)
                
                    # x axis slice spec
                    if 'xslice' in subplotconf:
                        # get slice conf
                        if type(subplotconf['xslice']) is list:
                            subplotconf_xslice = subplotconf['xslice'][k]
                        else:
                            subplotconf_xslice = subplotconf['xslice']
                        # set slice
                        xslice = slice(subplotconf_xslice[0], subplotconf_xslice[1])
                        # update plot length
                        plotlen = xslice.stop - xslice.start
                        # and plot shape
                        plotshape = (plotlen, ) + tuple((b for b in plotshape[1:]))
                    
                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s xslice xslice = %s, plotlen = %d, plotshape = %s" % (
                        i, j, k, ink, xslice, plotlen, plotshape))

                    # explicit shape key
                    # FIXME: shape overrides xslice
                    if 'shape' in subplotconf:
                        if len(subplotconf['shape']) > 1:
                            subplotconf_shape = subplotconf['shape'][k]
                        else:
                            subplotconf_shape = subplotconf['shape'][0]
                        # get the shape spec, custom transpose from horiz t to row t
                        plotshape = mytupleroll(subplotconf_shape)
                        # update plot length
                        plotlen = plotshape[0]
                        # and xsclice
                        xslice = slice(0, plotlen)

                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s shape xslice = %s, plotlen = %d, plotshape = %s" % (
                        i, j, k, ink, xslice, plotlen, plotshape))
                    
                    # configure x axis, default implicit number of steps
                    if 'xaxis' in subplotconf:
                        if type(subplotconf['xaxis']) is str and subplotconf['xaxis'] in list(self.inputs.keys()):
                            t = self.inputs[subplotconf['xaxis']]['val'].T[xslice] # []
                        else:
                            t = subplotconf['xaxis'] # self.inputs[ink]['val'].T[xslice] # []
                            self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s xaxis setting t = %s from subplotconf['xaxis']" % (
                                i, j, k, ink, t, ))
                    else:
                        if xslice.stop > plotlen:
                            t = np.linspace(0, plotlen - 1, plotlen)
                        else:
                            t = np.linspace(xslice.start, xslice.start+plotlen-1, plotlen)[xslice]
                    
                    # print "%s.plot_subplots k = %s, ink = %s" % (self.cname, k, ink)
                    # plotdata[ink] = input_ink['val'].T[xslice]
                    # if ink == 'd0':
                    #     print "plotblock2", input_ink['val'].shape
                    #     print "plotblock2", input_ink['val'][0,...,:]
                    # ink_ = "%s_%d" % (ink, k)
                    ink_ = "%d-%s" % (k + 1, ink)
                    # print "      input shape %s: %s" % (ink, input_ink['val'].shape)

                    # if explicit n-dimensional slice is given
                    if 'ndslice' in subplotconf:
                        # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                        # slice the data to spec, custom transpose from h to v time
                        ndslice = subplotconf['ndslice'][k]
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice ndslice = %s" % (
                            i, j, k, ink, ndslice))
                        plotdata[ink_] = myt(input_ink['val'])[ndslice]
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice sb['ndslice'] = %s, numslice = %d" % (
                            i, j, k, ink, subplotconf['ndslice'][k], len(subplotconf['ndslice'])))
                        self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s ndslice plotdata[ink_] = %s, input = %s" % (
                            i, j, k, ink, plotdata[ink_].shape, input_ink['val'].shape, ))
                    else:
                        plotdata[ink_] = myt(input_ink['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))
                        
                    # dual plotdata record
                    axd = axs['main']
                    # ax_ = axs['main']['ax']
                    if 'xtwin' in subplotconf:
                        if type(subplotconf['xtwin']) is list:
                            if subplotconf['xtwin'][k]:
                                if 'xtwin' not in axs:
                                    axs['xtwin'] = {'ax': axs['main']['ax'].twinx(), 'labels': []}
                                axd = axs['xtwin'] # ['ax']
                                
                        else:
                            if subplotconf['xtwin']:
                                if 'xtwin' not in axs:
                                    axs['xtwin'] = {'ax': axs['main']['ax'].twinx(), 'labels': []}
                                axd = axs['xtwin'] # ['ax']

                    assert plotdata[ink_].shape != (0,), "no data to plot"
                    # print "      input = %s" % input_ink['val']
                    # print "      id %s, ink = %s, plotdata = %s, plotshape = %s" % (self.id, ink_, plotdata[ink_], plotshape)
                    # plotdata[ink_] = plotdata[ink_].reshape((plotshape[1], plotshape[0])).T
                    plotdata[ink_] = plotdata[ink_].reshape(plotshape)
                    
                    # fix nans
                    plotdata[ink_][np.isnan(plotdata[ink_])] = -1.0
                    plotvar += "%s, " % (input_ink['bus'],)

                    # generate labels
                    # FIXME: range(?) when len(shape) != 2
                    numlabels = plotdata[ink_].shape[0]
                    if len(plotdata[ink_].shape) > 1:
                        numlabels = plotdata[ink_].shape[-1]
                    l1  = ['%s[%d]' % (ink, invd) for invd in range(numlabels)]
                    l2 = reduce(lambda x, y: x+y, l1)
                    
                    self._debug("plot_subplots pass 1 subplot[%d,%d] input[%d] = %s labels numlabels = %s, l1 = %s, l2 = %s" % (
                        i, j, k, ink, numlabels, l1, l2, ))
                    l = l1
                    labels.append(l)
                    
                    axd['labels'] += l # .append(l)
                    ax_ = axd['ax']
                    
                    # store ax, labels for legend
                    plotdatad[ink_] = {'data': plotdata[ink_], 'ax': ax_, 'labels': l}

                if len(labels) == 1:
                    labels = labels[0]
                elif len(labels) > 1:
                    l3 = reduce(lambda x, y: x+y, labels)
                    labels = l3
                    
                self._debug("plot_subplots pass 1 subplot[%d,%d] labels after subplotconf.input = %s" % (
                    i, j, labels, ))
                subplotconf['labels'] = labels
                # end loop over subplot 'input'
                ################################################################################

                
                ################################################################################
                # combine inputs into one backend plot call to automate color cycling etc
                if 'mode' in subplotconf:
                    """FIXME: fix dangling effects of stacking"""
                    # ivecs = tuple(myt(input_ink['val'])[xslice] for k, ink in enumerate(subplotconf['input']))
                    ivecs = [plotdatav for plotdatak, plotdatav in list(plotdata.items())]
                    # plotdata = {}
                    if subplotconf['mode'] in ['stack', 'combine', 'concat']:
                        plotdata['_stacked'] = np.hstack(ivecs)
                        plotdatad['_stacked'] = {'data': plotdata['_stacked'], 'ax': plotdatad[list(plotdata.keys())[0]]['ax'], 'labels': labels}

                # if type(subplotconf['input']) is list:
                if 'xaxis' in subplotconf:
                    if type(subplotconf['xaxis']) is str and subplotconf['xaxis'] in list(self.inputs.keys()):
                        inv = self.inputs[subplotconf['xaxis']]
                    else:
                        inv = self.inputs[ink]
                        
                    if 'bus' in inv:
                        plotvar += " over %s" % (inv['bus'], )
                    else:
                        plotvar += " over %s" % (inv['val'], )
                    # plotvar = ""
                    # # FIXME: if len == 2 it is x over y, if len > 2 concatenation
                    # for k, inputvar in enumerate(subplotconf['input']):
                    #     tmpinput = self.inputs[inputvar][2]
                    #     plotvar += str(tmpinput)
                    #     if k != (len(subplotconf['input']) - 1):
                    #         plotvar += " revo "
                # else:
                # plotvar = self.inputs[subplotconf['input'][0]][2]

                # transfer plot_subplot configuration keywords subplotconf to plot kwargs
                kwargs = {}
                for kw in [
                        'aspect', 'orientation', 'labels',
                        'title_pos',
                        'xlabel', 'xlim', 'xticks', 'xticklabels', 'xinvert', 'xtwin',
                        'ylabel', 'ylim', 'yticks', 'yticklabels', 'yinvert', 'ytwin', ]:
                    if kw in subplotconf:
                        kwargs[kw] = subplotconf[kw]
                self._debug("plot_subplots pass 1 subplot[%d,%d] kwargs = %s" % (i, j, kwargs))
                
                # prep axis
                ax = axs['main']['ax']
                # self.fig.axes[idx].clear()
                ax.clear()
                inkc = 0
                
                # colors
                num_cgroups = 5
                num_cgroup_color = 5
                num_cgroup_dist = 255/num_cgroups
                # cmap_str = 'cyclic_mrybm_35_75_c68'
                # cmap_str = 'colorwheel'
                cmap_str = 'rainbow'

                ax.set_prop_cycle(
                    get_colorcycler(
                        cmap_str = cmap_str, cmap_idx = None,
                        c_s = inkc * num_cgroup_dist, c_e = (inkc + 1) * num_cgroup_dist, c_n = num_cgroup_color
                    )
                )
                
                # stacked data?
                if '_stacked' in plotdata:
                    self._debug("plot_subplots pass 1 subplot[%d,%d] plotting stacked" % (i, j, ))
                    plotfunc_conf[0](ax, data = plotdata['_stacked'], ordinate = t, title = title, **kwargs)
                    # interaction
                    fig_interaction(self.fig, ax, plotdata['_stacked'])

                
                # iterate over plotdata items
                title_ = title
                inv_accum = []
                for ink, inv in list(plotdata.items()):
                    ax = plotdatad[ink]['ax']
                    self._debug("plot_subplots pass 1 subplot[%d,%d] plotdata[%s] = inv.sh = %s, plotvar = %s, t.sh = %s" % (
                        i, j, ink, inv.shape, plotvar, t.shape))

                    # if multiple input groups, increment color group
                    if inkc > 0:
                        ax.set_prop_cycle(
                            get_colorcycler(
                                cmap_str = cmap_str, cmap_idx = None,
                                c_s = (inkc + 1) * num_cgroup_dist, c_e = (inkc + 2) * num_cgroup_dist, c_n = num_cgroup_color
                            ),
                        )

                    # select single element at first slot or increment index with plotdata items
                    plotfunc_idx = inkc % len(plotfunc_conf)
                    
                    # # ax.set_prop_cycle(get_colorcycler(cmap_str = tmp_cmaps_[inkc]))
                    # for invd in range(inv.shape[1]):
                    #     label_ = "%s[%d]" % (ink, invd + 1)
                    #     if len(label_) > 16:
                    #         label_ = label_[:16]
                    #     labels.append(label_)
                        
                    # this is the plot function array from the config
                    if '_stacked' not in plotdata:
                        # print "    plot_subplots plotfunc", plotfunc_conf[plotfunc_idx]
                        # print "                      args", ax, inv, t, title, kwargs
                        plotfunc_conf[plotfunc_idx](ax = ax, data = inv, ordinate = t, title = title_, **kwargs)
                        # avoid setting title multiple times
                        # title_ = None

                    # label = "%s" % ink, title = title
                    # tmp_cmaps_ = [k for k in cc.cm.keys() if 'cyclic' in k and not 'grey' in k]

                    inv_accum.append(inv)
                        
                    # metadata
                    inkc += 1

                if len(plotdata) > 0:
                    inv_accum_ = np.hstack(inv_accum)
                    
                    # interaction
                    fig_interaction(self.fig, ax, inv_accum_)
                    
                # reset to main axis
                ax = axs['main']['ax']
                # store the final plot data
                # print "sbdict", self.subplots[i][j]
                sb = self.subplots[i][j]
                sb['p1_plottitle'] = title
                sb['p1_plotdata'] = plotdata
                sb['p1_plotvar'] = plotvar
                sb['p1_plotlabels'] = labels
                sb['p1_plotxlim'] = ax.get_xlim()
                sb['p1_plotylim'] = ax.get_ylim()
                sb['p1_axs'] = axs
                sb['p1_plotdatad'] = plotdatad

                # save axis limits
                # print "xlim", ax.get_xlim()
                if sb['p1_plotxlim'][0] < cols_xlim_max[j][0]: cols_xlim_max[j] = (sb['p1_plotxlim'][0], cols_xlim_max[j][1])
                if sb['p1_plotxlim'][1] > cols_xlim_max[j][1]: cols_xlim_max[j] = (cols_xlim_max[j][0], sb['p1_plotxlim'][1])
                if sb['p1_plotylim'][0] < rows_ylim_max[i][0]: rows_ylim_max[i] = (sb['p1_plotylim'][0], rows_ylim_max[i][1])
                if sb['p1_plotylim'][1] > rows_ylim_max[i][1]: rows_ylim_max[i] = (rows_ylim_max[i][0], sb['p1_plotylim'][1])

                    
                # self.fig.axes[idx].set_title("%s of %s" % (plottype, plotvar, ), fontsize=8)
                # [subplotconf['slice'][0]:subplotconf['slice'][1]].T)
        # subplots pass 1: done

        ################################################################################
        # subplots pass 2: clean up and compute globally shared dynamic vars
        # adjust xaxis
        for i, subplot in enumerate(self.subplots):
            idx = (i*self.fig_cols)            
            for j, subplotconf in enumerate(subplot):
                # subplot handle shortcut
                sb = self.subplots[i][j]
                
                self._debug("    0 subplotconf.keys = %s" % (list(subplotconf.keys()), ))
                
                # subplot index from rows*cols
                idx = (i*self.fig_cols)+j
                    
                # axis handle shortcut
                ax = self.fig.axes[idx]

                # check empty input
                if len(subplotconf['input']) < 1:
                    # ax = self.fig.axes[idx]
                    # ax = fig.gca()
                    ax.set_xticks([])
                    ax.set_xticklabels([])
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    continue
                    
                # consolidate axis limits
                if self.xlim_share and 'xlim' not in subplotconf:
                    # self._debug("subplots pass 2 consolidate ax[%d,%d] = %s" % (i, j, ax, cols_xlim_max[j]))
                    # self._debug("subplots pass 2             xlim = %s" % (cols_xlim_max[j]))
                    # self._debug("subplots pass 2             subplotconf.keys = %s" % (subplotconf.keys()))
                    ax.set_xlim(cols_xlim_max[j])
                if self.ylim_share and 'ylim' not in subplotconf:
                    # self._debug("subplots pass 2 consolidate ax[%d,%d] = %s" % (i, j, ax, rows_ylim_max[j]))
                    # self._debug("subplots pass 2             ylim = %s" % (rows_ylim_max[j]))
                    # self._debug("subplots pass 2             subplotconf.keys = %s" % (subplotconf.keys()))
                    ax.set_ylim(rows_ylim_max[i])

                # check axis inversion
                ax_invert(ax, **subplotconf)
                
                # fix legends
                # ax.legend(labels)
                loc = 'left'
                if 'legend_loc' in sb:
                    loc = sb['legend_loc']
                    
                # twin axes headache
                if len(sb['p1_axs']) == 1:
                    custom_legend(
                        labels = sb['p1_plotlabels'],
                        ax = ax, resize_by = 0.9,
                        loc = loc)
                    ax_set_aspect(ax, **subplotconf)
                else:
                    lg_ = None
                    # for axk, ax in sb['p1_axs'].items():
                    # for pdk, pdv in sb['plotdatad'].items():
                    #    ax = pdv['ax']
                    for k, axk in enumerate(sb['p1_axs']):
                        ax = sb['p1_axs'][axk]['ax']
                        labels = sb['p1_axs'][axk]['labels']
                        # if axk == 'main' and sb['p1_axs'].has_key('xtwin'): loc_ = 'right'
                        # else: loc_ = loc
                        # if lg_ is not None:
                        #     print "lg_.loc", lg_
                        # if loc == 'left': locx = 1.05
                        # elif loc == 'right': locx = -0.15
                        # else: locx = 0.0
                        # loc_ = (locx, (k * 0.45))
                        loc_ = loc
                        custom_legend(
                            labels = labels, # sb['p1_plotlabels'],
                            ax = ax, resize_by = 0.9,
                            loc = loc_, lg = lg_)
                        lg_ = ax.get_legend()

                        # set aspect after placing legend
                        # self._debug("    1 subplotconf.keys = %s" % (subplotconf.keys(), ))
                        ax_set_aspect(ax, **subplotconf)
                
                # put_legend_out_top(labels = labels, ax = ax, resize_by = 0.8)
                
        self._debug("plot_subplots len fig.axes = %d" % (len(self.fig.axes)))
            
        plt.draw()
        plt.pause(1e-9)

# plot a matrix via imshow/pcolor
class ImgPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        self._debug("plot_subplots self.inputs = %s" % (self.inputs, ))
        numrows = len(self.subplots)
        numcols = len(self.subplots[0])

        extrema = np.zeros((2, numrows, numcols))
        
        vmins_sb = [[] for i in range(numcols)]
        vmaxs_sb = [[] for i in range(numcols)]

        vmins = [None for i in range(numcols)]
        vmaxs = [None for i in range(numcols)]
        vmins_r = [None for i in range(numrows)]
        vmaxs_r = [None for i in range(numrows)]
                
        for i, subplot in enumerate(self.subplots): # rows
            for j, subplotconf in enumerate(subplot): # cols
                # check conditions
                assert 'shape' in subplotconf, "image plot needs shape spec"
                
                # make it a list if it isn't
                for input_spec_key in ['input', 'ndslice', 'shape']:
                    if input_spec_key in subplotconf:
                        subplotconf[input_spec_key] = subplot_input_fix(subplotconf[input_spec_key])
                        
                # for img plot use only first input item
                subplotin = self.inputs[subplotconf['input'][0]]
                # print "subplotin[%d,%d].shape = %s / %s" % (i, j, subplotin['val'].shape, subplotin['shape'])
                vmins_sb[j].append(np.min(subplotin['val']))
                vmaxs_sb[j].append(np.max(subplotin['val']))
                extrema[0,i,j] = np.min(subplotin['val'])
                extrema[1,i,j] = np.max(subplotin['val'])
                # print "i", i, "j", j, vmins_sb, vmaxs_sb
        self._debug("%s mins = %s" % (self.id, extrema[0], ))
        self._debug("%s maxs = %s" % (self.id, extrema[1], ))
        vmins_sb = np.array(vmins_sb)
        vmaxs_sb = np.array(vmaxs_sb)
        # print "vmins_sb, vmaxs_sb", i, j, vmins_sb.shape, vmaxs_sb.shape

        for i in range(numcols):
            vmins[i] = np.min(vmins_sb[i])
            # vmins[1] = np.min(vmins_sb[1])
            vmaxs[i] = np.max(vmaxs_sb[i])
            # vmaxs[1] = np.max(vmaxs_sb[1])

        # for i in range(numrows):
        #     vmins_r[i] = np.min(vmins_sb[i])
        #     # vmins[1] = np.min(vmins_sb[1])
        #     vmaxs_r[i] = np.max(vmaxs_sb[i])
        #     # vmaxs[1] = np.max(vmaxs_sb[1])
            
        rowmins = np.min(extrema[0], axis = 0) 
        rowmaxs = np.max(extrema[1], axis = 0) 
        colmins = np.min(extrema[0], axis = 1) 
        colmaxs = np.max(extrema[1], axis = 1)
        
        self._debug("plot_subplots rowmins = %s, rowmaxs = %s, colmins = %s, colmaxs = %s" % (rowmins, rowmaxs, colmins, colmaxs))
        
        if True:
            for i, subplot in enumerate(self.subplots): # rows
                for j, subplotconf in enumerate(subplot): # cols

                    # map loop indices to gridspec linear index
                    idx = (i*self.fig_cols)+j
                    # print "self.inputs[subplotconf['input']][0].shape", self.inputs[subplotconf['input'][0]]['val'].shape, self.inputs[subplotconf['input'][0]]['shape']

                    xslice = slice(None)
                    yslice = slice(None)
                    
                    # check for slice specs
                    if 'xslice' in subplotconf:
                        xslice = slice(subplotconf['xslice'][0], subplotconf['xslice'][1])
                        # print "xslice", xslice, self.inputs[subplotconf['input']][0].shape

                    if 'yslice' in subplotconf:
                        yslice = slice(subplotconf['yslice'][0], subplotconf['yslice'][1])
                        # print "yslice", yslice, self.inputs[subplotconf['input']][0].shape

                    # min, max values for colormap
                    axis = 0
                    aidx = j
                    if 'vaxis' in subplotconf:
                        if subplotconf['vaxis'] == 'rows':
                            axis = 1
                            aidx = i
                            
                    vmin = np.min(extrema[0], axis = axis)[aidx]
                    vmax = np.max(extrema[1], axis = axis)[aidx]
                    # print "vmins, vmaxs", i, vmins, vmaxs
                    # vmin = vmins[sbidx]
                    # vmax = vmaxs[sbidx]
                    # vmin = extrema[0]

                    # print "vmin", vmin, "vmax", vmax
                    if 'vmin' in subplotconf:
                        vmin = subplotconf['vmin']
                    if 'vmax' in subplotconf:
                        vmax = subplotconf['vmax']
                        
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][xslice,0]
                    # plotdata_cand = self.inputs[subplotconf['input']][0][:,xslice]
                    
                    # print "%s plot_subplots self.inputs[subplotconf['input'][0]]['val'].shape = %s" % (self.cname, self.inputs[subplotconf['input'][0]]['val'].shape)
                    # old version
                    # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][yslice,xslice]

                    ink = subplotconf['input'][0]
                    input_ink = self.check_plot_input(ink, [i, j, 0])
                    if not input_ink: continue
                        
                    # FIXME completeness if input is ndim, currently only first dim is handled
                    if 'ndslice' in subplotconf:
                        # di = subplotconf['ndslice'][0]
                        # dj = subplotconf['ndslice'][1]
                        # plotdata_cand = self.inputs[subplotconf['input'][0]]['val'][di, dj, :, -1]
                        # ink = subplotconf['input'][0]
                        plotdata_cand = myt(input_ink['val'])[subplotconf['ndslice'][0]]
                        # print "%s[%d]-%s.step plotdata_cand.shape = %s, ndslice = %s, shape = %s, xslice = %s, yslice = %s" % (self.cname, self.cnt, self.id, plotdata_cand.shape, subplotconf['ndslice'], subplotconf['shape'], xslice, yslice)
                        # print "plotdata_cand", plotdata_cand
                    else:
                        try:
                            # plotdata_cand = myt(self.inputs[subplotconf['input'][0]]['val'])[xslice,yslice]
                            plotdata_cand = myt(input_ink['val'])[xslice,yslice]
                        except Exception as e:
                            print(self.cname, self.id, self.cnt, self.inputs, subplotconf['input'])
                            # print "%s[%d]-%s.step, inputs = %s, %s " % (self.cname, self.cnt, self.id, self.inputs[subplotconf['input']][0].shape)
                            print(e)
                    #                                         self.inputs[subplotconf['input']][0])
                    # print "plotdata_cand", plotdata_cand.shape

                    ################################################################################
                    # digitize a random sample (continuous arguments, continuous values)
                    # to an argument grid and average the values
                    # FIXME: to separate function
                    if 'digitize' in subplotconf:
                        argdims = subplotconf['digitize']['argdims']
                        numbins = subplotconf['digitize']['numbins']
                        valdims = subplotconf['digitize']['valdim']

                        # print "%s.plot_subplots(): digitize argdims = %s, numbins = %s, valdims = %s" % (self.cname, argdims, numbins, valdims)
                        
                        # plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims)
                        plotdata_cand = digitize_pointcloud(data = plotdata_cand, argdims = argdims, numbins = numbins, valdims = valdims, f_fval = np.mean)
                    plotdata = {}

                    # if we're dimstacking, now is the time
                    if 'dimstack' in subplotconf:
                        plotdata['i_%d_%d' % (i, j)] = dimensional_stacking(plotdata_cand, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                        # print "plotdata[" + 'i_%d_%d' % (i, j) + "].shape", plotdata['i_%d_%d' % (i, j)].shape
                        # print "%s.plot_subplots(): dimstack x = %s, y = %s" % (self.cname, subplotconf['dimstack']['x'], subplotconf['dimstack']['y'])
                    else:
                        plotdata['i_%d_%d' % (i, j)] = plotdata_cand.reshape(subplotconf['shape'][0])
                    if 'ylog' in subplotconf:
                        # plotdata['i_%d_%d' % (i, j)] = np.log(plotdata['i_%d_%d' % (i, j)] + 1.0)
                        # print plotdata['i_%d_%d' % (i, j)]
                        yscale = 'log'
                    else:
                        yscale = 'linear'
                    plotvar = self.inputs[subplotconf['input'][0]]['bus']

                    title = "img plot"
                    if 'title' in subplotconf: title = subplotconf['title']
                    # for k, ink in enumerate(subplotconf['input']):
                    #     plotdata[ink] = input_ink[0].T[xslice]
                    #     # fix nans
                    #     plotdata[ink][np.isnan(plotdata[ink])] = -1.0
                    #     plotvar += "%s, " % (input_ink[2],)
                    # title += plotvar

                    # colormap
                    if 'cmap' not in subplotconf:
                        subplotconf['cmap'] = 'gray'
                    cmap = plt.get_cmap(subplotconf['cmap'])
                                                                
                    # plot the plotdata
                    for ink, inv in list(plotdata.items()):
                        # FIXME: put the image plotting code into function
                        ax = self.fig.axes[idx]
                        
                        inv[np.isnan(inv)] = -1.0

                        # Linv = np.log(inv + 1)
                        Linv = inv
                        # print "Linv.shape", Linv.shape
                        # print "Linv", np.sum(np.abs(Linv))
                        plotfunc = "pcolorfast"
                        plot_img(ax = ax, data = Linv, plotfunc = plotfunc,
                                     vmin = vmin, vmax = vmax, cmap = cmap,
                                     title = title)
        # update
        plt.draw()
        plt.pause(1e-9)

################################################################################
# from input vector of size n: set of vector elements, set of all undirected pairs
# from input time offsets: set of all time offset for a pair
# matrix of binary measure for all pairs (rows) for all time offsets (cols)

# compute delay bank (block)
# compute full variable stack vector (block)

# compute measure
# compute image0d
#  - avg MI
# compute image2d
#  - hexbin(x,y) or hexbin(x,y,C)
#  - hist2d
#  - scatter2d
#  - recurrence_plot_2d
#  - dimstack(imagend)
# compute imagend
#  - histnd
#  - scatternd

# plot stack of images: single array, list of arrays, list of lists of array
#  - subplotgrid
#  - imagegrid


class MatrixPlotBlock2(FigPlotBlock2):
    def __init__(self, conf = {}, paren = None, top = None):
        FigPlotBlock2.__init__(self, conf = conf, paren = paren, top = top)

    def plot_subplots(self):
        pass

################################################################################
# non FigPlot plot blocks
class SnsMatrixPlotBlock2(BaseplotBlock2):
    """SnsMatrixPlotBlock2 class

    Plot block for seaborn pairwaise matrix plots: e.g. scatter, hexbin, ...

    Seaborne (stubornly) manages figures itself, so it can't be a FigPlotBlock2
    
    Arguments:
    - blocksize: usually numsteps (meaning plot all data created by that episode/experiment)
    - f_plot_diag: diagonal cells
    - f_plot_matrix: off diagonal cells
    - numpy matrix of data, plot iterates over all pairs with given function
    """
    
    @decInit()
    def __init__(self, conf = {}, paren = None, top = None):
        # self.saveplot = False
        # self.savetype = 'jpg'
        BaseplotBlock2.__init__(self, conf = conf, paren = paren, top = top)

        self.prepare_saveplot()
        
    @decStep()
    def step(self, x = None):
        # print "%s.step inputs: %s"  % (self.cname, self.inputs.keys())

        subplotconf = self.subplots[0][0]
        
        # vector combination
        if 'mode' not in subplotconf:
            subplotconf['mode'] = 'stack'

        # plotting func
        subplotconf_plot = self.check_plot_type(subplotconf, defaults = {'plot': plt.hexbin})
        
        # if not subplotconf.has_key('plot'):
        #     subplotconf['plot'] = plt.hexbin
            
        # ilbls = [[['%s%d' % (self.inputs[ink]['bus'], j)] for j in range(self.inputs[ink]['shape'][0])] for i, ink in enumerate(subplotconf['input'])]
        # print "ilbls", ilbls
        # ivecs = tuple(self.inputs[ink]['val'].T for k, ink in enumerate(subplotconf['input']))
        
        def unwrap_ndslice(self, subplotconf, k, ink):
            # default x-axis slice
            xslice = slice(None)
            # apply ndslice
            if 'ndslice' in subplotconf:
                # plotdata[ink_] = myt(self.inputs[ink_]['val'])[-1,subplotconf['ndslice'][0],subplotconf['ndslice'][1],:] # .reshape((21, -1))
                print("      ndslice %s: %s, numslice = %d" % (ink, subplotconf['ndslice'][k], len(subplotconf['ndslice'])))
                plotdata = myt(self.inputs[ink]['val'])
                print("      ndslice plotdata", plotdata.shape)
                plotdata = plotdata[subplotconf['ndslice'][k]]
                print("      ndslice plotdata", plotdata.shape)
            else:
                plotdata = myt(self.inputs[ink]['val'])[xslice] # .reshape((xslice.stop - xslice.start, -1))
            print("       ndslice plotdata", plotdata.shape)
            
            # apply shape
            if 'shape' in subplotconf:
                if type(subplotconf['shape']) is list:
                    plotdata_shape = subplotconf['shape'][k]
                else:
                    plotdata_shape = subplotconf['shape']
                print("       ndslice plotshape", plotdata_shape)
            else:
                plotdata_shape = plotdata.T.shape

            plotdata = myt(plotdata).reshape(plotdata_shape)
            print("        shape plotdata", plotdata.shape)
    
            return plotdata    
            
        ivecs = []
        ilbls = []
        for k, ink in enumerate(subplotconf['input']):
            # ivec = myt(self.inputs[ink]['val'])
            ivec = unwrap_ndslice(self, subplotconf, k, ink)
            ivecs.append(ivec)
            ilbls += ['%s%d' % (self.inputs[ink]['bus'], j) for j in range(ivec.shape[0])] # range(self.inputs[ink]['shape'][0])]
            # ilbls.append(ilbl)
        print("ilbls", ilbls)
        
        # ivecs = tuple(myt(self.inputs[ink]['val']) for k, ink in enumerate(subplotconf['input']))
        # for ivec in ivecs:
        #     print "ivec.shape", ivec.shape
        plotdata = {}
        if subplotconf['mode'] in ['stack', 'combine', 'concat']:
            # plotdata['all'] = np.hstack(ivecs)
            plotdata['all'] = np.vstack(ivecs).T

        data = plotdata['all']
        print("data", data)
        
        print("SnsPlotBlock2:", data.shape)
        scatter_data_raw  = data
        # scatter_data_cols = ["x_%d" % (i,) for i in range(data.shape[1])]
        
        scatter_data_cols = np.array(ilbls).flatten().tolist()

        # prepare dataframe
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
        
        g = sns.PairGrid(df)
        # ud_cmap = cc.cm['diverging_cwm_80_100_c22'] # rainbow
        histcolorcycler = get_colorcycler('isoluminant_cgo_70_c39')
        # print histcolorcycler
        histcolor = histcolorcycler.by_key()['color'][:df.shape[1]]
        # print histcolor
        
        # # rcParams['axes.prop_cycle'] = histcolorcycler
        # g.map_diag(plt.hist, histtype = 'step')
        # for i in range(df.shape[1]):
        #     ax_diag = g.axes[i,i]
        #     # print type(ax_diag), dir(ax_diag)
        #     ax_diag.grid()
        #     ax_diag.set_prop_cycle(histcolorcycler)
            
        # g.map_diag(sns.kdeplot)
        # g.map_offdiag(plt.hexbin, cmap="gray", gridsize=40, bins="log");
        # g.map_offdiag(plt.histogram2d, cmap="gray", bins=30)
        # g.map_offdiag(plt.plot, linestyle = "None", marker = "o", alpha = 0.5) # , bins="log");
        plotf = partial(uniform_divergence, f = subplotconf_plot)
        g.map_diag(plotf)
        # g = g.map_diag(sns.kdeplot, lw=3, legend=False)
        g.map_offdiag(plotf) #, cmap="gray", gridsize=40, bins='log')

        # clean up figure
        self.fig = g.fig
        self.fig_rows, self.fig_cols = g.axes.shape
        self.fig.suptitle(self.title)
        # print "dir(g)", dir(g)
        # print g.diag_axes
        # print g.axes
        # if self.saveplot:
        #     FigPlotBlock2.save(self)
        # for i in range(data.shape[1]):
        #     for j in range(data.shape[1]): # 1, 2; 0, 2; 0, 1
        #         if i == j:
        #             continue
        #         # column gives x axis, row gives y axis, thus need to reverse the selection for plotting goal
        #         # g.axes[i,j].plot(df["%s%d" % (self.cols_goal_base, j)], df["%s%d" % (self.cols_goal_base, i)], "ro", alpha=0.5)
        #         g.axes[i,j].plot(df["x_%d" % (j,)], df["x_%d" % (i,)], "ro", alpha=0.5)

        # plt.show()
        
                    
