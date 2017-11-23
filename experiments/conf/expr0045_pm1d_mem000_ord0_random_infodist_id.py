"""smp_graphs configuration

baseline behaviour - open-loop uniform random search in finite isotropic space identity.

id:thesis_smp_expr0050

Oswald Berthold 2017

special case of kinesis with coupling = 0 between measurement and action
"""

from smp_base.plot import table

from smp_graphs.block import FuncBlock2, TrigBlock2
from smp_graphs.block_cls import PointmassBlock2, SimplearmBlock2
from smp_graphs.block_models import ModelBlock2
from smp_graphs.block_meas import MeasBlock2, MomentBlock2
from smp_graphs.block_meas_infth import MIBlock2, InfoDistBlock2

from smp_graphs.funcs import f_sin, f_motivation, f_motivation_bin, f_meansquare, f_sum

from smp_graphs.utils_conf import get_systemblock

# global parameters can be overwritten from the commandline
ros = False
numsteps = 10000/10
numbins = 21
recurrent = True
debug = False
showplot = True
saveplot = True
randseed = 126

desc = """The purpose of this experiment is to illustrate the effect
of projection distortion and external entropy on the information
distance between two sensory modalities. This is done using a
parameterized model of these effects and a few examples of different
configurations are shown in the plot."""

# local conf dict for looping
lconf = {
    'dim': 1,
    'dt': 0.1,
    'lag': 1,
    'budget': 1000/1,
    'lim': 1.0,
    'order': 0,
    'd_i': 0.0,
    'infodistgen': {
        'type': 'random_lookup',
        'numelem': 1001,
        'l_a': 1.0,
        'd_a': 0.0,
        'd_s': 1.0,
        's_a': 0.0,
        's_f': 3.0,
        'e': 0.0,
    },
    'div_meas': 'chisq', # 'kld'
}

div_meas = lconf['div_meas']

meas_hist_bins       = np.linspace(-1.1, 1.1, numbins + 1)
meas_hist_bincenters = meas_hist_bins[:-1] + np.mean(np.abs(np.diff(meas_hist_bins)))/2.0

# local variable shorthands
dim = lconf['dim']
order = lconf['order']
budget = lconf['budget'] # 510
lim = lconf['lim'] # 1.0

outputs = {
    'latex': {'type': 'latex',},
}

# configure system block 
systemblock   = get_systemblock['pm'](
    dim_s_proprio = dim, dim_s_extero = dim, lag = 1, order = order)
# systemblock   = get_systemblock['sa'](
#     dim_s_proprio = dim, dim_s_extero = dim, lag = 1)
systemblock['params']['sysnoise'] = 0.0
systemblock['params']['anoise_std'] = 0.0
dim_s_proprio = systemblock['params']['dim_s_proprio']
dim_s_extero  = systemblock['params']['dim_s_extero']
# dim_s_goal   = dim_s_extero
dim_s_goal    = dim_s_proprio


infodistgen = lconf['infodistgen']

# print "sysblock", systemblock['params']['dim_s_proprio']

# TODO
# 1. loop over randseed
# 2. loop over budget vs. density (space limits, distance threshold), hyperopt
# 3. loop over randseed with fixed optimized parameters
# 4. loop over kinesis variants [bin, cont] and system variants ord [0, 1, 2, 3?] and ndim = [1,2,3,4,8,16,...,ndim_max]

# TODO low-level
# block groups
# experiment sig, make hash, store config and logfile with that hash
# compute experiment hash: if exists, use logfile, else compute
# compute experiment/model_i hash: if exists, use pickled model i, else train
# pimp smp_graphs graph visualisation

# graph
graph = OrderedDict([
    # triggers
    ('trig', {
        'block': TrigBlock2,
        'params': {
            'trig': np.array([numsteps]),
            'outputs': {
                'pre_l2_t1': {'shape': (1, 1)},
            }
        },
    }),

    # robot
    ('robot1', systemblock),
        
    # brain
    ('brain', {
        # FIXME: idea: this guy needs to pass down its input/output configuration
        #        to save typing / errors on the individual modules
        'block': Block2,
        'params': {
            'numsteps': 1, # numsteps,
            'id': 'brain',
            'nocache': True,
            'graph': OrderedDict([
                # every brain has a budget
                ('budget', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'credit': np.ones((1, 1)) * budget,
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {
                            # 'credit': {'bus': 'budget/credit', 'shape': (1,1)},
                            # 's0': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            's0': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, 1)},
                            's0_ref': {'bus': 'pre_l1/pre', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'credit': {'shape': (1,1)},
                        },
                        'models': {
                            'budget': {'type': 'budget_linear'},
                        },
                        'rate': 1,
                    },
                }),

                # new modality m2 with distance parameter
                ('pre_l2', {
                    'block': ModelBlock2,
                    'params': {
                        'debug': False,
                        'models': {
                            # 'infodistgen': {
                            #     'type': 'random_lookup',
                            #     'numelem': 1001,
                            #     # 'd': 0.5,
                            #     # 's_a': 0.5,
                            #     # 's_f': 1.0,
                            #     # 'e': 0.0,
                            #     'l_a': 1.0,
                            #     'd_a': 1.0,
                            #     'd_s': 1.0,
                            #     's_a': 0.1,
                            #     's_f': 3.0,
                            #     'e': 0.0,
                            # },
                            'infodistgen': infodistgen,
                        },
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                        },
                        'outputs': {
                            'y': {'shape': (dim_s_proprio, 1)},
                            'h': {'shape': (dim_s_proprio, 1001), 'trigger': 'trig/pre_l2_t1'},
                        },
                    }
                }),

                # uniformly dist. random goals, triggered when error < goalsize
                ('pre_l1', {
                    'block': ModelBlock2,
                    'params': {
                        'blocksize': 1,
                        'blockphase': [0],
                        'rate': 1,
                        # 'ros': ros,
                        'goalsize': 0.1, # np.power(0.01, 1.0/dim_s_proprio), # area of goal
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim, 'shape': (dim_s_proprio, 1)},
                            'hi': {'val': lim, 'shape': (dim_s_proprio, 1)},
                            # 'mdltr': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, 1)},
                            'mdltr': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, 1)},
                            },
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                        },
                        'models': {
                            'goal': {'type': 'random_uniform_modulated'}
                            },
                        },
                    }),
                    
                # uniformly distributed random action, no modulation
                ('pre_l0', {
                    'block': UniformRandomBlock2,
                    'params': {
                        'id': 'search',
                        'inputs': {
                            'credit': {'bus': 'budget/credit'},
                            'lo': {'val': -lim},
                            'hi': {'val': lim}},
                        'outputs': {
                            'pre': {'shape': (dim_s_proprio, 1)},
                        }
                    },
                }),
                
                # measure mutual information I(m1;m2)
                ('mi', {
                    'block': MIBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'shift': (0, 1),
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            # 'y': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                        },
                        'outputs': {
                            'mi': {'shape': (1, 1, 1)},
                        }
                    }
                }),
                # measure information distance d(m1, m2)
                ('infodist', {
                    'block': InfoDistBlock2,
                    'params': {
                        'blocksize': numsteps,
                        'shift': (0, 1),
                        'inputs': {
                            'x': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            # 'y': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                            'y': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                        },
                        'outputs': {
                            'infodist': {'shape': (1, 1, 1)},
                        }
                    }
                }),
            ]),
        }
    }),
    
    # # robot
    # ('robot1', systemblock),

    # measures
    ('meas_budget', {
        'block': MomentBlock2,
        'params': {
            'id': 'meas_budget',
            # 'debug': True,
            'blocksize': numsteps,
            'inputs': {
                # 'credit': {'bus': 'pre_l1/credit', 'shape': (1, numsteps)},
                'y': {'bus': 'budget/credit', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y_mu': {'shape': (1, 1)},
                'y_var': {'shape': (1, 1)},
                'y_min': {'shape': (1, 1)},
                'y_max': {'shape': (1, 1)},
            },
        },
    }),

    # measures
    ('meas_err', {
        'block': MeasBlock2,
        'params': {
            'id': 'meas_err',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'basic',
            'scope': 'local',
            'meas': 'sub',
            'inputs': {
                'x1': {'bus': 'robot1/s_proprio', 'shape': (1, numsteps)},
                'x2': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y': {'shape': (1, numsteps)},
            },
        },
    }),
    
    # measures
    ('meas_mse', {
        'block': FuncBlock2,
        'params': {
            # 'id': 'meas_mse',
            'blocksize': numsteps,
            'debug': False,
            'func': f_meansquare,
            'inputs': {
                'x': {'bus': 'meas_err/y', 'shape': (1, numsteps)},
            },
            'outputs': {
                'y': {'shape': (1, 1)},
            },
        },
    }),
    
    # measures
    ('meas_hist', {
        'block': MeasBlock2,
        'params': {
            'id': 'meas_hist',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'hist',
            'scope': 'local',
            'meas': 'hist',
            # direct histo input?
            # or signal input
            'inputs': {
                'x1': {'bus': 'robot1/s_proprio', 'shape': (1, numsteps)},
                'x2': {'bus': 'pre_l2/y', 'shape': (1, numsteps)},
            },
            'bins': meas_hist_bins,
            'outputs': {
                'h_x1': {'shape': (1, numbins)},
                'h_x2': {'shape': (1, numbins)},
            },
        },
    }),
    
    # measures
    ('meas_div', {
        'block': MeasBlock2,
        'params': {
            'id': 'meas_div',
            'blocksize': numsteps,
            'debug': False,
            'mode': 'basic',
            'scope': 'local',
            'meas': div_meas, # ['chisq', 'kld'],
            # direct histo input?
            # or signal input
            'inputs': {
                'x1': {'bus': 'meas_hist/h_x1', 'shape': (1, numbins)},
                'x2': {'bus': 'meas_hist/h_x2', 'shape': (1, numbins)},
            },
            'outputs': {
                'y': {'shape': (1, numbins)},
            },
        },
    }),
    
    # measures
    ('meas_sum_div', {
        'block': FuncBlock2,
        'params': {
            'blocksize': numsteps,
            'debug': False,
            'func': f_sum,
            'inputs': {
                'x': {'bus': 'meas_div/y', 'shape': (1, numbins)},
            },
            'outputs': {
                'y': {'shape': (1, 1)},
            },
        },
    }),
    
    # plotting random_lookup influence
    # one configuration plot grid:
    # | transfer func h | horizontal output | horziontal histogram |
    # | vertical input  | information meas  | -                    |
    # | vertical histo  | -                 | - (here the model)   |
    ('plot_infodist', {
        'block': PlotBlock2,
        'params': {
            'debug': False,
            'blocksize': numsteps,
            'saveplot': saveplot,
            'savetype': 'pdf',
            'wspace': 0.15,
            'hspace': 0.1,
            'xlim_share': True,
            'ylim_share': True,
            'inputs': {
                's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
                's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
                'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
                'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
                'pre_l2_h': {'bus': 'pre_l2/h', 'shape': (dim_s_proprio, 1001)},
                'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
                'budget_mu': {'bus': 'meas_budget/y_mu', 'shape': (1, 1)},
                'budget_var': {'bus': 'meas_budget/y_var', 'shape': (1, 1)},
                'budget_min': {'bus': 'meas_budget/y_min', 'shape': (1, 1)},
                'budget_max': {'bus': 'meas_budget/y_max', 'shape': (1, 1)},
                'infodist': {
                    'bus': 'infodist/infodist',
                    'shape': (dim_s_proprio, 1, 1)
                },
                'mi': {
                    'bus': 'mi/mi',
                    'shape': (dim_s_proprio, 1, 1)
                },
                'meas_err': {'bus': 'meas_err/y', 'shape': (1, numsteps)},
                'meas_mse': {'bus': 'meas_mse/y', 'shape': (1, 1)},
                'meas_div': {'bus': 'meas_div/y', 'shape': (1, numbins)},
                'meas_sum_div': {'bus': 'meas_sum_div/y', 'shape': (1, 1)},
            },
            'desc': 'Single infodist configuration',

            # subplot
            'subplots': [
                # row 1: transfer func, out y time, out y histo
                [
                    {
                        'input': ['pre_l2_h'], 'plot': timeseries,
                        'title': 'transfer function $h$', 'aspect': 1.0, 
                        'xaxis': np.linspace(-1, 1, 1001), # 'xlabel': 'input [x]',
                        'xlim': (-1.1, 1.1), 'xticks': True, 'xticklabels': False,
                        'ylabel': 'output $y = h(x)$',
                        'ylim': (-1.1, 1.1), 'yticks': True,
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['pre_l2'], 'plot': timeseries,
                        'title': 'timeseries $y$', 'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'xlim': None, 'xticks': False, 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                    {
                        'input': ['pre_l2'], 'plot': histogram,
                        'title': 'histogram $y$', 'aspect': 'auto', # (1*numsteps)/(2*2.2),
                        'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, 'xticklabels': None,
                        'xlabel': 'count $c$',
                        'ylim': (-1.1, 1.1),
                        'yticks': True, 'yticklabels': False,
                        'legend_loc': 'left',
                    },
                ],
                
                # row 2: in x time, error x - y time, none
                [
                    {
                        'input': ['s_p'], 'plot': timeseries,
                        'title': 'timeseries $x$',
                        'aspect': 2.2/numsteps,
                        'orientation': 'vertical',
                        'xlim': None, 'xticks': False, # 'xticklabels': False,
                        # 'xlabel': 'time step $k$',
                        'yticks': False,
                        'ylim': (-1.1, 1.1),
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['meas_err'], 'plot': timeseries,
                        'title': 'error $x - y$',
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xlabel': 'time step $k$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': (-1.1, 1.1), # None,
                        # 'legend_loc': 'right',
                    },
                    {},
                ],
                
                # row 3: in x histo, measures global, divergence h1, h2
                [
                    {
                        'input': ['s_p'], 'plot': histogram,
                        'title': 'histogram $x$', 'aspect': 'shared', # (1*numsteps)/(2*2.2),
                        'orientation': 'vertical',
                        'xlim': (-1.1, 1.1), 'xinvert': False, # 'xticks': False, 'xticklabels': None, #
                        'xlabel': 'input $x \in [-1, ..., 1]$', 
                        'ylim': None, 'yinvert': True,  # 'yticks': None, 'yticklabels': None,
                        'ylabel': 'count $c$',
                        'legend_loc': 'right',
                    },
                    {
                        'input': ['budget_%s' % (outk,) for outk in ['mu', 'var', 'min', 'max']] + ['mi', 'infodist', 'meas_mse', 'meas_sum_div'], # , 'meas_mkld']
                        'shape': [(1, 1) for outk in ['mu', 'var', 'min', 'max', 'mi', 'infodist', 'meas_mse', 'meas_sum_div']], # , 'meas_mse', 'meas_mkld']
                        'mode': 'stack',
                        'title': 'measures', 'title_pos': 'bottom',
                        'plot': table,
                    },
                    {
                        'input': ['meas_div'], 'plot': partial(timeseries, linestyle = 'none', marker = 'o'),
                        'title': 'histogram divergence %s $h1 - h2$' % (div_meas, ),
                        # 'aspect': 'auto',
                        # 'orientation': 'horizontal',
                        'xlim': None, # 'xticks': False, # 'xticklabels': False,
                        'xaxis': meas_hist_bincenters, 'xlabel': 'bins $k$',
                        # 'yticks': False,
                        # normalize to original range
                        'ylim': None, 'ylabel': 'divergence'
                        # 'legend_loc': 'right',
                    },
                ],
                
            ],
        },
    }),

    # # plotting
    # ('plot', {
    #     'block': PlotBlock2,
    #     'params': {
    #         'id': 'plot',
    #         'blocksize': numsteps,
    #         'saveplot': saveplot,
    #         'savetype': 'pdf',
    #         'wspace': 0.15,
    #         'hspace': 0.15,
    #         'xlim_share': True,
    #         'inputs': {
    #             's_p': {'bus': 'robot1/s_proprio', 'shape': (dim_s_proprio, numsteps)},
    #             's_e': {'bus': 'robot1/s_extero', 'shape': (dim_s_extero, numsteps)},
    #             'pre_l0': {'bus': 'pre_l0/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l1': {'bus': 'pre_l1/pre', 'shape': (dim_s_goal, numsteps)},
    #             'pre_l2': {'bus': 'pre_l2/y', 'shape': (dim_s_proprio, numsteps)},
    #             'credit_l1': {'bus': 'budget/credit', 'shape': (1, numsteps)},
    #             'infodist': {
    #                 'bus': 'infodist/infodist',
    #                 'shape': (dim_s_proprio, 1, 1)
    #             },
    #         },
    #         'desc': 'Single episode pm1d baseline',
            
    #         'subplots': [
    #             # row 1: pre, s
    #             [
    #                 {
    #                     'input': ['pre_l0', 's_p', 'pre_l1'],
    #                     'plot': [
    #                         partial(timeseries, linewidth = 1.0, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, linewidth = 2.0, alpha = 1.0, xticks = False, xlabel = None)],
    #                     'title': 'two-level prediction and measurement (timeseries)',
    #                 },
    #                 {
    #                     'input': ['pre_l0', 's_p', 'pre_l1'],
    #                     'plot': [partial(
    #                         histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                         yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(3)],
    #                     'title': 'two-level prediction and measurement (histogram)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                     # 'mode': 'stack'
    #                 },
    #             ],
                
    #             # row 2: pre_l2, s
    #             [
    #                 {
    #                     'input': ['pre_l2', 's_p'],
    #                     'plot': [
    #                         partial(timeseries, alpha = 1.0, xlabel = None),
    #                         partial(timeseries, alpha = 0.5, xlabel = None),
    #                     ],
    #                     'title': 'proprio and f(proprio)',
    #                 },
    #                 {
    #                     'input': ['pre_l2', 's_p'],
    #                     'plot': [
    #                         partial(
    #                             histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                             yticks = False, xticks = False, alpha = 0.5, normed = False) for _ in range(2)],
    #                     'title': 'proprio and f(proprio) (histogram)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                     # 'mode': 'stack'
    #                 },
    #             ],

    #             # row 3: budget
    #             [
    #                 {'input': 'credit_l1', 'plot': partial(timeseries, ylim = (0, 1000), alpha = 1.0),
    #                      'title': 'agent budget (timeseries)',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                 },
    #                 {'input': 'credit_l1', 'plot': partial(
    #                     histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #                     yticks = False, ylim = (0, 1000), alpha = 1.0, normed = False),
    #                     'title': 'agent budget (histogram)',
    #                     'xlabel': 'count [n]',
    #                     'desc': 'Single episode pm1d baseline \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #                 },
    #             ],
                
    #             # [
    #             #     {
    #             #         'input': ['infodist'],
    #             #         'ndslice': (slice(None), 0, slice(None)),
    #             #         'shape': (dim_s_proprio, 1),
    #             #         'plot': [
    #             #             partial(timeseries, linewidth = 1.0, alpha = 1.0, marker = 'o', xlabel = None),
    #             #         ],
    #             #         'title': 'd(proprio, f(proprio))',
    #             #     },
    #             #     {
    #             #         'input': ['infodist'],
    #             #         'ndslice': (slice(None), 0, slice(None)),
    #             #         'shape': (dim_s_proprio, 1),
    #             #         'plot': [partial(
    #             #             histogram, orientation = 'horizontal', histtype = 'stepfilled',
    #             #             yticks = False, xticks = False, alpha = 1.0, normed = False) for _ in range(1)],
    #             #         'title': 'd(proprio, f(proprio)) (histogram)',
    #             #         'desc': 'infodist \autoref{fig:exper-mem-000-ord-0-baseline-single-episode}',
    #             #         # 'mode': 'stack'
    #             #     },
    #             # ],
    #         ],
    #     },
    # })
])
