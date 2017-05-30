"""smp_graphs perform windowed short time mutual info scan"""

from smp_graphs.block_meas_infth import MIMVBlock2
from smp_graphs.block import SliceBlock2
from smp_graphs.block_plot import ImgPlotBlock2

ppycnf = {
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170518_161544_puppy_process_logfiles_pd.h5',
    # 'numsteps': 27000,
    # 'logfile': 'data/experiment_20170526_160018_puppy_process_logfiles_pd.h5',
    # 'numsteps': 147000,
    # 'logfile': 'data/experiment_20170510_155432_puppy_process_logfiles_pd.h5', # 147K
    # 'numsteps': 29000,
    # 'logfile': 'data/experiment_20170517_160523_puppy_process_logfiles_pd.h5', 29K
    # 'numsteps': 10000,
    # 'logfile': 'data/experiment_20170511_145725_puppy_process_logfiles_pd.h5', # 10K
    # 'numsteps': 2000,
    # 'logfile': 'data/experiment_20170510_173800_puppy_process_logfiles_pd.h5', 2K
    'numsteps': 20000,
    'logfile': 'data/experiment_20170530_174612_process_logfiles_pd.h5', # step fast-to-slow newB all concatenated
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'logtype': 'selflog',
}

ppycnf2 = {
    # 'logfile': 'data/stepPickles/step_period_4_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_10_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_12_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_76_0.pickle',
    # 'logfile': 'data/stepPickles/step_period_26_0.pickle',
    'logfile': 'data/sin_sweep_0-6.4Hz_newB.pickle', # continuous sweep
    'logtype': 'puppy',
    'xdim': 6,
    'xdim_eff': 3,
    'ydim': 4,
    'numsteps': 5000,
}

cnf = ppycnf

# copy params to namespace
numsteps = cnf['numsteps']
xdim = cnf['xdim']
ydim = cnf['ydim']
xdim_eff = cnf['xdim_eff']
if cnf.has_key('sys_slicespec'):
    sys_slicespec = cnf['sys_slicespec']
else:
    sys_slicespec = {'x': {'acc': slice(0, 3), 'gyr': slice(3, xdim)}}

scanstart = -30
scanstop = 0
scanlen = scanstop - scanstart

# 1000/1000
winsize = 1000
overlap = 1000
srcsize = overlap

graph = OrderedDict([
    ('data', {
        'block': FileBlock2,
        'params': {
            'id': 'data',
            'debug': False,
            # 'blocksize': overlap, # numsteps,
            'blocksize': srcsize, # numsteps,
            'type': cnf['logtype'],
            'file': {'filename': cnf['logfile']},
            'outputs': {'log': None, 'x': {'shape': (xdim, srcsize)},
                            'y': {'shape': (ydim, srcsize)}},
        },
    }),

    # slice block to split puppy sensors x into gyros x_gyr and accels x_acc
    ('dataslice', {
        'block': SliceBlock2,
        'params': {
            'id': 'dataslice',
            # 'blocksize': overlap,
            'blocksize': srcsize,
            'debug': True,
            # puppy sensors
            'inputs': {'x': {'bus': 'data/x', 'shape': (xdim, srcsize)}},
            'slices': sys_slicespec,
            # 'slices': ,
            }
        }),
        
    # multivariate mutual information analysis of data I(X^n ; Y^m)
    ('mimvl', {
        'block': LoopBlock2,
        'params': {
            'id': 'mimvl',
            'blocksize': overlap,
            'loop': [('inputs', {'x': {'bus': 'dataslice/x_acc',
                                           'shape': (xdim_eff, winsize)},
                                 'y': {'bus': 'data/y',
                                           'shape': (ydim, winsize)}}),
                          # ('inputs', {'x': {'bus': 'dataslice/x_gyr'}, 'y': {'bus': 'data/y'}}),
                     # ('inputs', {'x': {'bus': 'data/x'}, 'y': {'bus': 'data/r'}}),
                     # ('inputs', {'x': {'bus': 'data/y'}, 'y': {'bus': 'data/r'}}),
            ],
            'loopblock': {
                'block': MIMVBlock2,
                'params': {
                    'id': 'mimv',
                    'blocksize': overlap,
                    'debug': False,
                    'inputs': {'x': {'bus': 'dataslice/x_gyr',
                                         'shape': (xdim_eff, winsize)},
                                   'y': {'bus': 'data/y',
                                             'shape': (ydim, winsize)}},
                    # 'shift': (-120, 8),
                    'shift': (scanstart, scanstop), # len 21
                    # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
                    'outputs': {'mimv': {'shape': (1, scanlen)}}
                }
            },
        }
    }),

    ('mimv', {
        'block': MIMVBlock2,
        'params': {
            'id': 'mimv',
            'blocksize': overlap,
            'debug': False,
            'inputs': {'x': {'bus': 'dataslice/x_gyr', 'shape': (xdim_eff, winsize)},
                           'y': {'bus': 'data/y', 'shape': (ydim, winsize)}},
            # 'shift': (-120, 8),
            'shift': (scanstart, scanstop), # len 21
            # 'outputs': {'mi': {'shape': ((ydim + xdim)**2, 1)}}
            'outputs': {'mimv': {'shape': (1, scanlen)}}
        }
    }),
    
    # plot multivariate (global) mutual information over timeshifts
    ('plot_infth', {
        'block': ImgPlotBlock2,
        'params': {
            'id': 'plot_infth',
            'logging': False,
            'saveplot': saveplot,
            'debug': False,
            'wspace': 0.5,
            'hspace': 0.5,
            'blocksize': overlap, # numsteps,
            'inputs': {
                'd1': {'bus': 'mimv/mimv', 'shape': (1, scanlen * numsteps/overlap)},
                'd2': {'bus': 'mimvl_0/mimv', 'shape': (1, scanlen * numsteps/overlap)},
                't': {'val': np.linspace(scanstart, scanstop-1, scanlen)},},
            'outputs': {}, #'x': {'shape': (3, 1)}},
            'subplots': [
                [
                    {'input': 'd1', 
                         'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/overlap, scanlen)},
                    
                    {'input': 'd2', 
                    'cmap': 'Reds',
                    'title': 'Multivariate mutual information I(X;Y) for time shifts [0, ..., 20]',
                    'ndslice': (slice(None), slice(None)),
                    # 'dimstack': {'x': [0], 'y': [1]},
                    'shape': (numsteps/overlap, scanlen)},
                    # {
                    # 'input': 'd3',
                    # 'ndslice': (slice(None), slice(None), slice(None)),
                    # 'dimstack': {'x': [2, 1], 'y': [0]},
                    # 'shape': (scanlen, ydim, xdim),
                    # 'cmap': 'Reds'},
                ],
            ]
        },
    }),
    
    
])
