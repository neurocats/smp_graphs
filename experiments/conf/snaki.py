
from collections import OrderedDict

from smp_graphs.block_snaki import SnakiBlock2

randseed = 0
numsteps = 1000

graph = OrderedDict([
    ('snaki', {
        'block': SnakiBlock2,
    }),
])
