# __init__ file
import warnings
warnings.filterwarnings('ignore')

from . import preprocessing as pp
from . import tools as tl
from . import plotting as pl
from . import utilities as ul

#import sys
#sys.modules.update({f'{__name__}.{m}': globals()[m] for m in ['pp','tl','pl', 'ul']})