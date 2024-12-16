r"""
Pyomic (A omic framework for multi-omic analysis)
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

from . import bulk,single,utils,bulk2single,pp,space,pl,externel
#usually
from .utils._data import read
from .utils._plot import palette,ov_plot_set,plot_set
from datetime import datetime,timedelta

name = "omicverse"
__version__ = version(name)
omics="""
   ____            _     _    __                  
  / __ \____ ___  (_)___| |  / /__  _____________ 
 / / / / __ `__ \/ / ___/ | / / _ \/ ___/ ___/ _ \ 
/ /_/ / / / / / / / /__ | |/ /  __/ /  (__  )  __/ 
\____/_/ /_/ /_/_/\___/ |___/\___/_/  /____/\___/                                              
"""
days_christmas="""
      .
   __/ \__
   \     /
   /.'o'.\
    .o.'.         Merry Christmas!
   .'.'o'.
  o'.o.'.o.
 .'.o.'.'.o.       ____ 
.o.'.o.'.o.'.     / __ \____ ___  (_)___| |  / /__  _____________ 
   [_____]       / / / / __ `__ \/ / ___/ | / / _ \/ ___/ ___/ _ \ 
    \___/       / /_/ / / / / / / / /__ | |/ /  __/ /  (__  )  __/ 
                \____/_/ /_/ /_/_/\___/ |___/\___/_/  /____/\___/
"""
#Tua Xiong
days_chinese_new_year="""
                                        ,   ,
                                        $,  $,     ,
                                        "ss.$ss. .s'
                                ,     .ss$$$$$$$$$$s,
                                $. s$$$$$$$$$$$$$$`$$Ss
                                "$$$$$$$$$$$$$$$$$$o$$$       ,
                               s$$$$$$$$$$$$$$$$$$$$$$$$s,  ,s
                              s$$$$$$$$$"$$$$$$ssss$$$$$$"$$$$$,
                              s$$$$$$$$$$sss$$$$ssssss"$$$$$$$$"
                             s$$$$$$$$$$'         `\"\"\"ss"$"$s\"\"
                             s$$$$$$$$$$,              `\"\"\"\"\"$  .s$$s
                             s$$$$$$$$$$$$s,...               `s$$'  `
                         `ssss$$$$$$$$$$$$$$$$$$$$####s.     .$$"$.   , s-
                           `""\""$$$$$$$$$$$$$$$$$$$$#####$$$$$$"     $.$'
                                 "$$$$$$$$$$$$$$$$$$$$$####s""     .$$$|
                                  "$$$$$$$$$$$$$$$$$$$$$$$$##s    .$$" $
                                   $$""$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"   `
                                  $$"  "$"$$$$$$$$$$$$$$$$$$$$S""\""'
                             ,   ,"     '  $$$$$$$$$$$$$$$$####s
                             $.          .s$$$$$$$$$$$$$$$$$####"
                 ,           "$s.   ..ssS$$$$$$$$$$$$$$$$$$$####"
                 $           .$$$S$$$$$$$$$$$$$$$$$$$$$$$$#####"
                 Ss     ..sS$$$$$$$$$$$$$$$$$$$$$$$$$$$######""
                  "$$sS$$$$$$$$$$$$$$$$$$$$$$$$$$$########"
           ,      s$$$$$$$$$$$$$$$$$$$$$$$$#########""'
           $    s$$$$$$$$$$$$$$$$$$$$$#######""'      s'         ,
           $$..$$$$$$$$$$$$$$$$$$######"'       ....,$$....    ,$
            "$$$$$$$$$$$$$$$######"' ,     .sS$$$$$$$$$$$$$$$$s$$
              $$$$$$$$$$$$#####"     $, .s$$$$$$$$$$$$$$$$$$$$$$$$s.
   )          $$$$$$$$$$$#####'      `$$$$$$$$$###########$$$$$$$$$$$.
  ((          $$$$$$$$$$$#####       $$$$$$$$###"       "####$$$$$$$$$$
  ) \         $$$$$$$$$$$$####.     $$$$$$###"             "###$$$$$$$$$   s'
 (   )        $$$$$$$$$$$$$####.   $$$$$###"                ####$$$$$$$$s$$'
 )  ( (       $$"$$$$$$$$$$$#####.$$$$$###' -OmicVerse     .###$$$$$$$$$$"
 (  )  )   _,$"   $$$$$$$$$$$$######.$$##'                .###$$$$$$$$$$
 ) (  ( \.         "$$$$$$$$$$$$$#######,,,.          ..####$$$$$$$$$$$"
(   )$ )  )        ,$$$$$$$$$$$$$$$$$$####################$$$$$$$$$$$"
(   ($$  ( \     _sS"  `"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$S$$,
 )  )$$$s ) )  .      .   `$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"'  `$$
  (   $$$Ss/  .$,    .$,,s$$$$$$##S$$$$$$$$$$$$$$$$$$$$$$$$S""        '
    \)_$$$$$$$$$$$$$$$$$$$$$$$##"  $$        `$$.        `$$.
        `"S$$$$$$$$$$$$$$$$$#"      $          `$          `$
            `\"""\""\""\""\""\""'         '           '           '
"""

spring_festival = { 
    2022: datetime(2022, 2, 1), 
    2023: datetime(2023, 1, 22), 
    2024: datetime(2024, 2, 10), # ... 
    2025: datetime(2025, 1, 29),
    2026: datetime(2026, 2, 17),
    2027: datetime(2027, 2, 6),
}

today = datetime.now()
chinese_new_year = spring_festival[today.year]
if today.month == 12 and (today.day == 25 or today.day == 24):
    # december 12.25 or 12.24 (christmas)
    print(days_christmas)
elif (today.year in spring_festival) and (chinese_new_year - timedelta(days=1) <= today <= chinese_new_year + timedelta(days=3)):
    print(days_chinese_new_year)
else:
    print(omics)

print(f'Version: {__version__}, Tutorials: https://omicverse.readthedocs.io/')

from ._settings import settings
