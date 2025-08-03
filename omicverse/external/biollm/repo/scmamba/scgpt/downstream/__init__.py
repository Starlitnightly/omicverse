# -*- coding = utf-8 -*-
# Author:jiangwenjian
# Email: jiangwenjian@genomics.cn; aryn1927@gmail.com
# @File:__init__.py
# @Software:PyCharm
# @Created Time:2024/2/26 6:03 PM
import logging
import sys

logger = logging.getLogger("downstream")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

from .finetune_utils import *