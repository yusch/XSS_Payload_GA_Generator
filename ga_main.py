import os
import sys
import random
import codecs
import configparser
import numpy as np
import pandas as pd
from util import Utilty
import torch
import torch.optim as optim

# Type of printing.
OK = 'ok'         # [*]
NOTE = 'note'     # [+]
FAIL = 'fail'     # [-]
WARNING = 'warn'  # [!]
NONE = 'none'     # No label.
