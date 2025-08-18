import pandas as pd

from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple

Triple = Tuple[int, int, int]
Triples = List[Triple]
# Named Tuple for DF SPlit
SplitTuple = namedtuple("SplitTuple", ["train", "dev", "test"])

@dataclass
class DFSplit:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame