from collections import defaultdict
from typing import Dict, Type, Callable, List, Tuple

def boundary(iterable: List[Tuple], num=1):
    return iterable[:num] + iterable[-num:]

def top(iterable: List[Tuple], num=1):
    return iterable[:num]

def bottom(iterable: List[Tuple], num=1):
    return iterable[:1] + iterable[-num:]

def top_bottom(iterable: List[Tuple], n1=1, n2=1):
    return iterable[:n1] + iterable[-n2:]

def strided(iterable: List[Tuple], d=3):
    ret = [iterable[0]]
    for i in range(len(iterable)):
        if i % d == 0:
            ret.append(iterable[i])
    ret.append(iterable[-1])
    return ret

def identity(iterable: List[Tuple], **kwargs):
    return iterable


filter_function_map = defaultdict(lambda: identity)
filter_function_map["boundary"] = boundary
filter_function_map["top"] = top
filter_function_map["bottom"] = bottom
filter_function_map["top_bottom"] = top_bottom
filter_function_map["strided"] = strided
