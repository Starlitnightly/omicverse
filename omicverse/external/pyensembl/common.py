# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle

from functools import wraps


def dump_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        # use lower protocol for compatibility between Python 2 and Python 3
        pickle.dump(obj, file=f, protocol=2)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def _memoize_cache_key(args, kwargs):
    """Turn args tuple and kwargs dictionary into a hashable key.

    Expects that all arguments to a memoized function are either hashable
    or can be uniquely identified from type(arg) and repr(arg).
    """
    cache_key_list = []

    # hack to get around the unhashability of lists,
    # add a special case to convert them to tuples
    for arg in args:
        if type(arg) is list:
            cache_key_list.append(tuple(arg))
        else:
            cache_key_list.append(arg)
    for k, v in sorted(kwargs.items()):
        if type(v) is list:
            cache_key_list.append((k, tuple(v)))
        else:
            cache_key_list.append((k, v))
    return tuple(cache_key_list)


def memoize(fn):
    """Simple reset-able memoization decorator for functions and methods,
    assumes that all arguments to the function can be hashed and
    compared.
    """
    cache = {}

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        cache_key = _memoize_cache_key(args, kwargs)
        try:
            return cache[cache_key]
        except KeyError:
            value = fn(*args, **kwargs)
            cache[cache_key] = value
            return value

    def clear_cache():
        cache.clear()

    # Needed to ensure that EnsemblRelease.clear_cache
    # is able to clear memoized values from each of its methods
    wrapped_fn.clear_cache = clear_cache
    # expose the cache so we can check if an item has already been computed
    wrapped_fn.cache = cache
    # if we want to check whether an item is in the cache, first need
    # to construct the same cache key as used by wrapped_fn
    wrapped_fn.make_cache_key = _memoize_cache_key
    return wrapped_fn
