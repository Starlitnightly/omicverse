from __future__ import annotations

from typing import Union

import numpy as np


AnyRandom = Union[int, np.random.RandomState, None]


def str2list(value: Union[str, list]) -> list:
    return [value] if isinstance(value, str) else list(value)


def spatial_connectivity_key(key: str | None = None) -> str:
    return "spatial_connectivities" if key is None else key


def spatial_distance_key(key: str | None = None) -> str:
    return "spatial_distances" if key is None else key


def sample_obs_key(key: str | None = None) -> str:
    return "sample" if key is None else key
