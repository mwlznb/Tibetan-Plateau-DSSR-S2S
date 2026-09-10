from __future__ import annotations

import numpy as np


def recenter(residual_members: np.ndarray, axis: int = 0) -> np.ndarray:
    result = residual_members - residual_members.mean(axis=axis, keepdims=True)
    tolerance = max(1e-5, 8*np.finfo(result.dtype).eps*np.max(np.abs(residual_members),initial=1.0))
    if np.max(np.abs(result.mean(axis=axis))) > tolerance:
        raise AssertionError("recentered residual mean exceeds floating-point tolerance")
    return result


def empirical_crps(members: np.ndarray, observation: np.ndarray, axis: int = 0) -> np.ndarray:
    first=np.mean(np.abs(members-np.expand_dims(observation,axis)),axis=axis)
    expanded_a=np.expand_dims(members,axis+1)
    expanded_b=np.expand_dims(members,axis)
    second=0.5*np.mean(np.abs(expanded_a-expanded_b),axis=(axis,axis+1))
    return first-second

