# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:46:27 2022

@author: Sagun Shakya
"""
from functools import reduce

class DotDict(dict):
    
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            return super().__getattr__(k)
        if isinstance(v, dict):
            return DotDict(v)
        return v

    def __getitem__(self, k):
        if isinstance(k, str) and '.' in k:
            k = k.split('.')
        if isinstance(k, (list, tuple)):
            return reduce(lambda d, kk: d[kk], k, self)
        return super().__getitem__(k)

    def get(self, k, default=None):
        if isinstance(k, str) and '.' in k:
            try:
                return self[k]
            except KeyError:
                return default
        return super().get(k, default=default)
