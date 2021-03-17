# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 18:42:04 2021

@author: ANIF
"""

from sklearn import datasets
digits = datasets.load_digits()
print(digits.data)
print(digits.target)