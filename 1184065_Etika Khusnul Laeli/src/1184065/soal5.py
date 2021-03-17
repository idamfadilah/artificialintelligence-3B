# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:21:04 2021

@author: ANIF
"""

import numpy as np#memanggil library numpy dan dibuat alias np
from sklearn import random_projection#memanggil class random_projection pada library sklearn

rng = np.random.RandomState(0)#membuat variable rng, dan mendefinisikan np, fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000)#membuat variabel X, dan menentukan nilai random dari 10-2000
X = np.array(X, dtype='float32')#menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan type datanya sebagai float32
X.dtype#Mengubah data tipe menjadi float32

transformer = random_projection.GaussianRandomProjection()#membuat variabel transformer dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X)#membuat variabel baru dan melakukan perhitungan label pada variabel X
X_new.dtype#mengubah data tipe menjadi float64
print(X_new)#menampilkan isi variabel X_new