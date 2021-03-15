# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:47:22 2021

@author: ANIF
"""

from sklearn import svm, datasets #memanggil class sv, dan class datasets dari sklearn
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class svc dan argumen constructor svc 
X, y = datasets.load_iris(return_X_y=True)#mengambil datasets iris dan mengembalikan nilai nilainya 
clf.fit(X, y)#perhitungan nilai label

#Joblib
from joblib import dump, load #memanggil class dump dan library pada joblib
dump(clf, '1184065.joblib')#menyimpan model kedalam 1184065.joblib
hasil = load('1184065.joblib')#memanggil model 1184065
print(hasil.predict(X[0:1]))#menamapilkan model yang dipanggil sebelumnya