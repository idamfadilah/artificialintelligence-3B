# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 20:35:16 2021

@author: DyningAida
"""
# import library pandas
import pandas as pd
# load dataset student-mat.csv
d_apel = pd.read_csv('student-mat.csv', sep=';')
# menghitung length dataset csv
len(d_apel)
# generate binary label (pass/fail) berdasar nilai G1+G2+G3, apabila total >= 35, maka bernilai 1, jika tidak maka 0
d_apel['pass'] = d_apel.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) 
										>= 35 else 0, axis=1)
# drop row G1, G2 dan G3
d_apel = d_apel.drop(['G1', 'G2', 'G3'], axis=1)
# menampilkan 5 data teratas
d_apel.head()
# use one-hot encoding on categorical columns
d_apel = pd.get_dummies(d_apel, columns=['sex', 'school', 'address', 
								'famsize',
								'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
								'famsup', 'paid', 'activities','nursery', 'higher', 'internet',
								'romantic'])
# menampilkan 5 data teratas
d_apel.head()
# shuffle rows
d_jeruk = d_apel.sample(frac=1)
# split training and testing data
d_jeruk_train = d_jeruk[:250]
d_jeruk_test = d_jeruk[250:]
# train atribut drop row pass
d_jeruk_train_att = d_jeruk_train.drop(['pass'], axis=1)
# train label menggunakan row pass
d_jeruk_train_pass = d_jeruk_train['pass']
# test atribut drop row pass
d_jeruk_test_att = d_jeruk_test.drop(['pass'], axis=1)
# test label menggunakan row pass
d_jeruk_test_pass = d_jeruk_test['pass']
# atribut drop row pass
d_jeruk_att = d_jeruk.drop(['pass'], axis=1)
# menggunakan row pass untuk label
d_jeruk_pass = d_jeruk['pass']

# import library
import numpy as np
# print number of passing students in whole dataset:
print("Passing: %d out of %d (%.2f%%)" % (np.sum(d_jeruk_pass), len(d_jeruk_pass), 
	       100*float(np.sum(d_jeruk_pass)) / len(d_jeruk_pass)))

# import library
from sklearn import tree
# instansiasi desicion tree classifier
melon = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
# fit decision tree
melon = melon.fit(d_jeruk_train_att, d_jeruk_train_pass)
# import library graphviz untuk visualisasi
import graphviz
# instansiasi graphviz dari fit decision tree sebelumnya
mangga = tree.export_graphviz(melon, out_file=None, label="all", 
									impurity=False, proportion=True,
	                                feature_names=list(d_jeruk_train_att), 
									class_names=["fail", "pass"], 
	                                filled=True, rounded=True)
# buat variabel grafik visualisasi tree
graph = graphviz.Source(mangga)
# jalankan visualisasinya
graph
# save tree
tree.export_graphviz(melon, out_file="student-performance.dot", 
						 label="all", impurity=False, 
						 proportion=True,
	                     feature_names=list(d_jeruk_train_att), 
	                     class_names=["fail", "pass"], 
	                     filled=True, rounded=True)
# cek score
melon.score(d_jeruk_test_att, d_jeruk_test_pass)
# import cross val score untuk cek cross validation score
from sklearn.model_selection import cross_val_score
# cek cross validation score
nangka_score = cross_val_score(melon, d_jeruk_att, d_jeruk_pass, cv=5)
# show average score and +/- two standard deviations away 
#(covering 95% of scores)
# print akurasi
print("Accuracy: %0.2f (+/- %0.2f)" % (nangka_score.mean(), nangka_score.std() * 2))
# buat akurasi max depth
for max_depth in range(1, 20):
	    melon = tree.DecisionTreeClassifier(criterion="entropy", 
			max_depth=max_depth)
	    nangka_score = cross_val_score(melon, d_jeruk_att, d_jeruk_pass, cv=5)
	    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % 
				(max_depth, nangka_score.mean(), nangka_score.std() * 2)
			 )
# buat akurasi depth acc
depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
    melon = tree.DecisionTreeClassifier(criterion="entropy", 
			max_depth=max_depth)
    nangka_score = cross_val_score(melon, d_jeruk_att, d_jeruk_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = nangka_score.mean()
    depth_acc[i,2] = nangka_score.std() * 2
    i += 1
# jalankan dept_acc
depth_acc
# import library matplotlib
import matplotlib.pyplot as plt
# buat plot untuk visualisasi
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()