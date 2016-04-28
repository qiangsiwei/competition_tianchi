# -*- coding: utf-8 -*- 

import sys
from operator import add
from pyspark import SparkConf
from pyspark import SparkContext
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ####################################################################################
# ############################             预测            ############################
# ####################################################################################
def prediction(line):
	feature = [float(line.strip().split("\t")[3:][i]) for i in xrange(len(line.strip().split("\t")[3:])) if i in [1,2,4,5,9,12,15,17,18,20,23,27,48,50,62,66]]
	return (sum([clfs[i].predict_proba(feature)[0][0] for i in xrange(N)]), "\t".join(line.strip().split("\t")[:2]))

global N
global clfs

if __name__ == "__main__":
	N = 3
	import time
	import random
	import pickle
	import fileinput
	time_start = time.time()
	conf = (SparkConf()
    	.setMaster("spark://namenode.omnilab.sjtu.edu.cn:7077")
    	.setAppName("Extract")
    	.set("spark.cores.max", "32")
    	.set("spark.driver.memory", "4g")
		.set("spark.executor.memory", "6g"))
	sc = SparkContext(conf = conf)
	featureset = [line.strip() for line in fileinput.input("feature_list.txt")]
	buyset = {}
	for line in sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/buy', 1).collect():
		buyset[line.strip()] = True
	X_train_0, Y_train_0, X_train_1, Y_train_1 = [], [], [], []
	dates = ["12-15-0","12-16-0"]
	for date in dates:
		for line in sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/prediction/'+date, 1).collect():
		# for line in fileinput.input("prediction/"+date+".txt"):
			if buyset.has_key(date+"\t"+"\t".join(line.strip().split("\t")[:2])):
				# X_train_1.append([float(i) for i in line.strip().split("\t")[3:]])
				X_train_1.append([float(line.strip().split("\t")[3:][i]) for i in xrange(len(line.strip().split("\t")[3:])) if i in [1,2,4,5,9,12,15,17,18,20,23,27,48,50,62,66]])
				Y_train_1.append(1)
			else:
				# X_train_0.append([float(i) for i in line.strip().split("\t")[3:]])
				X_train_0.append([float(line.strip().split("\t")[3:][i]) for i in xrange(len(line.strip().split("\t")[3:])) if i in [1,2,4,5,9,12,15,17,18,20,23,27,48,50,62,66]])
				Y_train_0.append(0)	
	X_train, Y_train = [[] for i in xrange(N)], [[] for i in xrange(N)]
	for i in xrange(N):
		for j in random.sample([j for j in xrange(len(X_train_0))], 10*len(X_train_1)):
			X_train[i].append(X_train_0[j])
			Y_train[i].append(Y_train_0[j])
		X_train[i].extend(X_train_1)
		Y_train[i].extend(Y_train_1)
	print X_train[0][0]
	print Y_train[0][0]
	print len(Y_train[0])
	# clfs = [RandomForestClassifier(n_estimators=20,max_features=10,max_depth=3,random_state=1,class_weight={0:1,1:1}) for i in xrange(N)]
	clfs = [GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1) for i in xrange(N)]
	# criterion = "gini"|"entropy"
	# clf = Pipeline([
	#   ('feature_selection', LinearSVC(penalty="l1", dual=False)),
	#   # ('classification', RandomForestClassifier(n_estimators=20,max_features=30,max_depth=3,random_state=1,class_weight={1:2}))
	#    ('classification', RandomForestClassifier(n_estimators=100,max_features=30,max_depth=3,random_state=1,class_weight={0:5}))
	#   # ('classification', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1))
	#   # ('classification', SVC(kernel='rbf', probability=True, class_weight={1:10}))
	# ])
	for i in xrange(N):
		clfs[i].fit(X_train[i],Y_train[i])
	# for i in xrange(len(clf.feature_importances_)):
	# 	if clf.feature_importances_[i] >= 0.01:
	# 		print i, clf.feature_importances_[i], featureset[i]
	############################ 备份模型 ############################ 
	# pickle.dump(clf,open('clf.p', 'wb'))
	# clf = pickle.load(open("clf.p","rb"))
	############################ 备份模型 ############################ 
	print "training finished!~"
	pred = "12-17-0"
	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/prediction/'+pred, 1)
	counts = lines.map(lambda x : prediction(x)).sortByKey().collect()
	recall, precision, correct = len([1 for key in buyset.keys() if key.strip().split("\t")[0] == pred]), 400, len([1 for item in counts[:400] if buyset.has_key(pred+"\t"+item[1])])
	print "prediction finished!~"
	print recall, precision, correct, 1.0*correct/recall, 1.0*correct/precision, 2*1.0*correct/recall*1.0*correct/precision/(1.0*correct/recall+1.0*correct/precision)
	with open("tianchi_mobile_recommendation_predict.csv","w") as f:
		f.write("user_id,item_id\n")
		for item in counts[:400]:
			f.write(",".join(item[1].split("\t"))+"\n")
	time_end = time.time()
	print time_end-time_start
