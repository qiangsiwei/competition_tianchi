# -*- coding: utf-8 -*- 

import sys
from operator import add
from pyspark import SparkConf
from pyspark import SparkContext
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

####################################################################################
############################          验证集提取          ############################
####################################################################################
def extract1(line):
	import time
	(uid, iid, ict), items = line.strip().split("\t")[0].split(" "), filter(lambda x:x[1]==4, [(i.split(",")[0], int(i.split(",")[1])) for i in line.strip().split("\t")[1].split(" ")])
	return (uid+"\t"+iid, ["-".join(item[0].split("-")[:2])+"-0" for item in items])

global subset

if __name__ == "__main__":
	import fileinput
	conf = (SparkConf()
    	.setMaster("spark://namenode.omnilab.sjtu.edu.cn:7077")
    	.setAppName("Extract")
    	.set("spark.cores.max", "32")
    	.set("spark.driver.memory", "4g")
		.set("spark.executor.memory", "6g"))
	sc = SparkContext(conf = conf)
	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/uid_iid', 1)
	subset = {} 	
	for line in fileinput.input("./tianchi_mobile_recommend_train_item.csv"):
		subset[line.strip().split(",")[0]] = True
	def f(x): return x
	counts = lines.filter(lambda x : subset.has_key(x.strip().split("\t")[0].split(" ")[1]))\
				  .map(lambda x : extract1(x))\
				  .flatMapValues(f)\
				  .map(lambda x : x[1]+"\t"+x[0])
	output = counts.saveAsTextFile("./competition_tianchi/buy")
