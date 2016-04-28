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
############################         数据预处理           ############################
####################################################################################
def extract(line):
	import time
	try:
		part = line.strip().split(",")
		uid, iid, beh, ict, time = part[0], part[1], part[2], part[4], "-".join(part[5].split(" ")[0].split("-")[1:])+"-"+part[5].split(" ")[1]
		return ((uid, iid, ict), time+","+beh)
	except:
		return ((""), "")

global bss

if __name__ == "__main__":
	conf = (SparkConf()
    	.setMaster("spark://namenode.omnilab.sjtu.edu.cn:7077")
    	.setAppName("Extract")
    	.set("spark.cores.max", "32")
    	.set("spark.driver.memory", "4g")
		.set("spark.executor.memory", "6g"))
	sc = SparkContext(conf = conf)
	lines = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/tianchi_mobile_recommend_train_user.csv', 1)
	counts = lines.map(lambda x : extract(x)) \
				  .filter(lambda x : x[0]!="") \
				  .groupByKey() \
				  .map(lambda x : (" ".join(x[0])+"\t"+" ".join([str(item["time"])+","+item["beh"] for item in sorted([{"time":content.split(",")[0],"beh":content.split(",")[1]} for content in x[1]],key=lambda x:x["time"])])))
	output = counts.saveAsTextFile("./competition_tianchi/uid_iid")
