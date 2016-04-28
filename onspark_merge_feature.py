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
############################           特征合并           ############################
####################################################################################
if __name__ == "__main__":
	import fileinput
	conf = (SparkConf()
    	.setMaster("spark://namenode.omnilab.sjtu.edu.cn:7077")
    	.setAppName("Extract")
    	.set("spark.cores.max", "32")
    	.set("spark.driver.memory", "4g")
		.set("spark.executor.memory", "6g"))
	sc = SparkContext(conf = conf)
	target, etime, subset = "12-19-0", "12-18-23", {}
	# target, etime, subset = "12-18-0", "12-17-23", {}
	# target, etime, subset = "12-17-0", "12-16-23", {}
	# target, etime, subset = "12-16-0", "12-15-23", {}
	# target, etime, subset = "12-15-0", "12-14-23", {}
	# target, etime, subset = "12-14-0", "12-13-23", {}
	# target, etime, subset = "12-13-0", "12-12-23", {}
	# target, etime, subset = "12-12-0", "12-11-23", {}
	# target, etime, subset = "12-11-0", "12-10-23", {}
	# target, etime, subset = "12-10-0", "12-09-23", {}
	# target, etime, subset = "12-09-0", "12-08-23", {}
	# target, etime, subset = "12-08-0", "12-07-23", {}
	# target, etime, subset = "12-07-0", "12-06-23", {}
	# target, etime, subset = "12-06-0", "12-05-23", {}
	# target, etime, subset = "12-05-0", "12-04-23", {}
	# target, etime, subset = "12-04-0", "12-03-23", {}
	# target, etime, subset = "12-03-0", "12-04-23", {}
	# target, etime, subset = "12-02-0", "12-01-23", {}
	# target, etime, subset = "12-01-0", "11-30-23", {}
	for line in fileinput.input("./tianchi_mobile_recommend_train_item.csv"):
		subset[line.strip().split(",")[0]] = True
	feature_user_prod = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/feature/'+target+'/user_prod', 1).map(lambda x : (x.strip().split("\t")[0]+"\t"+x.strip().split("\t")[2],x.strip()))
	feature_user_ict = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/feature/'+target+'/user_ict', 1).map(lambda x : (x.strip().split("\t")[0]+"\t"+x.strip().split("\t")[1],"\t".join(x.strip().split("\t")[2:])))
	feature_prod = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/feature/'+target+'/prod', 1).map(lambda x : (x.strip().split("\t")[0],"\t".join(x.strip().split("\t")[1:])))
	feature_user = sc.textFile('hdfs://namenode.omnilab.sjtu.edu.cn/user/qiangsiwei/competition_tianchi/feature/'+target+'/user', 1).map(lambda x : (x.strip().split("\t")[0],"\t".join(x.strip().split("\t")[1:])))
	counts = feature_user_prod.join(feature_user_ict).map(lambda x:(x[1][0].split("\t")[1],x[1][0]+"\t"+x[1][1]))
	counts = counts.join(feature_prod).map(lambda x:(x[1][0].split("\t")[0],x[1][0]+"\t"+x[1][1]))
	counts = counts.join(feature_user).map(lambda x:x[1][0]+"\t"+x[1][1])
	counts = counts.filter(lambda x : int(x.strip().split("\t")[3:][9])>0 and int(x.strip().split("\t")[3:][17])>=2)
	output = counts.saveAsTextFile("./competition_tianchi/prediction/"+target)
