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
############################       用户-商品特征          ############################
####################################################################################
def extract(line):
	import time
	import itertools
	(uid, iid, ict) = line.strip().split("\t")[0].split(" ")
	if subset.has_key(iid):
		items = filter(lambda x:x[0]>0, [(int(time.mktime(time.strptime('2014-'+etime,'%Y-%m-%d-%H'))-time.mktime(time.strptime('2014-'+i.split(",")[0],'%Y-%m-%d-%H')))/(24*3600)+1, int(i.split(",")[1])) for i in line.strip().split("\t")[1].split(" ")])
		f, inf = [0]*22, 100
		buy = filter(lambda x:x[1]==4, items)
		last = buy[-1][0] if len(buy)!=0 else inf
		a1 = filter(lambda x:x[0]<last and x[1]==1, items) 
		a2 = filter(lambda x:x[0]<last and x[1]==2, items) 
		a3 = filter(lambda x:x[0]<last and x[1]==3, items)
		# 基本统计特征
		f[0] = a2[-1][0] if len(a2)!=0 else inf # 购买后最后一次加入收藏夹距离天数
		f[1] = a3[-1][0] if len(a3)!=0 else inf # 购买后最后一次加入购物车距离天数
		f[2] = len(a1) # 购买后点击次数
		f[3] = len(a2) # 购买后加收次数
		f[4] = len(a3) # 购买后加购次数
		f[5] = len(filter(lambda x:x[0]<=1, items)) # 最后1天交互次数
		f[6] = len(filter(lambda x:x[0]==2, items)) # 倒数第2天交互次数
		f[7] = len(filter(lambda x:x[0]==3, items)) # 倒数第3天交互次数
		f[8] = len(filter(lambda x:x[0]==4, items)) # 倒数第4天交互次数
		f[9] = len(filter(lambda x:x[0]<=7, items)) # 最后7天交互次数
		f[10] = len(buy) # 历史购买次数
		f[11] = last # 最后一次购买距离天数
		f[12] = len(set([item[0] for item in items if item[0]<=3])) # 最后3天内交互天数
		f[13] = len(set([item[0] for item in items if item[0]<=7])) # 最后1周内交互天数
		f[14] = len(set([item[0] for item in items if item[0]<=21])) # 最后3周内交互天数
		f[15] = items[-1][0] if len(items)!=0 else inf # 最后1次交互距离天数
		inter = [len(list(i)) for _,i in itertools.groupby(items, lambda x: x[0])]
		f[16] = len(inter) #交互天数
		f[17] = max(inter) if len(inter)!=0 else 0 # 交互最多的一天交互次数
		f[18] = len(filter(lambda x:x[1]==1, items)) # 总点击次数
		f[19] = len(filter(lambda x:x[1]==2, items)) # 总添加收藏夹次数
		f[20] = len(filter(lambda x:x[1]==3, items)) # 总加购次数
		f[21] = items[0][0]-items[-1][0]+1 if len(items)!=0 else 0 # 第一次交互到最后一次交互的持续时间
		return (uid+"\t"+iid+"\t"+ict+"\t"+"\t".join([str(i) for i in f]))
	else:
		return ("")

global etime
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
	counts = lines.map(lambda x : extract(x))\
				  .filter(lambda x : x!="")
	output = counts.saveAsTextFile("./competition_tianchi/feature/"+target+"/user_prod/")
