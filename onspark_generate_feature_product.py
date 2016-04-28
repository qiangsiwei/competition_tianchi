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
############################           商品特征           ############################
####################################################################################
def extract1(line):
	import time
	(uid, iid, ict) = line.strip().split("\t")[0].split(" ")
	if subset.has_key(iid):
		items = filter(lambda x:x[0]>0, [(int(time.mktime(time.strptime('2014-'+etime,'%Y-%m-%d-%H'))-time.mktime(time.strptime('2014-'+i.split(",")[0],'%Y-%m-%d-%H')))/(24*3600)+1, int(i.split(",")[1])) for i in line.strip().split("\t")[1].split(" ")])
		return (iid,items)
	else:
		return ("","")

def extract2(items_list):
	items, items_buy, items_buy_3, f, inf = [], [], [], [0]*37, 100
	f[30] = len(items_list) # 交互人数
	for i in items_list:
		if len(filter(lambda x:x[1]==4,i))>0:
			items_buy.append(i)
		if len(filter(lambda x:x[1]==4 and x[0]<=3,i))>0:
			items_buy_3.append(i)	
		items.extend(i)
	f[31] = len(items_buy) # 购买人数
	f[32] = len(items_buy_3) # 三天内购买人数
	f[33] = len(filter(lambda x:len(x)==1,items_list)) # 只有过一次交互的用户数
	f[34] = len(filter(lambda x:len(x)==2,items_list)) # 有过两次交互的用户数
	f[35] = len(filter(lambda x:len(x)==3,items_list)) # 有过三次交互的用户数
	items = sorted(items, key=lambda x:x[0], reverse=True)
	f[0] = len(filter(lambda x:x[0]<=1 and x[1]==1, items)) # 最后1天点击次数
	f[1] = len(filter(lambda x:x[0]<=1 and x[1]==2, items)) # 最后1天加收次数
	f[2] = len(filter(lambda x:x[0]<=1 and x[1]==3, items)) # 最后1天加购次数
	f[3] = len(filter(lambda x:x[0]<=1 and x[1]==4, items)) # 最后1天购买次数
	f[4] = len(filter(lambda x:x[0]==2 and x[1]==1, items)) # 倒数第2天点击次数
	f[5] = len(filter(lambda x:x[0]==2 and x[1]==2, items)) # 倒数第2天加收次数
	f[6] = len(filter(lambda x:x[0]==2 and x[1]==3, items)) # 倒数第2天加购次数
	f[7] = len(filter(lambda x:x[0]==2 and x[1]==4, items)) # 倒数第2天购买次数
	f[8] = len(filter(lambda x:x[0]==3 and x[1]==1, items)) # 倒数第3天点击次数
	f[9] = len(filter(lambda x:x[0]==3 and x[1]==2, items)) # 倒数第3天加收次数
	f[10] = len(filter(lambda x:x[0]==3 and x[1]==3, items)) # 倒数第3天加购次数
	f[11] = len(filter(lambda x:x[0]==3 and x[1]==4, items)) # 倒数第3天购买次数
	f[12] = len(filter(lambda x:x[0]<=7 and x[1]==1, items)) # 最后1周点击次数
	f[13] = len(filter(lambda x:x[0]<=7 and x[1]==2, items)) # 最后1周加收次数
	f[14] = len(filter(lambda x:x[0]<=7 and x[1]==3, items)) # 最后1周加购次数
	f[15] = len(filter(lambda x:x[0]<=7 and x[1]==4, items)) # 最后1周购买次数
	f[16] = len(filter(lambda x:x[0]<=14 and x[1]==1, items)) # 最后2周点击次数
	f[17] = len(filter(lambda x:x[0]<=14 and x[1]==2, items)) # 最后2周加收次数
	f[18] = len(filter(lambda x:x[0]<=14 and x[1]==3, items)) # 最后2周加购次数
	f[19] = len(filter(lambda x:x[0]<=14 and x[1]==4, items)) # 最后2周购买次数
	f[20] = min(1.0,round(1.0*(f[3]+f[7]+f[11])/(f[0]+f[4]+f[8]),4)) if (f[0]+f[4]+f[8]) else 0.0 # 最后3天点击转化率
	f[21] = min(1.0,round(1.0*(f[3]+f[7]+f[11])/(f[1]+f[5]+f[9]),4)) if (f[1]+f[5]+f[9])!=0 else 0.0 # 最后3天加收转化率
	f[22] = min(1.0,round(1.0*(f[3]+f[7]+f[11])/(f[2]+f[6]+f[10]),4)) if f[2]!=0 else 0.0 # 最后3天加购转化率
	f[23] = min(1.0,round(1.0*f[7]/f[4],4)) if f[4]!=0 else 0.0 # 最后2周点击转化率
	f[24] = min(1.0,round(1.0*f[7]/f[5],4)) if f[5]!=0 else 0.0 # 最后2周加收转化率
	f[25] = min(1.0,round(1.0*f[7]/f[6],4)) if f[6]!=0 else 0.0 # 最后2周加购转化率
	buy = filter(lambda x:x[1]==4, items)
	last = buy[-1][0] if len(buy)!=0 else inf
	f[26] = len(buy) # 购买总次数 
	f[27] = len(filter(lambda x:x[0]==last and x[1]==1, items)) # 最后一次发生购买的当天发生的点击次数
	f[28] = len(filter(lambda x:x[0]==last and x[1]==2, items)) # 最后一次发生购买的当天发生的加收次数
	f[29] = len(filter(lambda x:x[0]==last and x[1]==3, items)) # 最后一次发生购买的当天发生的加购次数
	f[36] = round(1.0*len(items)/f[30],4) if f[30]!=0 else 0.0 # 人均交互次数
	return "\t".join([str(i) for i in f])
	
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
		subset[line.split(",")[0]] = True
	counts = lines.map(lambda x : extract1(x))\
				  .filter(lambda x : x[0]!="")\
				  .groupByKey()\
				  .map(lambda x : x[0]+"\t"+extract2(x[1]))
	output = counts.saveAsTextFile("./competition_tianchi/feature/"+target+"/prod/")
