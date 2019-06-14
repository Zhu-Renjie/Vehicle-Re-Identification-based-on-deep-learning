import os 
import os.path 
import shutil 
import time,  datetime
import csv
import random
import glob
def load_info(mode):
    infos = file('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/attribute/model_attr.txt' % (ROOT, mode)).read().splitlines()
    data = {}
    for x in infos:
        x = x.strip().split(' ')#split返回的是一个列表
        data[x[0]] = int(x[1])#model_attr.txt# vehicle ID -> model ID
    
    return data#data字典 vehicle ID：model ID
models = load_info('model')#由model_attr.txt 加载出字典models vehicleID ：modelID

def load_info11():
	global models
	i=0;
    infos = file('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/train_test_split/test_list_800.txt').read().splitlines()
    data = {}
    for x in infos:
    	if i>100:
    		break
        x = x.strip().split(' ')
        i++
        #if x[1]>20000:
        	#continue
        if x[1] not in data:
        	shutil.copy(('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/image/%s.jpg'% x[0]), ('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/image_query/%s_c%s_%s_0.jpg'%(x[1],models[x[1]],x[0])))
        	data[x[1]] = int(x[0])#model_attr.txt# imageName -> VehicleID
        shutil.copy(('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/image/%s.jpg'% x[0]), ('/home/lvyibing/zhurenjie/DJDRRUN/VehicleID/image_test/%s_c%s_%s_0.jpg'%(x[1],models[x[1]],x[0])))
    return data
load_info11()