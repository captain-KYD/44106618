# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,Imputer


def stocGradAscent0(dataMat,labelMat):
    m, n = np.shape(dataMat)  # m*n: m个样本，n个特征
    alpha = 0.0006  # 学习步长
    maxCycles=500
    weights = np.ones(n)
    for cycle in range(maxCycles):
        for i in range(m):
            y = sigmoid(sum(dataMat[i] * weights) )  # 预测值
            error = labelMat[i] - y
            weights = weights + alpha  * error* dataMat[i]
        #print(type(weights))
    print(weights)
    return weights


"""
函数：sigmoid函数
"""
def sigmoid(z):
    return 1.0/(1+np.exp(-z) )


"""
函数：sigmoid回归分类
"""
def classifyVector(dataIn,weights):
    h=sigmoid(sum(dataIn*weights))
    if h>0.5:
        return 1.0 #大于0.5为1
    else:
        return 0.0 #小于0.5为0

"""
函数：疝气预测
"""
def colicTest():
    trainData=open('data\horseColicTraining.txt')
    testData = open('data\horseColicTest.txt')
    trainSet=[]
    trainLabel=[]
    for line in trainData.readlines():   #训练集
        curLine=line.strip().split('\t')   #按行读取，并且分割文本
        lineArr=[] #初始化列表，用来存储每一行的数据
        #print(curLine)
        for i in range(21):
            lineArr.append(float (curLine[i]))  #特征值的存储
        trainSet.append(lineArr)   #特征值的存储
        trainLabel.append(float(curLine[21]))  #标签值的存储
    #进行数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(trainSet)
    trainSet = scaler.transform(trainSet)
    #print(trainSet)
    #print(lineArr)
    trainWeights=stocGradAscent0(np.array(trainSet),trainLabel)  #随机梯度上升计算回归系数
    
    errorCount=0
    numTestVec=0
    testSet=[]
    for line in testData.readlines():  #测试集
        numTestVec+=1  #计算测量样本个数
        curLine=line.strip().split('\t') #按行读取，并且分割文本
        lineArr=[]   #初始化列表，用来存储每一行的数据
        for i in range(21):
            lineArr.append(float (curLine[i]))  #特征值的存储
        testSet.append(lineArr)
    #print(numTestVec)
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(curLine[21]): 
            errorCount+=1   #将回归系数与特征向量相乘，所得输入到sigmiod函数，并求出错误预测值个数
    #print(numTestVec,errorCount)
    errorRate=float(errorCount/numTestVec)   #计算错误率
    #print(numTestVec)
    print("the error rate is: %f" % errorRate)
    return errorRate   


if __name__ == '__main__':
   colicTest()