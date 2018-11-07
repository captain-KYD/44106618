import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import re
import nltk
import nltk.stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import csv

def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^a-z']", " ", text)
        #进行词干提取
        new_text = ""
        s = nltk.stem.snowball.EnglishStemmer()  # 英文词干提取器
        for word in word_tokenize(text):
            new_text = new_text + " " + s.stem(word)
        # 放回去
        comment_list.append(new_text)
    #print(comment_list)
    return comment_list

def read_data(file):
    train_data = csv.reader(open(file, encoding="utf-8"))
    lines = 0
    for r in train_data:
        lines += 1
    train_data_label = np.zeros([lines - 1, ])
    train_data_content = []
    train_data = csv.reader(open(file, encoding="utf-8"))
    i = 0
    for data in train_data:
        if data[0] == "Label" or data[0] == "SmsId":
            continue
        if data[0] == "ham":
            train_data_label[i] = 0
        if data[0] == "spam":
            train_data_label[i] = 1
        train_data_content.append(data[1])
        i += 1
    #print(train_data_label,train_data_content)
    return train_data_label,train_data_content


# 载入数据
train_y,train_data_content = read_data("C:\\Users\\admin\\Desktop\\垃圾短信分类data\\train.csv")
test_y,test_data_content = read_data("C:\\Users\\admin\\Desktop\\垃圾短信分类data\\test.csv")
train_data_content = clean_text(train_data_content)
test_data_content = clean_text(test_data_content)
#print(train_data_content)
#print(train_y.shape)
#print(len(train_data_content))
# 数据的TF-IDF信息计算
all_comment_list = list(train_data_content) + list(test_data_content)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_x = text_vector.transform(train_data_content)
test_x = text_vector.transform(test_data_content)
train_x = train_x.toarray()
test_x = test_x.toarray()
print(train_x.shape)
print(test_x.shape)
word = text_vector.get_feature_names()  
print(word) 
#print(train_x.shape,test_x.shape,type(train_x))

#训练朴素贝叶斯模型（得到所需后验概率）
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #邮件数目w
    numWords = len(trainMatrix[0])   #邮件长度
    #print(numWords)
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #垃圾邮件概率
    p0Num = np.ones(numWords) 
    p1Num = np.ones(numWords)   
    p0Denom = 2.0; p1Denom = 2.0                        
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   #如果是垃圾邮件
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive
#朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
    
p0V,p1V,pSpam = trainNB0(train_x,train_y)
print(pSpam)
# 预测答案
#print(test_y.shape)
answer = pd.read_csv(open("C:\\Users\\admin\\Desktop\\垃圾短信分类data\\sampleSubmission.csv"))
for i in range(test_x.shape[0]):
    if classifyNB(test_x[i],p0V,p1V,pSpam)==1:
        answer.loc[i,"Label"] = "spam"
    else:
        answer.loc[i,"Label"] = "ham"
answer.to_csv("C:\\Users\\admin\\Desktop\\垃圾短信分类data\\Submission.csv",index=False)  # 不要保存引索列