import os
import re
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('xxx.tsv', srp='\t', escapechar='\\')
print(df.head)
#id , sentiment(0, 1)  review(文本)
#去掉html标签， 移除标点符号， 切词， 去掉停用词， 重组为新的句子

example = BeautifulSoup(df['review'][1000], 'html.parser').get_text()

example_letters = re.sub(r'[a-zA-Z]', '', example)#去标点符号

works = example_letters.lower().split()# 大小写转化

#去掉停用词 stopwork.txt停用词表
stopworks = {}.fromkeys([line.rstrip() for line in open('stopwork.txt')])

works_nostop = [w for w in works if w not in stopwords]


eng_syopworks = set(stopwords)
#写成一个函数
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[a-zA-Z]', '', text)
    works = text.lower().split()
    works = [w for w in works if w not in eng_syopworks]
    return ''.join(works)

#加一行
df['clean_review'] = df.review.apply(clean_text())

#计算词频 前5000个词频高的
vecotizer = CountVectorizer(max_features=5000)
#提取特征
train_data_feayures = vecotizer.fit_transform(df.clean_review).toarray()
print(train_data_feayures.shape)#(xxxx, 5000)

#分训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_text = train_test_split(train_data_feayures, df.sentiment, test_size=0.2, random_state=0)

#画图

import matplotlib.pyplot as plt
import itertools

def plot_matrix(cm, classes,
                title='confusion matrix',
                cmap=plt.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    #plt.show()



#训练分类
LR_model = LogisticRegression()#逻辑回归sigoxxx函数
LR_model.fit(x_train, y_train)
y_pred = LR_model.predict(x_test)

cnf_matrix = confusion_matrix(y_text, y_pred)

print('recall metric in the testing dataset', cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
print('accuracy metric in the testing dataset', (cnf_matrix[1,1]+cnf_matrix[0,0])/(cnf_matrix[0,0]+cnf_matrix[1,1]))

class_names = [0,1]
plt.figure()
plot_matrix(cnf_matrix, classes=class_names, title='confusion matrix')
plt.show()

reciew_part = df['clean_review']
print(reciew_part.shape) #[50000, ]

#import 