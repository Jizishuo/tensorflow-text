'''准确率难以提高，数据影响大,不需要调参'''

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report#计算召回率

def naviebayes():
    '''朴树贝叶斯进行文本分类'''
    news = fetch_20newsgroups(subset="all")

    #进行数据分隔
    x_train, x_text, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    print(news.data)

    #对数据集进行特征抽取
    tf = TfidfVectorizer()
    #以训练集当中的词列表进行每篇重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.fit_transform(x_text)

    #进行朴树贝叶斯算法
    mlt = MultinomialNB(alpha=1.0)#平滑
    mlt.fit(x_train, y_train)
    #预测
    y_predict = mlt.predict(x_text)
    #准确率
    print(mlt.score(x_text, y_test))

    #计算召回率，每一个特征一个精确率召回率
    print(classification_report(y_test, y_predict,target_names=news.target_names))

    return None

if __name__=='__main__':
    naviebayes()
    print("11111111")