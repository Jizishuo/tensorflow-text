'''
from sklearn.datasets import load_iris
#营维花的样本 150个 3种 4个特征

#划分数据集
from sklearn.model_selection import train_test_split

li = load_iris()
print(li.data,'特征值')
print(li.target, '目标值')

#划分 返回值包含训练集和测试集
x_train, x_text, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25)

print('训练集', x_train, y_train)
print('测试集', x_text, y_test)
'''

#获取大数据20个新闻分类数据集
from sklearn.datasets import fetch_20newsgroups

news = fetch_20newsgroups(subset='all')

print(news.data)