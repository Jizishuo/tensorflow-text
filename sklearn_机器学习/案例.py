import pandas as pd
from sklearn.decomposition import PCA

#读取4张表的数据
prior = pd.read_csv('1.csv')
products = pd.read_csv('2.csv')
orders = pd.read_csv('3.csv')
aisles = pd.read_csv('4.csv')

#合并4张表
#(用户-用户类别)
_mg = pd.merge(prior, products, on=['product_id','product_id'])
_mg = pd.merge(_mg,orders, on=['order_id', 'order_id'])
mt = pd.merge(_mg, aisles, on=['aisle_id', 'aisle_id'])

#print(mt.head(10))
#交叉表
cross = pd.crosstab(mt['user_id'], mt['aisle'])
#主成分分析
pac = PCA(n_components=0.9)#保存90%对的信息量
data = pac.fit_transform(cross)
print(data.shape)#特征127-27
print(data)
