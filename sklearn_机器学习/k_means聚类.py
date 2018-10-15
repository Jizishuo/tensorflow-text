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


#假设分成4类

from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

#减少样本数量减少
x = data[:500]

#聚类4类别
km = KMeans(n_clusters=4)
#训练
km.fit(x)

#结果是500个4分类
predict = km.predict(x)

#聚类效果轮廓系数  接近1 是最好的  -1最差
s = silhouette_score(x, predict)
print(s)


#显示
plt.figure(figsize=[10, 10])
#抽取2个特征画出来
#建立4个颜色的列表
colored = ['orange', 'green', 'blue', 'purple']
colr = [colored[i] for i in predict]
plt.scatter(x[:,1], x[:10], color = colr)
plt.xlabel('1')
plt.ylabel('10')
plt.show()