'''
特征的数量降维(不是数组的降维)
过滤式
嵌入式，决策树
'''


from sklearn.feature_selection import VarianceThreshold

def var():
    '''特征选择
    删除低方差的特征'''
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0,1,2],
                              [0,2,3],
                             [0,4,5]])
    print(data)
    return None

from sklearn.decomposition import PCA
def pac():
    '''主成分分析降维(主成分的90%)'''
    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[0,1,2],
                              [0,2,3],
                             [0,4,5]])
    print(data)
    return None
if __name__== '__main__':
    #var()
    pac()