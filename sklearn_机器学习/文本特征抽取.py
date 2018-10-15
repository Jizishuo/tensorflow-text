'''
from  sklearn.feature_extraction.text import CountVectorizer
#实例化
vertor = CountVectorizer()
#调用fit输入并转化数据
res  = vertor.fit_transform(['this is a pig', 'thi is too long'])
#打印结果
print(vertor.get_feature_names())
print(res.toarray())
'''

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#处理字典的

def dictves():
    #字典数据抽取
    #实例化
    dict = DictVectorizer(sparse=False)
    data = [{'city':'北京', 'tem':100},
            {'city':'上海', 'tem':250}]
    data1 = dict.fit_transform(data)
    print(data1)#矩阵
    print(dict.get_feature_names())#特征值
    return None

def countver():
    '''文本特征值化'''
    cv = CountVectorizer()
    data = cv.fit_transform(['this is a pig', 'this is too long'])
    #data = cv.fit_transform(['生死 看淡 了', '我 不用 ptrhon'])
    print(data.toarray())
    print(cv.get_feature_names())
    return None

import jieba
def coutwork():
    c1 = jieba.cut('野生大黄鱼一直是我国当代贵菜，这条11万一条。而且这条鱼是真大...\
    7斤4两，真的大，可以上新闻的那么大，之前舟山渔民捕捞到4斤重的大黄鱼都可以卖3万块。\
    这条成本也不便宜。因为太大了其中一半直接蒸了。传统做法。另外50%做成了西郊5号经典菜“花间冻黄鱼”\
    的超级放大版。这道菜的做法是把鱼要先风一下浓缩一下味道，再上蒸箱蒸熟。然后用小鱼炖汤过滤起冻，\
    最后和黄鱼一起定型。最后顶上一层鱼子酱来强化鱼的味道，这部分在鱼子酱部分说。这道菜主要吃的是咸鲜味。')
    #转化列表
    c1 = list(c1)
    #装换字符串加空格
    c1 = ' '.join(c1)
    return c1

def zhongwen():
    '''中文特征值化'''
    c1 = coutwork()
    print(c1)
    cv = CountVectorizer()
    data = cv.fit_transform([c1])
    #data = cv.fit_transform(['生死 看淡 了', '我 不用 ptrhon'])
    print(data.toarray())
    print(cv.get_feature_names())
    return None

from sklearn.preprocessing import MinMaxScaler
def mm():
    '''归一化处理(同等重要的特征归一)'''
    #使得某一个特征对最终结果不会赵成过大的影响
    #受到异常点的影响大
    mm = MinMaxScaler(feature_range=(2,3))
    data = mm.fit_transform([[80,2,10,40],[60,4,15,45],[75,3,13,46]])
    print(data)
    return None

from sklearn.preprocessing import StandardScaler
def stand():
    '''标准化(类似归一化)'''
    std = StandardScaler()
    data = std.fit_transform([[1.,-1.,3.],[1.,4.,2.],[4.,6.,-2.]])
    print(data)

from sklearn.preprocessing import Imputer
def im():
    '''缺失值填补'''
    im = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = im.fit_transform('数据')
    print(data)
    return None

if __name__=='__main__':
    #dictves()
    #countver()
    #zhongwen()
    #tf-idf几篇文章里相同词语的重要性tifdfVectorizer()
    #mm()
    stand()
    #im()