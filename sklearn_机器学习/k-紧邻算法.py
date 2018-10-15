from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


def knncls():
    #读取
    data = pd.read_csv('train.csv')
    #id, x, y, accuracy(误差), time, place_id
    #处理数据(xy缩小范围)
    data = data.query('x>1.0 & x < 1.25 & y >0.5 &y<0.75')

    #处理时间(一行数据time_values)
    time_values = pd.to_datetime(data['time'], unit='s')

    #把日期格式装换成字典格式
    time_values = pd.DatetimeIndex(time_values)
    data['day'] = time_values.day  #time_values的天
    data['hour'] = time_values.hour
    data['weekday'] = time_values.weekday

    #把原来的时间戳删除
    data = data.drop(['time'], axis=1) #按列删除
    data = data.drop(['row_id'], axis=1)

    #把签到数量少于n个目标位置删除
    place_count = data.groupby('place_id').count() #index = place_id
    # 筛选row_id > 3 ,reset_index() 把place_count分出来
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    #取出数据特征值，目标值
    y = data['place_id']
    #删除y行 剩下的都是特征
    x = data.drop(['place_id'], axis=1)

    #进行数据的分隔（train, text）
    #x数据特征， y数据的目标 textsize 是text的大小
    x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.25)

    #特征工程（标准化）木有标准化准确率0.02， 标准化后0.4
    std = StandardScaler()

    #测试，训练集特征值标准化
    x_train = std.fit_transform(x_train)
    std.transform(x_text)

    #进行算法流程
    '''不交叉网格'''
    knn = KNeighborsClassifier(n_neighbors=5) #n_nei就是k值
    #fit(训练), predict(预测), score(准确率)
    knn.fit(x_train, y_train)
    #得出预测结果
    y_predict = knn.predict(x_text)
    print('预测的签到位置:',y_predict)

    #准确率
    print(knn.score(x_text, y_test))

    '''交叉网格'''
    #knn = KNeighborsClassifier()
    param = {'n_neighbors':[3,5,10]}
    gc = GridSearchCV(knn, param_grid=param,cv=2)#cv几折分类
    gc.fit(x_train, y_train)
    #预测
    gc.score(x_text, y_test)
    gc.best_score_(x_text, y_test)#最好的参数
    gc.best_estimator_(x_text, y_test)#最好的模型
    return None

if __name__=='__main__':
    '''k-紧邻预测用户签到位置'''
    knncls()