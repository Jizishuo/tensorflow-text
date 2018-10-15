"""
波士顿放假数据线性回归
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#均方误差
from sklearn.metrics import mean_squared_error
#保存模型
from sklearn.externals import joblib

def mylinear():
    '''
    线性回归直接预测
    '''
    #获取数据
    lb = load_boston()

    #分隔数据
    x_train, x_text, y_train, y_text = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_text)
    #标准化处理(训练，结果都要标准化)
    #特征值标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_text = std_x.fit_transform(x_text)

    #目标值 y_是一维的转2维再标准化reshape(-1, 1)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_text = std_y.fit_transform(y_text.reshape(-1, 1))



    #读取保存好的训练模型
    model = joblib.load('liner.pkl')

    y_result = std_y.inverse_transform(model.predict(x_text))
    print('保存的模型预测结果:', y_result)

    #estimator预测
    #正规方程求解
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    #参数
    print("权重:",lr.coef_)

    #保存训练好的模型
    joblib.dump(lr, 'liner.pkl')

    #预测
    y_predict = lr.predict(x_text)
    print("预测：", y_predict)
    #转化为标准化前的值
    result = std_y.inverse_transform(y_predict)
    print("标准化前:",result)

    print("正规方程的均方误差:", mean_squared_error(std_y.inverse_transform(y_text), result))

    return None


def mylinear2():
    '''
     线性回归直接预测(梯度下降)
     '''
    # 获取数据
    lb = load_boston()

    # 分隔数据
    x_train, x_text, y_train, y_text = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_text)
    # 标准化处理(训练，结果都要标准化)
    # 特征值标准化
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_text = std_x.fit_transform(x_text)

    # 目标值 y_是一维的转2维再标准化reshape(-1, 1)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_text = std_y.fit_transform(y_text.reshape(-1, 1))

    # estimator预测
    # 梯度下降求解
    sqd = SGDRegressor()
    sqd.fit(x_train, y_train)

    # 参数
    print("权重:", sqd.coef_)

    # 预测
    y_predict = sqd.predict(x_text)
    print("预测：", y_predict)
    # 转化为标准化前的值
    result = std_y.inverse_transform(y_predict)
    print("标准化前:", result)

    print("梯度下降的均方误差:", mean_squared_error(std_y.inverse_transform(y_text), result))



    # 正则化的岭回归求解
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)

    # 参数
    print("权重:", rd.coef_)

    # 预测
    y_predict = rd.predict(x_text)
    print("灵归预测：", y_predict)
    # 转化为标准化前的值
    result = std_y.inverse_transform(y_predict)
    print("标准化前:", result)

    print("岭回归的均方误差:", mean_squared_error(std_y.inverse_transform(y_text), result))

    return None


if __name__=='__main__':
    mylinear()
    #mylinear2()

