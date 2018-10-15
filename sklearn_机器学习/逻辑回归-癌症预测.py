import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#均方误差
from sklearn.metrics import mean_squared_error, classification_report
#保存模型
from sklearn.externals import joblib

def logistic():
    """
    逻辑回归做2分类进行癌症预测（根据细胞大小）
    """
    #构造列标签名字11个
    column_name = ['Sample code number', 'Clump Thickness',\
                    'Uniformity of Cell Size','Uniformity of Cell Shape',\
                    'Marginal Adhesion', 'Single Epithelial Cell Size',\
                    'Bare Nuclei','Bland Chromatin', 'Normal Nucleoli',\
                    'Mitoses', 'Class']
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',\
                names=column_name)
    print(data)
    #缺失值处理
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna()

    #进行数据的分隔
    x_train, x_text, y_train, y_text = train_test_split(data[column_name[1:10]], data[column_name[10]], test_size=0.25)

    #进行标准化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_text = std.fit_transform(x_text)

    #逻辑回归预测
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    print("逻辑回归权重",lg.coef_)
    print("准确率", lg.score(x_text,y_text))

    y_predict = lg.predict(x_text)

    #召回率
    #良性2 恶性4 label,tar 相互对应
    print("召回率", classification_report(y_text, y_predict,\
                                       labels=[2, 4],
                                       target_names=['良性', '恶性']))
    return None

if __name__ == '__main__':
    logistic()