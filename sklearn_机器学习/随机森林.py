import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def decision():
    '''决策树预测生死'''
    # 读取数据
    titan = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    # 处理数据,找出特征和目标值
    # "row.names","pclass(船票类别)","survived","name","age","embarked","home.dest","room","ticket","boat","sex"
    x = titan[['pclass', 'age', 'sex']]  # 抽取几个特征
    y = titan['survived']  # 是否存活
    # 处理缺失补充平均
    x['age'].fillna(x['age'].mean(), inplace=True)
    print(x.head(10))
    # 分隔数据
    x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程）-ont hot 编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))  # 默认一行x_train转化为一行字典
    print(dict.get_feature_names())

    x_text = dict.transform(x_text.to_dict(orient='records'))
    print(x_train)

    # 用决策树进行预测
    '''
    dec = DecisionTreeClassifier(max_depth=5) #预测的深度 这里可以加减枝参数最后分支少于多少不要
    dec.fit(x_train, y_train)
    #预测准确率
    print('预测的准确率:', dec.score(x_text, y_test))

    #导出决策树的结构
    export_graphviz(dec, out_file='tree.doc', feature_names=['age', 'pclass=1st',\
                                                             'pclass=2nd', 'pclass=3rd',\
                                                             'sex=female', 'sex=male'])
    #graphviz转为png
    #dot -Tpng tree.dot -o tree.npg
    '''

    # 使用随机森林 (超参数调优)
    rf = RandomForestClassifier()
    # 网格搜索与交叉验证
    param = {'n_estimators': [120, 200, 300, 500, 800, 1200], \
             'max_depth': [5, 8, 15, 25, 30]}  # 5*6 30次组合
    gc = GridSearchCV(rf, param_grid=param)
    gc.fit(x_train, y_train)
    # 预测准确率
    print('准确率:', gc.score(x_text, y_test))
    print("选择的参数模型:", gc.best_params_)
    #准确率: 0.8419452887537994
    #选择的参数模型: {'max_depth': 5, 'n_estimators': 200}
    return None


if __name__ == '__main__':
    decision()
