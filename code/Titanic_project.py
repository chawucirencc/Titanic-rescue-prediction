#!/usr/bin/env.python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


def load_data(train_path):
    """
    加载数据，设置显示格式以及查看数据的基础特征
    :return: 返回Dataframe数据格式
    """
    data = pd.read_csv(train_path)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 180)
    # print(data.head())
    # print(data.columns)
    # print(file_data.shape)
    # print(file_data.info())                 # 查看数据的属性信息
    # print(file_data.describe())             # 查看数据的描述信息
    # print(file_data.corr())                 # 查看数据属性之间的相关性
    return data


def plot_feature(all_data):
    """
    用图形展示特征各个特征
    """
    fig_1 = plt.figure()
    fig_1.set(alpha=0.8)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig_1.subplots_adjust(wspace=0.5, hspace=0.5)
    fig_1.add_subplot(221)
    all_data['Survived'].value_counts().plot(kind='bar', width=0.25)
    plt.xlabel('乘客是否获救（1为获救）'); plt.ylabel(u'人数'); plt.title(u'乘客获救情况')
    fig_1.add_subplot(222)
    all_data['Pclass'].value_counts().plot(kind='bar', width=0.4)
    plt.xlabel('舱位'); plt.ylabel('人数'); plt.title('各个舱位的人数')
    fig_1.add_subplot(223)
    all_data['Embarked'].value_counts().plot(kind='bar', width=0.4)
    plt.xlabel('登船口'); plt.ylabel('人数'); plt.title('通过各个登船口的人数')
    fig_1.add_subplot(224)
    all_data['Sex'].value_counts().plot(kind='bar', width=0.25)
    plt.xlabel('性别'); plt.ylabel('人数'); plt.title('乘客男女比例')

    fig_2 = plt.figure()
    fig_2.set(alpha=0.8)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig_2.subplots_adjust(wspace=0.5, hspace=0.5)
    fig_2.add_subplot(231)
    all_data['Age'].plot(kind='kde')
    plt.xlim([-10, 100]); plt.xlabel('年龄'); plt.ylabel('频率')
    plt.title('关于年龄的频率图')
    fig_2.add_subplot(232)
    all_data['SibSp'].value_counts().plot(kind='bar')
    plt.xlabel('堂兄弟/妹个数'); plt.ylabel('人数'); plt.title('乘客堂兄弟/妹个数')
    fig_2.add_subplot(233)
    all_data['Parch'].value_counts().plot(kind='bar')
    plt.xlabel('父母/小孩的个数'); plt.ylabel('人数'); plt.title('乘客的父母/小孩的个数')
    fig_2.add_subplot(2,3, (4, 6))
    all_data['Age'][all_data['Pclass'] == 1].plot(kind='kde')
    all_data['Age'][all_data['Pclass'] == 2].plot(kind='kde')
    all_data['Age'][all_data['Pclass'] == 3].plot(kind='kde')
    plt.title('年龄和舱位的频率图'); plt.xlabel('年龄'); plt.xlim([-10, 100])
    plt.legend(('1等舱', '2等舱', '3等舱'))

    plt.show()


def feature_associate(all_data):
    """
    特征的关联性，关于客舱等级、性别、年龄、亲属个数、登船口和是否获救的关系。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sur_0 = all_data['Pclass'][all_data['Survived'] == 0].value_counts()
    sur_1 = all_data['Pclass'][all_data['Survived'] == 1].value_counts()
    sur_df = pd.DataFrame({'获救': sur_1, '未获救': sur_0})
    sur_df.plot(kind='bar', stacked=True, width=0.35)
    plt.xlabel('客舱等级'); plt.ylabel('人数'); plt.title('客舱等级与获救人数的关系')

    sur_m = all_data['Survived'][all_data['Sex'] == 'male'].value_counts()
    sur_f = all_data['Survived'][all_data['Sex'] == 'female'].value_counts()
    sex_df = pd.DataFrame({'男性': sur_m, '女性': sur_f})
    sex_df.plot(kind='bar', stacked=True, width=0.35)
    plt.title('关于性别的获救情况')
    plt.xlabel('是否获救（1为获救）'); plt.ylabel('人数')

    sur_age_0 = all_data['Age'][all_data['Survived'] == 0]
    sur_age_1 = all_data['Age'][all_data['Survived'] == 1]
    age__df = pd.DataFrame({'获救': sur_age_1, '未获救': sur_age_0})
    age__df.plot(kind='kde')
    plt.xlim([-20, 100]); plt.xlabel('年龄')
    plt.title('分别在获救和未获救中的年龄分布')

    sib_0 = all_data['SibSp'][all_data['Survived'] == 0].value_counts()
    sib_1 = all_data['SibSp'][all_data['Survived'] == 1].value_counts()
    sib_df = pd.DataFrame({'获救': sib_1, '未获救': sib_0})
    sib_df.plot(kind='bar', stacked=True)
    plt.title('堂兄弟/妹个数与是否获救的关联')
    plt.ylabel('人数'); plt.xlabel('堂兄弟/妹个数')

    parch_0 = all_data['Parch'][all_data['Survived'] == 0].value_counts()
    parch_1 = all_data['Parch'][all_data['Survived'] == 1].value_counts()
    parch_df = pd.DataFrame({'获救': parch_1, '未获救': parch_0})
    parch_df.plot(kind='bar', stacked=True)
    plt.title('父母与小孩个数和是否获救的关联')
    plt.ylabel('人数'); plt.xlabel('父母与小孩个数')

    emb_0 = all_data['Embarked'][all_data['Survived'] == 0].value_counts()
    emb_1 = all_data['Embarked'][all_data['Survived'] == 1].value_counts()
    emb_df = pd.DataFrame({'获救': emb_1, '未获救': emb_0})
    emb_df.plot(kind='bar', stacked=True)
    plt.title('登船口和是否获救的关联'); plt.xlabel('登船口'); plt.ylabel('人数')

    cabin_yes = all_data['Survived'][pd.notnull(all_data['Cabin'])].value_counts()
    cabin_no = all_data['Survived'][pd.isnull(all_data['Cabin'])].value_counts()
    cabin_pd = pd.DataFrame({'有信息': cabin_yes, '无信息': cabin_no})
    cabin_pd.plot(kind='bar', stacked=True)
    plt.ylabel('人数'); plt.xlabel('是否获救（1为获救）')
    plt.title('关于是否有客舱信息与是否获救的关联')

    plt.show()


def deal_missing_data(all_Data):
    """
    处理缺失数据，Age缺失数据较多，用随机森林回归预测的方法填补缺失值，
    对缺失数据较多的客舱属性进行转化处理，并且添加一个新的Child特征（Age小于等于12的赋值为1，其余的为0）。
    :return:返回完整的数据
    """
    all_Data.loc[(all_Data.Fare.isnull()), 'Fare'] = 0
    age_df = all_Data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    y = known_age[:, 0]
    x = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=1, n_estimators=2000)
    rfr.fit(x, y)
    pre_age = rfr.predict(unknown_age[:, 1::])
    all_Data.loc[all_Data['Age'].isnull(), 'Age'] = pre_age
    all_Data.loc[all_Data['Cabin'].notnull(), 'Cabin'] = 'Yes'
    all_Data.loc[all_Data['Cabin'].isnull(), 'Cabin'] = 'No'
    Child = []
    for i in all_Data['Age']:
        if i <= 12:
            Child.append(1)
        else:
            Child.append(0)
    all_Data['Child'] = Child
    # print(all_Data.head())
    # print(all_Data.info())
    
    return all_Data


def deal_data(complete_data):
    """
    对非数值型特征进行数值转化，将其合并成新的数据。
    """
    dummies_cabin = pd.get_dummies(complete_data['Cabin'], prefix='Cabin')
    dummies_sex = pd.get_dummies(complete_data['Sex'], prefix='Sex')
    # dummies_embarked = pd.get_dummies(complete_data['Embarked'], prefix='Embarked')
    dummies_pclass = pd.get_dummies(complete_data['Pclass'], prefix='Pclass')
    data = pd.concat([complete_data, dummies_cabin, dummies_pclass, dummies_sex], axis=1)
    drop = ['Cabin', 'Sex', 'Embarked', 'Pclass', 'Name', 'Ticket']
    data.drop(drop, axis=1, inplace=True)
    # print(data.head())
    # print(data.columns)
    return data


def split_data(new_data):
    """
    切分数据，将最终结果和特征进行分离。
    :return:返回特征数据X和结果数据y
    """
    y = np.array(new_data['Survived'])
    data = new_data.drop(['Survived'], axis=1)
    # print(data.columns)
    X = data.values
    # print(X[:5, :])
    return X, y


def std_data(X):
    """
    可选函数，选择是否对数据进行标准化，只对特征进行标准化。
    :return:返回转化后的数据
    """
    std = StandardScaler()
    result = std.fit_transform(X)
    return result


def create_model(X, y):
    """
    建立模型，通过交叉验证得到平均得分，并使用GridSearchCV进行网格搜索得到最优参数。
    :return:返回最终建立的模型。
    """
    kfo = KFold(n_splits=10, shuffle=True, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    log_model = LogisticRegression(C=1, max_iter=100, penalty='l2', tol=1e-6)
    log_model.fit(X_train, y_train)

    """para_meter = [{'penalty':['l1', 'l2'], 'tol': [1e-4, 1e-5, 1e-6, 1e-7],
                   'C':[0.5, 1.0, 1.5], 'max_iter':[100, 200, 300, 400, 500]}]
    clf_model = GridSearchCV(estimator=log_model, param_grid=para_meter, cv=kfo, n_jobs=-1)
    clf_model.fit(X_train, y_train)
    print('最优参数-', clf_model.best_params_)"""
    """# gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1)
    # gb_model.fit(X_train, y_train)
    # para_meter_gb = [{'n_estimators': [100, 150, 200, 250, 300],
    #                   'max_depth': [1, 2, 3, 4], 'learning_rate':[0.001, 0.01, 0.1]}]
    # clf_gb = GridSearchCV(estimator=gb_model, param_grid=para_meter_gb, cv=kfo)
    # clf_gb.fit(X_train, y_train)
    # print(clf_gb.best_params_)"""

    result = cross_val_score(log_model, X_train, y_train, cv=kfo, scoring='accuracy')
    print('交叉验证结果的均值-->', result.mean())
    print('gb_model-->', log_model.score(X_test, y_test))
    return log_model


def model_fusion(model, X, y):
    """
    模型融合，通过 BaggingClassifier 对模型进行融合，输出对测试集的评分。
    :return:返回融合后的模型。
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    bc_clf = BaggingClassifier(model, random_state=10, n_estimators=10, max_features=1.0, max_samples=0.1)
    bc_clf.fit(X_train, y_train)
    print('bc_model-->', bc_clf.score(X_test, y_test))
    return bc_clf


def plot_learning_curve(model, X, y):
    """
    画出学习曲线，误差曲线以及ROC曲线，计算出AUC的值。（AUC=0.89）
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
    train_size, train_scores, test_scores = learning_curve(model, X, y,  cv=20, scoring='accuracy', n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(131)
    plt.plot(train_size, train_scores_mean, 'o-')
    plt.plot(train_size, test_scores_mean, 'o-')
    plt.xlabel('样本数量'); plt.ylabel('评分')
    plt.legend(('train_scores_mean', 'test_scores_mean'))
    plt.title('学习曲线')
    plt.subplot(132)
    plt.plot(train_size, train_scores_std, 'o-')
    plt.plot(train_size, test_scores_std, 'o-')
    plt.xlabel('样本数量'); plt.ylabel('评分')
    plt.legend(('train_scores_std', 'test_scores_std'))
    plt.title('误差曲线')
    plt.subplot(133)
    fpr, tpr, thr = roc_curve(y_test, model.decision_function(X_test))
    print('AUC-->', auc(fpr, tpr))
    plt.plot(fpr, tpr)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC曲线')
    plt.show()


def get_result(model, X):
    """
    应用模型，得到预测结果，并将结果按指定格式保存到文件。
    """
    pre = model.predict(X)
    df = pd.read_csv('L:/data_deal/data/all/gender_submission.csv')
    result = pd.DataFrame({'PassengerId': df['PassengerId'], 'Survived': pre.astype(np.int32)})
    result.to_csv('log_result.csv', index=False)


def main():
    train_path = 'L:/data_deal/data/all/train.csv'        # 训练数据集的文件路径
    test_path = 'L:/data_deal/data/all/test.csv'          # 测试数据的文件路径
    train_data = load_data(train_path)
    test_data = load_data(test_path)                      # 加载测试数据
    plot_feature(train_data)                              # 特征可视化
    feature_associate(train_data)                           # 描述特征和结果的关联性
    complete_train_data = deal_missing_data(train_data)   # 处理缺失值
    complete_test_data = deal_missing_data(test_data)
    new_train_data = deal_data(complete_train_data)       # 处理数据，对数据进行转化。
    new_test_data = deal_data(complete_test_data)
    X, y = split_data(new_train_data)                     # 切分数据
    X_s = std_data(X)                                     # 可选择对数据是否标准化
    X_test_new= std_data(new_test_data)                   # 标准化数据
    model = create_model(X_s, y)                          # 建立模型
    bag_model = model_fusion(model, X_s, y)               # 模型融合和优化
    plot_learning_curve(bag_model, X_s, y)                # 画出学习曲线等
    get_result(bag_model, X_test_new)                     # 得到结果


if __name__ == '__main__':
    main()
