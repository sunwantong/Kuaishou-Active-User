# -*- coding: utf-8 -*-
#用来划分训练集和验证集
import pandas as pd
# import lightgbm as lgb
# from compiler.ast import flatten
import time
import operator
from functools import reduce
from scipy.sparse import hstack,vstack,csc_matrix
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import numpy as np
from sklearn import preprocessing
import operator
import matplotlib.pyplot as plt
from dateutil.parser import parse
from sklearn.cross_validation import train_test_split
from pandas import Series,DataFrame

import time
print('sleeping.....')
# time.sleep(1800)
print('start...')

# 训练
# xgboost
def xgboosts(df_train,df_test,df_eval):
    import pandas as pd
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    print('F select..')
    feature1 = [x for x in df_train.columns if
                x not in ['register_type', 'week', 'today_create_num', 'today_launch_num', 'video_id', 'user_id',
                          'register_day', 'day', 'author_id', 'action_type', 'page', 'label']]
    feature2 = [x for x in df_test.columns if
                x not in ['register_type', 'week', 'today_create_num', 'today_launch_num', 'video_id', 'user_id',
                          'register_day', 'day', 'author_id', 'action_type', 'page', 'label']]

    feature = [v for v in feature1 if v in feature2]
    print('F len :%s' % len(feature))


    df_train2 = df_train[feature]
    df_train2.fillna(-999, inplace=True)
    lr = LinearRegression()
    rfe = RFE(lr,n_features_to_select=400)#
    print('start select..')
    rfe.fit(df_train2, df_train['label'])

    print("Features sorted by their rank:")

    feat = sorted(zip(map(lambda x: x, rfe.ranking_), feature))
    print(feat)
    feature = []
    for ele in feat:
        feature.append(ele[1])
    print('this is output:')
    print(feature)
    print('xgb---training')
    dtrain = xgb.DMatrix(df_train[feature].values,df_train['label'].values)
    dpre = xgb.DMatrix(df_test[feature].values)
    deva = xgb.DMatrix(df_eval[feature].values,df_eval['label'].values)
    deva2 = xgb.DMatrix(df_eval[feature].values)
    param = {'max_depth': 5,
             'eta': 0.02,
             'objective': 'binary:logistic',

             'eval_metric': 'auc',
             'colsample_bytree':0.8,
             'subsample':0.8,
             'scale_pos_weight':1,
             # 'booster':'gblinear',
             'silent':1,
             # 'early_stopping_rounds':20
             # 'min_child_weight':5
             }
    # param['nthread'] =5
    print('xxxxxx')
    watchlist = [(deva, 'eval'), (dtrain, 'train')]
    num_round =600
    bst = xgb.train(param, dtrain, num_round, watchlist)
    print('xxxxxx')
    # 进行预测
    # dtest= xgb.DMatrix(predict)
    preds2 = bst.predict(dpre)
    # 保存整体结果。
    predict = df_test[['user_id']]
    predict['predicted_score'] = preds2
    # temp = predict.drop_duplicates(['user_id'])  # 去重
    predict.to_csv('result_all_select.csv', encoding='utf-8', index=None)

    # 保存预测为活跃的结果
    # result = predict[predict.predicted_score>=0.5]
    temp = predict.groupby(['user_id'])['predicted_score'].agg({'max_score': np.max})  #
    temp = temp.reset_index()
    predict = pd.merge(predict, temp, on=['user_id'], how='left')  #
    predict = predict[predict.max_score==predict.predicted_score]
    predict = predict.drop_duplicates(['user_id'])  # 去重
    # result = predict.sort_values(["predicted_score"], ascending=False).head(32486)
    result = predict.sort_values(["predicted_score"], ascending=False).head(25000)
    result = result[['user_id']]
   # result.to_csv('result_xgb_.txt',sep=' ',line_terminator='\r',index=False)
    result = result.sample(frac=1)  # 打乱样本
    result.to_csv('result_select.csv', encoding='utf-8', index=None)
#     F1
    N = df_eval[df_eval.label == 1]
    N = N.drop_duplicates(['user_id'])  # 去重
    N = N.user_id.values
    print(len(N))
    predict2 = df_eval[['user_id']]
    M = bst.predict(deva2)
    predict2['predicted_score'] = M
    predict2 = predict2.drop_duplicates(['user_id'])  # 去重
    result = predict2.sort_values(["predicted_score"], ascending=False).head(len(N))
    M=result.user_id.values

    intersection = [v for v in N if v in M]
    precision = len(intersection)/len(M)
    recall = len(intersection)/len(N)
    F1 = (2*precision*recall)/(precision+recall)
    print('F1:%s'%F1)


# df_train = pd.read_csv(r'train_set.csv')
df_all_train = pd.read_csv(r'all_train_set.csv')
df_test = pd.read_csv(r'test_set.csv')
df_eval = pd.read_csv(r'eval_set.csv')

# xgboosts(df_train,df_eval,df_eval)
xgboosts(df_all_train,df_test,df_eval)
