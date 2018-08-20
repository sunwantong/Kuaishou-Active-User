# 读取压缩包，并且开始跑模型
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import operator
import os

def xgb_model(train,test):
    
    
    # id_columntr = ['user_id','register_day','label']
    # id_columnte = ['user_id','register_day']

    id_columntr = ['user_id', 'register_day', 'label','before_create_seven_day_rate','con_create_day_count_min',
                    'before_act_time_gap_max','before_launch_time_gap_min','gap_launch_day_max','before_create_time_gap_max',
                    'con_launch_day_count_min','before_act_time_gap_min','before_create_time_gap_mean']
                    
    id_columnte = ['user_id', 'register_day','before_create_seven_day_rate','con_create_day_count_min',
                    'before_act_time_gap_max','before_launch_time_gap_min','gap_launch_day_max','before_create_time_gap_max',
                    'con_launch_day_count_min','before_act_time_gap_min','before_create_time_gap_mean']


    train_x = train.drop(id_columntr, axis=1)
    train_y = train['label']
    test_x = test.drop(id_columnte, axis=1)

    xgb_train = xgb.DMatrix(train_x, label=train_y)
    xgb_test = xgb.DMatrix(test_x)

    params = {'booster': 'gbtree',
            #   'objective': 'binary:logistic',  # 二分类的问题
              'objective': 'rank:pairwise',  # 二分类的问题
              # 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
              'max_depth': 5,  # 构建树的深度，越大越容易过拟合
              # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
              'subsample': 0.7,  # 随机采样训练样本
              'colsample_bytree': 0.7,  # 生成树时进行的列采样
              'min_child_weight': 3,
              # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
              # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
              # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
              'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
              'eta': 0.03,  # 如同学习率
              'nthread': 7,  # cpu 线程数
              'eval_metric': 'auc'  # 评价方式
              }

    plst = list(params.items())
    num_rounds = 800  #
    watchlist = [(xgb_train, 'train')]
    # early_stopping_rounds    当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)
    pred_value = model.predict(xgb_test)

    #-----------------------important of feature start-----------------------------------------
    importance = model.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)
    # df.to_csv('../gene_data/feature_important.csv', index=None)
    # -----------------------important of feature end-----------------------------------------
    return pred_value


def gene_result(pred_value,test_range):
    tess = test_range[["user_id"]].astype('str')
    a = pd.DataFrame(pred_value, columns=["probability"])
    res = pd.concat([tess, a["probability"]], axis=1)
    res = res.drop_duplicates()

    res.to_csv("sun_result_0806_xgb_feat_selection.txt", index=None, header=False)  

def load_csv():
    train = pd.read_csv('/home/kesci/train.txt',sep=',')
    test = pd.read_csv('/home/kesci/test.txt',sep=',')

    # train = pd.read_csv('train.txt', sep=',')
    # test = pd.read_csv('test.txt', sep=',')
    
    print(len(train))
    print(len(test))

    return train,test



def main():
    train,test = load_csv()
    pred_value = xgb_model(train,test)
    gene_result(pred_value, test)

if __name__ == '__main__':
    main()