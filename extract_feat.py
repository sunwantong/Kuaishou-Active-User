#特征提取   8209 change_window

 
import pandas as pd
import numpy as np
import time
import datetime
from collections import Counter
import scipy.stats as sp
import xgboost as xgb
import matplotlib.pyplot as plt
import operator
import os
from concurrent.futures import ProcessPoolExecutor,wait, as_completed

def get_time_gap(strs,parm):
    time = strs.split(":")
    time = list(set(time))
    time = sorted(list(map(lambda x:int(x),time)))
    time_gap = []
    #用户只在当天活跃
    if len(time) == 1:
        return -20

    for index, value in enumerate(time):
        if index <= len(time) - 2:
            gap = abs(time[index] - time[index + 1])
            time_gap.append(gap)

    if parm == '1':
        return np.mean(time_gap)
    elif parm == '2':
        return np.max(time_gap)
    elif parm == '3':
        return np.min(time_gap)
    elif parm == '4':
        return np.std(time_gap)

# 用户连续启动的天数
def get_continue_launch_day(strs):
    time = strs.split(":")
    time = list(set(time))
    time = sorted(list(map(lambda x:int(x),time)))
    time_gap = []
    #用户只在当天活跃
    if len(time) == 1:
        return -20

    for index, value in enumerate(time):
        if index <= len(time) - 2:
            gap = abs(time[index] - time[index + 1])
            time_gap.append(gap)

    if np.mean(time_gap) == 1:  # 全是1
        return len(time_gap) + 1

    time_gap = [ele for ele in time_gap if ele == 1]
    return len(time_gap) + 1


# 连续几天启动的总的次数，均值，方差，max,min,mode
def get_continue_launch_count(strs,parm):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(), key=lambda x: x[0], reverse=False)
    key_list = []
    value_list = []
    if len(time) == 1:
        return -2
    for key,value in dict(time).items():
        key_list.append(int(key))
        value_list.append(int(value))

    if np.mean(np.diff(key_list, 1)) == 1:
        if parm == '1':
            return np.mean(value_list)
        elif parm == '2':
            return np.max(value_list)
        elif parm == '3':
            return np.min(value_list)
        elif parm == '4':
            return np.sum(value_list)
        elif parm == '5':
            return np.std(value_list)
    else:
        return -1


# 用户非连续启动的总次数，均值，标准差，max,min,skew,kurt
def get_not_continue_launch_count(strs,parm):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(), key=lambda x: x[0], reverse=False)
    key_list = []
    value_list = []
    if len(time) == 1:
        return -2
    for key, value in dict(time).items():
        key_list.append(int(key))
        value_list.append(int(value))

    if parm == '1':
        return np.mean(value_list)
    elif parm == '2':
        return np.max(value_list)
    elif parm == '3':
        return np.min(value_list)
    elif parm == '4':
        return np.sum(value_list)
    elif parm == '5':
        return np.std(value_list)


def cur_day_repeat_count(strs):
    time = strs.split(":")
    time = dict(Counter(time))
    time = sorted(time.items(), key=lambda x: x[1], reverse=False)
    # 一天一次启动
    if len(time) == 1 & time[0][1] == 1:
        return 0
    # 一天多次启动
    elif len(time) == 1 & time[0][1] > 1:
        return 1
    # 多天多次启动
    elif (len(time) > 1) & (time[0][1] >= 2):
        return 2
    else:
        return 3



def analysis_register():
    # user_register = pd.read_csv('../data/user_register_log.txt',sep='\t',header=None)
    user_register = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt', sep='\t', header=None)
    user_register.columns = ['user_id','register_day','register_type','device_type']
    return user_register

def analysis_launch():
    # user_launch = pd.read_csv('../data/app_launch_log.txt',sep='\t',header=None)
    user_launch = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt', sep='\t', header=None)
    user_launch.columns = ['user_id','launch_day']
    return user_launch

def analysis_create_log():
    # user_create = pd.read_csv('../data/video_create_log.txt',sep='\t',header=None)
    user_create = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt', sep='\t', header=None)
    user_create.columns = ['user_id','create_day']
    return user_create

def analysis_activity():
    # user_activity = pd.read_csv('../data/user_activity_log.txt',sep='\t',header=None)
    user_activity = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt', sep='\t', header=None)
    user_activity.columns = ['user_id','activity_day','page','video_id','author_id','action_type']
    return user_activity

"""
1-16预测17-23 1-23预测24-30 1-30预测31-37

"""


def data_range_train_one(user_reg,user_launch,user_create,user_activity):
    USER_ID = user_reg[(user_reg.register_day <= 16)]

    user_reg_one = user_reg[(user_reg.register_day <= 16)]
    user_launch_one = user_launch[(user_launch.launch_day >= 1) & (user_launch.launch_day <= 16)]
    user_creat_one = user_create[(user_create.create_day >= 1) & (user_create.create_day <= 16)]
    user_activity_one = user_activity[(user_activity.activity_day >= 1) & (user_activity.activity_day <= 16)]

    user_reg_one = extract_register_feature(user_reg_one,USER_ID,16)
    user_launch_one = extract_launch_feature(user_launch_one,USER_ID,16)
    user_creat_one = extract_create_feature(user_creat_one,USER_ID,16)
    user_activity_one = extract_activity_feature(user_activity_one,USER_ID,16)

    data = pd.merge(user_reg_one, user_activity_one, on='user_id', how="left")
    data = pd.merge(data, user_creat_one, on='user_id', how="left")
    data = pd.merge(data,user_launch_one, on='user_id', how="left")

    user_launch_pred = user_launch[(user_launch.launch_day >= 17) & (user_launch.launch_day <= 23)]
    user_create_pred = user_create[(user_create.create_day >= 17) & (user_create.create_day <= 23)]
    user_activity_pred = user_activity[(user_activity.activity_day >= 17) & (user_activity.activity_day <= 23)]

    user_id_launch = user_launch_pred['user_id']
    user_id_create = user_create_pred['user_id']
    user_id_activity = user_activity_pred['user_id']

    f_user_id = set(user_id_launch) | set(user_id_create) | set(user_id_activity)
    data['label'] = data['user_id'].apply(lambda x:1 if x in f_user_id else 0)


    print(data['label'].value_counts())
    return data


def data_range_train_two(user_reg,user_launch,user_create,user_activity):
    USER_ID = user_reg[(user_reg.register_day <= 23)]

    user_reg_one = user_reg[(user_reg.register_day <= 20)]
    user_launch_one = user_launch[(user_launch.launch_day >= 1) & (user_launch.launch_day <= 23)]
    user_creat_one = user_create[(user_create.create_day >= 1) & (user_create.create_day <= 23)]
    user_activity_one = user_activity[(user_activity.activity_day >= 1) & (user_activity.activity_day <= 23)]

    user_reg_one = extract_register_feature(user_reg_one,USER_ID,23)
    user_launch_one = extract_launch_feature(user_launch_one,USER_ID,23)
    user_creat_one = extract_create_feature(user_creat_one,USER_ID,23)
    user_activity_one = extract_activity_feature(user_activity_one,USER_ID,23)


    data = pd.merge(user_reg_one, user_activity_one, on='user_id', how="left")
    data = pd.merge(data, user_creat_one, on='user_id', how="left")
    data = pd.merge(data, user_launch_one, on='user_id', how="left")

    user_launch_pred = user_launch[(user_launch.launch_day >= 24) & (user_launch.launch_day <= 30)]
    user_create_pred = user_create[(user_create.create_day >= 24) & (user_create.create_day <= 30)]
    user_activity_pred = user_activity[(user_activity.activity_day >= 24) & (user_activity.activity_day <= 30)]

    user_id_launch = user_launch_pred['user_id']
    user_id_create = user_create_pred['user_id']
    user_id_activity = user_activity_pred['user_id']

    f_user_id = set(user_id_launch) | set(user_id_create) | set(user_id_activity)
    data['label'] = data['user_id'].apply(lambda x:1 if x in f_user_id else 0)


    print(data['label'].value_counts())
    return data


def data_range_test(user_reg,user_launch,user_create,user_activity):
    USER_ID = user_reg[(user_reg.register_day <= 30)]


    user_reg_test = user_reg[(user_reg.register_day <= 30)]
    user_launch_test = user_launch[(user_launch.launch_day >= 1) & (user_launch.launch_day <= 30)]
    user_creat_test = user_create[(user_create.create_day >= 1) & (user_create.create_day <= 30)]
    user_activity_test = user_activity[(user_activity.activity_day >= 1) & (user_activity.activity_day <= 30)]

    user_reg_test = extract_register_feature(user_reg_test,USER_ID,30)
    user_launch_test = extract_launch_feature(user_launch_test,USER_ID,30)
    user_creat_test = extract_create_feature(user_creat_test,USER_ID,30)
    user_activity_test = extract_activity_feature(user_activity_test,USER_ID,30)

    data = pd.merge(user_reg_test,user_activity_test,on='user_id',how="left")
    data = pd.merge(data, user_creat_test, on='user_id', how="left")
    data = pd.merge(data, user_launch_test,on='user_id', how="left")


    return data


def extract_activity_feature(user_activity,USER_ID,feature_high):
    # 用户活跃次数
    user_act_count = user_activity.groupby(['user_id'],as_index=False)['activity_day'].agg({"user_act_count":'count'})

    # user page
    user_page = pd.crosstab(user_activity.user_id,user_activity.page).reset_index()

    # user action_type
    user_action_type = pd.crosstab(user_activity.user_id, user_activity.action_type).reset_index()

    # user - page mode
    user_page_mode = user_activity.groupby(['user_id'])['page'].agg(lambda x: x.value_counts().index[0]).reset_index()
    user_page_mode.rename(columns={"page": "page_mode"}, inplace=True)

    # user - action_type mode
    user_action_mode = user_activity.groupby(['user_id'])['action_type'].agg(lambda x: x.value_counts().index[0]).reset_index()
    user_action_mode.rename(columns={"action_type": "action_type_mode"}, inplace=True)

    # 用户查看多少不同的video
    user_diff_video = user_activity.groupby(['user_id'],as_index=False)['video_id'].agg({"user_diff_video":"nunique"})

    # 用户查看多少不同的author
    user_diff_author = user_activity.groupby(['user_id'], as_index=False)['author_id'].agg({"user_diff_author": "nunique"})

    # is_self
    user_activity['is_self'] = list(map(lambda x,y: 1 if x == y else 0,user_activity['user_id'],user_activity['author_id']))
    user_is_author = user_activity[['user_id','is_self']]
    user_is_author = user_is_author.drop_duplicates(subset=['user_id'])

    # 距离窗口末端的时间
    user_activity['gap_activity_day'] = feature_high - user_activity['activity_day']

    # 均值
    gap_activity_day_mean = user_activity.groupby(['user_id'],as_index=False)['gap_activity_day'].agg({"gap_activity_day_mean":np.mean})
    # 最大值
    gap_activity_day_max = user_activity.groupby(['user_id'], as_index=False)['gap_activity_day'].agg({"gap_activity_day_max": np.max})
    # 最小值
    gap_activity_day_min = user_activity.groupby(['user_id'], as_index=False)['gap_activity_day'].agg({"gap_activity_day_min": np.min})
    # 众数
    gap_activity_day_mode = user_activity.groupby(['user_id'],as_index=False)['gap_activity_day'].agg(lambda x: x.value_counts().index[0])
    gap_activity_day_mode.rename(columns={"gap_activity_day": "gap_activity_day_mode"}, inplace=True)
    # skew
    gap_activity_day_skew = user_activity.groupby(['user_id'], as_index=False)['gap_activity_day'].agg({'gap_activity_day_skew': sp.stats.skew})
    # kurt
    gap_activity_day_kurt = user_activity.groupby(['user_id'], as_index=False)['gap_activity_day'].agg({'gap_activity_day_kurt': sp.stats.kurtosis})
    # std
    gap_activity_day_var = user_activity.groupby(['user_id'], as_index=False)['gap_activity_day'].agg({'gap_activity_day_std': np.std})
    # 用户活跃天数
    user_act_day_count = user_activity.groupby(['user_id'], as_index=False)['activity_day'].agg({"user_act_day_count": 'nunique'})


    feat1 = user_activity[['user_id', 'activity_day']]
    feat1['activity_day'] = feat1['activity_day'].astype('str')
    feat1 = feat1.groupby(['user_id'])['activity_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat1.rename(columns={'activity_day': 'act_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat1['cur_day_repeat_count'] = feat1['act_list'].apply(cur_day_repeat_count)
    # 用户连续活跃天数
    feat1['con_act_day_count'] = feat1['act_list'].apply(get_continue_launch_day)

    # 连续几天启动次数的均值，
    feat1['con_act_day_count_mean'] = feat1['act_list'].apply(get_continue_launch_count, args=('1'))
    # 最大值，
    feat1['con_act_day_count_max'] = feat1['act_list'].apply(get_continue_launch_count, args=('2'))
    # 最小值
    feat1['con_act_day_count_min'] = feat1['act_list'].apply(get_continue_launch_count, args=('3'))
    # 次数
    feat1['con_act_day_count_total'] = feat1['act_list'].apply(get_continue_launch_count, args=('4'))
    #方差
    feat1['con_act_day_count_std'] = feat1['act_list'].apply(get_continue_launch_count, args=('5'))

    # 均值
    feat1['act_time_gap_mean'] = feat1['act_list'].apply(get_time_gap, args=('1'))
    # max
    feat1['act_time_gap_max'] = feat1['act_list'].apply(get_time_gap, args=('2'))
    # min
    feat1['act_time_gap_min'] = feat1['act_list'].apply(get_time_gap, args=('3'))
    # std
    feat1['act_time_gap_std'] = feat1['act_list'].apply(get_time_gap, args=('4'))
    # 平均行为次数
    feat1['mean_act_count'] = feat1['act_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat1['act_seven_day_rate'] = feat1['act_list'].apply(lambda x: len(set(x.split(":"))) / 7)
    # 平均行为日期
    feat1['act_mean_date'] = feat1['act_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    del feat1['act_list']

    # 用户每天的活跃数
    user_activity_day_count = user_activity.groupby(['user_id', 'activity_day'], as_index=False)['activity_day'].agg({'user_activity_day_count': 'count'})
    # 用户每天的活跃数数的平均值
    feat6 = user_activity_day_count.groupby(['user_id'], as_index=False)['user_activity_day_count'].agg({'user_activity_day_count_mean': 'mean'})
    # 用户每天的活跃数数的max
    feat7 = user_activity_day_count.groupby(['user_id'], as_index=False)['user_activity_day_count'].agg({'user_activity_day_count_max': 'max'})
    # 用户每天的活跃数数的min
    feat8 = user_activity_day_count.groupby(['user_id'], as_index=False)['user_activity_day_count'].agg({'user_activity_day_count_min': 'min'})
    # 用户每天的活跃数数的std
    feat9 = user_activity_day_count.groupby(['user_id'], as_index=False)['user_activity_day_count'].agg({'user_activity_day_count_std': np.std})
    # 用户每天的活跃数数的众树
    feat10 = user_activity_day_count.groupby(['user_id'])['user_activity_day_count'].agg(lambda x: x.value_counts().index[0]).reset_index()
    feat10.rename(columns={"user_activity_day_count": "user_activity_day_count_mode"}, inplace=True)


    # 统计用户前3天的行为特征及其次数
    user_activity_before_three_day = user_activity[user_activity.activity_day > feature_high - 3]
    # 前三天每个用户的活跃次数
    feat11 = user_activity_before_three_day.groupby(['user_id'],as_index=False)['page'].agg({"before_three_act_count":"count"})

    feat12 = user_activity_before_three_day[['user_id', 'activity_day']]
    feat12['activity_day'] = feat12['activity_day'].astype('str')
    feat12 = feat12.groupby(['user_id'])['activity_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat12.rename(columns={'activity_day': 'before_act_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat12['before_cur_day_repeat_count'] = feat12['before_act_list'].apply(cur_day_repeat_count)
    # 用户连续活跃天数
    feat12['before_con_act_day_count'] = feat12['before_act_list'].apply(get_continue_launch_day)
    # 平均行为次数
    feat12['before_mean_act_count'] = feat12['before_act_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat12['before_act_seven_day_rate'] = feat12['before_act_list'].apply(lambda x: len(set(x.split(":"))) / 3)
    # 平均行为日期
    feat12['before_act_mean_date'] = feat12['before_act_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    # 均值
    feat12['before_act_time_gap_mean'] = feat12['before_act_list'].apply(get_time_gap, args=('1'))
    # max
    feat12['before_act_time_gap_max'] = feat12['before_act_list'].apply(get_time_gap, args=('2'))
    # min
    feat12['before_act_time_gap_min'] = feat12['before_act_list'].apply(get_time_gap, args=('3'))
    # std
    feat12['before_act_time_gap_std'] = feat12['before_act_list'].apply(get_time_gap, args=('4'))
    del feat12['before_act_list']

    # 统计用户前1天的行为特征及其次数
    user_activity_before_one_day = user_activity[user_activity.activity_day > feature_high - 1]
    # 前1天每个用户的活跃次数
    feat13 = user_activity_before_one_day.groupby(['user_id'],as_index=False)['page'].agg({"before_one_act_count":"count"})
    # 前1天是否活跃多次
    feat14 = user_activity_before_one_day.groupby(['user_id'],as_index=False)['activity_day'].agg(lambda x: 1 if len(x) > 1 else 0)


    user_id = USER_ID[['user_id']].drop_duplicates()
    data = pd.merge(user_id,user_act_count,on='user_id',how='left')
    data = pd.merge(data, user_page, on='user_id', how='left')
    data = pd.merge(data, user_action_type, on='user_id', how='left')
    data = pd.merge(data, user_page_mode, on='user_id', how='left')
    data = pd.merge(data, user_action_mode, on='user_id', how='left')
    data = pd.merge(data, user_diff_video, on='user_id', how='left')
    data = pd.merge(data, user_diff_author, on='user_id', how='left')
    data = pd.merge(data, user_is_author, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_mean, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_max, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_min, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_mode, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_skew, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_kurt, on='user_id', how='left')
    data = pd.merge(data, gap_activity_day_var, on='user_id', how='left')
    data = pd.merge(data, user_act_day_count, on='user_id', how='left')
    data = pd.merge(data, feat1, on='user_id', how='left')
    data = pd.merge(data, feat6, on='user_id', how='left')
    data = pd.merge(data, feat7, on='user_id', how='left')
    data = pd.merge(data, feat8, on='user_id', how='left')
    data = pd.merge(data, feat9, on='user_id', how='left')
    data = pd.merge(data, feat10, on='user_id', how='left')
    data = pd.merge(data, feat11, on='user_id', how='left')
    data = pd.merge(data, feat12, on='user_id', how='left')
    data = pd.merge(data, feat13, on='user_id', how='left')
    data = pd.merge(data, feat14, on='user_id', how='left')

    user_activity = data
    return user_activity


def extract_create_feature(user_create,USER_ID,feature_high):
    # 用户每天的拍摄数
    user_create_day_count = user_create.groupby(['user_id', 'create_day'], as_index=False)['create_day'].agg({'user_create_day_count': 'count'})
    # 用户每天平均拍摄次数
    feat1 = user_create_day_count.groupby(['user_id'], as_index=False)['user_create_day_count'].agg({"user_day_mean_create_count": "mean"})
    # 用户每天最大拍摄次数
    feat2 = user_create_day_count.groupby(['user_id'], as_index=False)['user_create_day_count'].agg({"user_day_max_create_count": "max"})
    # 用户每天最小拍摄次数
    feat3 = user_create_day_count.groupby(['user_id'], as_index=False)['user_create_day_count'].agg({"user_day_min_create_count": "min"})
    # 用户每天最小拍摄次数
    feat4 = user_create_day_count.groupby(['user_id'], as_index=False)['user_create_day_count'].agg({"user_day_std_create_count": np.std})
    # 用户每天的拍摄数的众树
    feat5 = user_create_day_count.groupby(['user_id'], as_index=False)['user_create_day_count'].agg(lambda x: x.value_counts().index[0])
    feat5.rename(columns={"user_create_day_count": "user_create_day_count_mode"}, inplace=True)

    # 用户拍摄天数
    feat6 = user_create.groupby(['user_id'], as_index=False)['create_day'].agg({"user_continue_create_count": 'nunique'})
    # 用户拍摄次数
    feat7 = user_create.groupby(['user_id'], as_index=False)['create_day'].agg({"user_create_count": 'count'})

    # 距离窗口末端的时间
    user_create['gap_create_day'] = feature_high - user_create['create_day']
    # 均值
    feat8 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({"gap_create_day_mean": np.mean})
    # 最大值
    feat9 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({"gap_create_day_max": np.max})
    # 最小值
    feat10 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({"gap_create_day_min": np.min})
    # 众数
    feat11 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg(lambda x: x.value_counts().index[0])
    feat11.rename(columns={"gap_create_day": "gap_create_day_mode"}, inplace=True)
    # skew
    feat12 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({'gap_create_day_skew': sp.stats.skew})
    # kurt
    feat13 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({'gap_create_day_kurt': sp.stats.kurtosis})
    # var
    feat14 = user_create.groupby(['user_id'], as_index=False)['gap_create_day'].agg({'gap_create_day_std': np.std})

    # 用户是否当天拍摄多次
    feat15 = user_create[['user_id', 'create_day']]
    feat15['create_day'] = feat15['create_day'].astype('str')
    feat15 = feat15.groupby(['user_id'])['create_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat15.rename(columns={'create_day': 'create_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat15['create_cur_day_repeat_count'] = feat15['create_list'].apply(cur_day_repeat_count)
    # 用户连续活跃天数
    feat15['create_con_act_day_count'] = feat15['create_list'].apply(get_continue_launch_day)
    # 连续几天启动次数的均值，
    feat15['con_create_day_count_mean'] = feat15['create_list'].apply(get_continue_launch_count, args=('1'))
    # 最大值，
    feat15['con_create_day_count_max'] = feat15['create_list'].apply(get_continue_launch_count, args=('2'))
    # 最小值
    feat15['con_create_day_count_min'] = feat15['create_list'].apply(get_continue_launch_count, args=('3'))
    # 次数
    feat15['con_create_day_count_total'] = feat15['create_list'].apply(get_continue_launch_count, args=('4'))
    # 方差
    feat15['con_create_day_count_std'] = feat15['create_list'].apply(get_continue_launch_count, args=('5'))


    # 均值
    feat15['create_time_gap_mean'] = feat15['create_list'].apply(get_time_gap, args=('1'))
    # max
    feat15['create_time_gap_max'] = feat15['create_list'].apply(get_time_gap, args=('2'))
    # min
    feat15['create_time_gap_min'] = feat15['create_list'].apply(get_time_gap, args=('3'))
    # std
    feat15['create_time_gap_std'] = feat15['create_list'].apply(get_time_gap, args=('4'))
    # 平均行为次数
    feat15['mean_create_count'] = feat15['create_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat15['create_seven_day_rate'] = feat15['create_list'].apply(lambda x: len(set(x.split(":"))) / 7)
    # 平均行为日期
    feat15['create_mean_date'] = feat15['create_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    del feat15['create_list']



    # 统计用户前3天的行为特征及其次数
    user_create_before_three_day = user_create[user_create.create_day > feature_high - 3]
    # 前三天每个用户的活跃次数
    feat16 = user_create_before_three_day.groupby(['user_id'],as_index=False)['create_day'].agg({"before_three_create_count": "count"})
    # 用户是否当天拍摄多次
    feat17 = user_create_before_three_day[['user_id', 'create_day']]
    feat17['create_day'] = feat17['create_day'].astype('str')
    feat17 = feat17.groupby(['user_id'])['create_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat17.rename(columns={'create_day': 'before_create_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat17['before_create_cur_day_repeat_count'] = feat17['before_create_list'].apply(cur_day_repeat_count)
    # 用户连续活跃天数
    feat17['before_create_con_act_day_count'] = feat17['before_create_list'].apply(get_continue_launch_day)
    # 平均行为次数
    feat17['before_mean_create_count'] = feat17['before_create_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat17['before_create_seven_day_rate'] = feat17['before_create_list'].apply(lambda x: len(set(x.split(":"))) / 3)
    # 平均行为日期
    feat17['before_create_mean_date'] = feat17['before_create_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    # 均值
    feat17['before_create_time_gap_mean'] = feat17['before_create_list'].apply(get_time_gap, args=('1'))
    # max
    feat17['before_create_time_gap_max'] = feat17['before_create_list'].apply(get_time_gap, args=('2'))
    # min
    feat17['before_create_time_gap_min'] = feat17['before_create_list'].apply(get_time_gap, args=('3'))
    # std
    feat17['before_create_time_gap_std'] = feat17['before_create_list'].apply(get_time_gap, args=('4'))
    del feat17['before_create_list']

    # 统计用户前1天的行为特征及其次数
    user_create_before_one_day = user_create[user_create.create_day > feature_high - 1]
    # 前1天每个用户的活跃次数
    feat18 = user_create_before_one_day.groupby(['user_id'], as_index=False)['create_day'].agg({"before_one_create_count": "count"})
    # 前1天是否活跃多次
    feat19 = user_create_before_one_day.groupby(['user_id'], as_index=False)['create_day'].agg(lambda x: 1 if len(x) > 1 else 0)


    user_id = USER_ID[['user_id']].drop_duplicates()
    user_create = pd.merge(user_id,feat1,on='user_id',how='left')
    user_create = pd.merge(user_create, feat2, on='user_id', how="left")
    user_create = pd.merge(user_create, feat3, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat4, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat5, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat6, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat7, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat8, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat9, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat10, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat11, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat12, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat13, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat14, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat15, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat16, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat17, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat18, on=['user_id'], how="left")
    user_create = pd.merge(user_create, feat19, on=['user_id'], how="left")
    user_create = user_create.drop_duplicates()

    return user_create


def extract_launch_feature(user_launch,USER_ID,feature_high):
    # 用户启动次数
    user_launch_count = user_launch.groupby(['user_id'],as_index=False)['launch_day'].agg({"user_launch_count":'count'})
    #用户活跃天数
    user_launch_day_count_sp = user_launch.groupby(['user_id'],as_index=False)['launch_day'].agg({"user_launch_day_count":'nunique'})
    user_launch_day_count_sp.drop_duplicates(subset=['user_id'])
    # 距离窗口末端的时间
    user_launch['gap_launch_day'] = feature_high - user_launch['launch_day']
    # 均值
    feat1 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({"gap_launch_day_mean": np.mean})
    # 最大值
    feat2 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({"gap_launch_day_max": np.max})
    # 最小值
    feat3 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({"gap_launch_day_min": np.min})
    # 众数
    feat4 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg(lambda x: x.value_counts().index[0])
    feat4.rename(columns={"gap_launch_day": "gap_launch_day_mode"}, inplace=True)
    # skew
    feat5 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({'gap_launch_day_skew': sp.stats.skew})
    # kurt
    feat6 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({'gap_launch_day_kurt': sp.stats.kurtosis})
    # var
    feat7 = user_launch.groupby(['user_id'], as_index=False)['gap_launch_day'].agg({'gap_launch_day_std': np.std})

    feat8 = user_launch[['user_id', 'launch_day']]
    feat8['launch_day'] = feat8['launch_day'].astype('str')
    feat8 = feat8.groupby(['user_id'])['launch_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat8.rename(columns={'launch_day': 'launh_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat8['cur_day_repeat_count_launcg'] = feat8['launh_list'].apply(cur_day_repeat_count)
    # 用户连续启动天数
    feat8['con_launch_day_count'] = feat8['launh_list'].apply(get_continue_launch_day)

    # 连续几天启动次数的均值，
    feat8['con_launch_day_count_mean'] = feat8['launh_list'].apply(get_continue_launch_count,args=('1'))
    # 最大值，
    feat8['con_launch_day_count_max'] = feat8['launh_list'].apply(get_continue_launch_count, args=('2'))
    # 最小值
    feat8['con_launch_day_count_min'] = feat8['launh_list'].apply(get_continue_launch_count, args=('3'))
    # 次数
    feat8['con_launch_day_count_total'] = feat8['launh_list'].apply(get_continue_launch_count, args=('4'))
    # 方差
    feat8['con_launch_day_count_std'] = feat8['launh_list'].apply(get_continue_launch_count, args=('5'))

    # 均值
    feat8['launch_time_gap_mean'] = feat8['launh_list'].apply(get_time_gap, args=('1'))
    # max
    feat8['launch_time_gap_max'] = feat8['launh_list'].apply(get_time_gap, args=('2'))
    # min
    feat8['launch_time_gap_min'] = feat8['launh_list'].apply(get_time_gap, args=('3'))
    # std
    feat8['launch_time_gap_std'] = feat8['launh_list'].apply(get_time_gap, args=('4'))

    # 平均行为次数
    feat8['mean_launcht_count'] = feat8['launh_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat8['launch_seven_day_rate'] = feat8['launh_list'].apply(lambda x: len(set(x.split(":"))) / 7)
    # 平均行为日期
    feat8['launch_mean_date'] = feat8['launh_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))

    del feat8['launh_list']

    # 用户每天的启动数
    user_launch_day_count = user_launch.groupby(['user_id', 'launch_day'], as_index=False)['launch_day'].agg({'user_launch_day_count': 'count'})
    # 用户每天的活跃数数的平均值
    feat13 = user_launch_day_count.groupby(['user_id'], as_index=False)['user_launch_day_count'].agg({'user_launch_day_count_mean': 'mean'})
    # 用户每天的活跃数数的max
    feat14 = user_launch_day_count.groupby(['user_id'], as_index=False)['user_launch_day_count'].agg({'user_launch_day_count_max': 'max'})
    # 用户每天的活跃数数的min
    feat15 = user_launch_day_count.groupby(['user_id'], as_index=False)['user_launch_day_count'].agg({'user_launch_day_count_min': 'min'})
    # 用户每天的活跃数数的std
    feat16 = user_launch_day_count.groupby(['user_id'], as_index=False)['user_launch_day_count'].agg({'user_launch_day_count_std': np.std})
    # 用户每天的活跃数数的众树
    feat17 = user_launch_day_count.groupby(['user_id'])['user_launch_day_count'].agg(lambda x: x.value_counts().index[0]).reset_index()
    feat17.rename(columns={"user_launch_day_count": "user_launch_day_count_mode"}, inplace=True)



    # 统计用户前3天的行为特征及其次数
    user_launch_before_three_day = user_launch[user_launch.launch_day > feature_high - 3]

    # 前三天每个用户的活跃次数
    feat18 = user_launch_before_three_day.groupby(['user_id'],as_index=False)['launch_day'].agg({"before_three_launch_count": "count"})

    # 用户是否当天拍摄多次
    feat19 = user_launch_before_three_day[['user_id', 'launch_day']]
    feat19['launch_day'] = feat19['launch_day'].astype('str')
    feat19 = feat19.groupby(['user_id'])['launch_day'].agg(lambda x: ':'.join(x)).reset_index()
    feat19.rename(columns={'launch_day': 'before_launch_list'}, inplace=True)
    # 用户是否多天多次启动
    # 用户是否当天多次启动
    feat19['before_launch_cur_day_repeat_count'] = feat19['before_launch_list'].apply(cur_day_repeat_count)
    # 用户连续活跃天数
    feat19['before_launch_con_act_day_count'] = feat19['before_launch_list'].apply(get_continue_launch_day)
    # 平均行为次数
    feat19['before_mean_launch_count'] = feat19['before_launch_list'].apply(lambda x: len(x.split(":")) / len(set(x.split(":"))))
    # 在7天时间内的次数
    feat19['before_launch_seven_day_rate'] = feat19['before_launch_list'].apply(lambda x: len(set(x.split(":"))) / 3)
    # 平均行为日期
    feat19['before_launch_mean_date'] = feat19['before_launch_list'].apply(lambda x: np.sum([int(ele) for ele in x.split(":")]) / len(x.split(":")))
    # 均值
    feat19['before_launch_time_gap_mean'] = feat19['before_launch_list'].apply(get_time_gap, args=('1'))
    # max
    feat19['before_launch_time_gap_max'] = feat19['before_launch_list'].apply(get_time_gap, args=('2'))
    # min
    feat19['before_launch_time_gap_min'] = feat19['before_launch_list'].apply(get_time_gap, args=('3'))
    # std
    feat19['before_launch_time_gap_std'] = feat19['before_launch_list'].apply(get_time_gap, args=('4'))
    del feat19['before_launch_list']

    # 统计用户前1天的行为特征及其次数
    user_launch_before_one_day = user_launch[user_launch.launch_day > feature_high - 1]
    # 前1天每个用户的活跃次数
    feat20 = user_launch_before_one_day.groupby(['user_id'], as_index=False)['launch_day'].agg({"before_one_launch_count": "count"})
    # 前1天是否活跃多次
    feat21 = user_launch_before_one_day.groupby(['user_id'], as_index=False)['launch_day'].agg(lambda x: 1 if len(x) > 1 else 0)


    user_id = USER_ID[['user_id']].drop_duplicates()
    data = pd.merge(user_id,user_launch_count,on='user_id',how='left')
    data = pd.merge(data,user_launch_day_count_sp,on='user_id',how='left')
    data = pd.merge(data, feat1, on='user_id', how='left')
    data = pd.merge(data, feat2, on='user_id', how='left')
    data = pd.merge(data, feat3, on='user_id', how='left')
    data = pd.merge(data, feat4, on='user_id', how='left')
    data = pd.merge(data, feat5, on='user_id', how='left')
    data = pd.merge(data, feat6, on='user_id', how='left')
    data = pd.merge(data, feat7, on='user_id', how='left')
    data = pd.merge(data, feat8, on='user_id', how='left')
    data = pd.merge(data, feat13, on='user_id', how='left')
    data = pd.merge(data, feat14, on='user_id', how='left')
    data = pd.merge(data, feat15, on='user_id', how='left')
    data = pd.merge(data, feat16, on='user_id', how='left')
    data = pd.merge(data, feat17, on='user_id', how='left')
    data = pd.merge(data, feat18, on='user_id', how='left')
    data = pd.merge(data, feat19, on='user_id', how='left')
    data = pd.merge(data, feat20, on='user_id', how='left')
    data = pd.merge(data, feat21, on='user_id', how='left')


    user_launch = data
    return user_launch


def extract_register_feature(user_register,USER_ID,feature_high):
    device_type_count = user_register.groupby(['device_type'], as_index=False)['register_day'].agg({'device_type_count':'count'})
    device_type_mode = user_register.groupby(['user_id'], as_index=False)['device_type'].agg(lambda x:x.value_counts().index[0])
    device_type_mode.rename(columns={"device_type": "device_type_mode"}, inplace=True)

    # 注册日期距离窗口末端的时间
    user_register['gap_reg_day'] = feature_high - user_register['register_day']

    # 均值
    feat1 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({"gap_reg_day_mean": np.mean})
    # 最大值
    feat2 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({"gap_reg_day_max": np.max})
    # 最小值
    feat3 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({"gap_reg_day_min": np.min})
    # skew
    feat4 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({'gap_reg_day_skew': sp.stats.skew})
    # kurt
    feat5 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({'gap_reg_day_kurt': sp.stats.kurtosis})
    # std
    feat6 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg({'gap_reg_day_std': np.std})
    # 众数
    feat7 = user_register.groupby(['user_id'], as_index=False)['gap_reg_day'].agg(lambda x: x.value_counts().index[0])
    feat7.rename(columns={"gap_reg_day": "gap_reg_day_mode"}, inplace=True)
    del user_register['gap_reg_day']

    user_register = user_register.drop_duplicates()
    user_register = pd.merge(user_register, device_type_count, on='device_type', how="left")
    user_register = pd.merge(user_register, device_type_mode, on='user_id', how="left")
    user_register = pd.merge(user_register, feat1, on='user_id', how="left")
    user_register = pd.merge(user_register, feat2, on='user_id', how="left")
    user_register = pd.merge(user_register, feat3, on='user_id', how="left")
    user_register = pd.merge(user_register, feat4, on='user_id', how="left")
    user_register = pd.merge(user_register, feat5, on='user_id', how="left")
    user_register = pd.merge(user_register, feat6, on='user_id', how="left")
    user_register = pd.merge(user_register, feat7, on='user_id', how="left")

    return user_register



def split_data_extact_feature():
    print('开始提取特征dgdg！！')
    user_reg = analysis_register()
    user_launch = analysis_launch()
    user_create = analysis_create_log()
    user_activity = analysis_activity()

    user_reg = pd.get_dummies(user_reg, columns=['register_type'], prefix='register_type', prefix_sep='_')
    train1 = data_range_train_one(user_reg,user_launch,user_create,user_activity)
    print("划分1结束")
    train2 = data_range_train_two(user_reg, user_launch, user_create, user_activity)
    print("划分2结束")
    test = data_range_test(user_reg, user_launch, user_create, user_activity)
    print("测试集划分结束")

    return train1,train2,test
 


def load_csv(train1,train2,test):
    train = pd.concat([train1, train2], axis=0)

    train = train.drop_duplicates()
    test = test.drop_duplicates()

    print(len(train))
    print(len(test))

    train.to_csv('/home/kesci/train.txt', sep=',', index=None)
    test.to_csv('/home/kesci/test.txt', sep=',', index=None)
    # train.to_csv('train.txt', sep=',', index=None)
    # test.to_csv('test.txt', sep=',', index=None)



def main():
    train1,train2,test = split_data_extact_feature()
    load_csv(train1,train2,test)



if __name__ == '__main__':
    main()