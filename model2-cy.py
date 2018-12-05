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
import scipy.stats as sp
print('sleeping.....')
# time.sleep(7200)
print('start...')
def get_av_time_dis(x):#apply

    x=x.str_day
    if x!=x:
        x='-1'
    # print(x)
    day = x.split(':')
    day = list(set(day))
    day = list(map(lambda x:float(x),day))
    day.sort()
    if day is None or len(day) == 0:
        return 0
    m={}
    res = 0
    for i in day:
        if i not in m:
            l=0
            r=0
            if i-1 in m:
                l = m[i-1]
            if i+1 in m:
                r = m[i+1]
            m[i] = 1+r+l
            m[i+r] = 1+r+l
            m[i-l] = 1+r+l
            res = max(res,m[i])
    return res

def mark_new():
    print('mark_new...')
    print('读取各个表。。')
    register = pd.read_csv('user_register_log.txt', sep='\t', header=None)
    register.columns = ['user_id', 'register_day', 'register_type', 'device_type']
    launch = pd.read_csv('app_launch_log.txt', sep='\t', header=None)
    launch.columns = ['user_id', 'day']
    create = pd.read_csv('video_create_log.txt', sep='\t', header=None)
    create.columns = ['user_id', 'day']
    activity = pd.read_csv('user_activity_log.txt', sep='\t', header=None)
    activity.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
    result = None
    for today in range(1,24):#只取1-23号用于打标，因为用后7天打标
        print('标第%s天'%today)
        activity_user_list = activity[(activity.day>today)&(activity.day<=(today+7))]
        activity_user_list = activity_user_list.drop_duplicates(['user_id'])  # 去重
        activity_user_list = activity_user_list.user_id.values

        launch_user_list = launch[(launch.day>today)&(launch.day<=(today+7))]
        launch_user_list = launch_user_list.drop_duplicates(['user_id'])
        launch_user_list = launch_user_list.user_id.values

        create_user_list = create[(create.day > today) & (create.day <= (today + 7))]
        create_user_list = create_user_list.drop_duplicates(['user_id'])
        create_user_list = create_user_list.user_id.values

        user_list_register = register[register.register_day<=today]
        user_list_register['day'] = today
        user_list_register['label'] = list(map(lambda x:1 if x in activity_user_list or x in launch_user_list or x in create_user_list else 0,user_list_register.user_id))
        if result is None:
            result=user_list_register
        else:
            frames = [result, user_list_register]  # 添加当日数据到结果集
            result = pd.concat(frames)
    result.to_csv('label_table.csv', encoding='utf-8', index=None)
    print(result)


#数据划分并提取特征（训练集，均等滑窗）
def split_F_train():
    print('划分并提取。。')
    print('读取各个表。。')
    register = pd.read_csv('user_register_log.txt', sep='\t', header=None)
    register.columns = ['user_id', 'register_day', 'register_type', 'device_type']
    launch = pd.read_csv('app_launch_log.txt', sep='\t', header=None)
    launch.columns = ['user_id', 'day']
    create = pd.read_csv('video_create_log.txt', sep='\t', header=None)
    create.columns = ['user_id', 'day']
    activity = pd.read_csv('user_activity_log.txt', sep='\t', header=None)
    activity.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']

    label_table = pd.read_csv(r'label_table.csv')

    train = None
    for today in [14,15,16,17,18,19,20,21,22,23,30]:#24 为测试集,特征区间长度都为7天
        print('day:%s'%today)
        activity_region = activity[(activity.day > (today - 13)) & (activity.day <= (today))]
        launch_region = launch[(launch.day > (today - 13)) & (launch.day <= (today))]
        create_region = create[(create.day > (today - 13)) & (create.day <= (today))]
        result =None
        if today<24:
            result = label_table[label_table.day==today]#标签区间

        elif today==30:
            result = register[register.register_day<=today]#标签区间
            result['day'] = today

###############################################################################################################
        #   提取历史特征
        print('提取历史特征。。')
        #         1、该用户历史的总启动次数
        temp = launch_region.groupby(['user_id'])['day'].agg({'h1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         2、该用户历史的总拍摄次数
        temp = create_region.groupby(['user_id'])['day'].agg({'h2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         3、该用户历史有多少天启动了
        temp = launch_region.drop_duplicates(['user_id', 'day'])  # 去重
        temp = temp.groupby(['user_id'])['day'].agg({'h3': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # result.f3 = result.f3.fillna(0)
        #         4、该用户历史有多少天拍摄了
        temp = create_region.drop_duplicates(['user_id', 'day'])  # 去重
        temp = temp.groupby(['user_id'])['day'].agg({'h4': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # result.f4 = result.f4.fillna(0)
        #         5、该用户历史行为次数
        temp = activity_region.groupby(['user_id'])['action_type'].agg({'h5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # crosstab action_type
        activity_region['action_type_new1'] = list(map(lambda x:'action_'+str(x),activity_region.action_type))
        temp = pd.crosstab(activity_region.user_id, activity_region.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region['action_type_new1']
        # crosstab page
        activity_region['page_new1'] = list(map(lambda x: 'page' + str(x), activity_region.page))
        temp = pd.crosstab(activity_region.user_id, activity_region.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region['page_new1']

        # # crosstab video_id
        temp = activity_region.groupby(['video_id'])['action_type'].agg({'video_sum': np.size}).reset_index()  #
        top20_video = temp.sort_values(["video_sum"], ascending=False).head(200)
        top20_video = top20_video.video_id.values
        temp = activity_region[activity_region['video_id'].isin(top20_video)]
        temp['video_id1'] = list(map(lambda x: 'video' + str(x), temp.video_id))
        temp = pd.crosstab(temp.user_id, temp.video_id1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # crosstab author_id
        temp = activity_region.groupby(['author_id'])['action_type'].agg({'author_sum': np.size}).reset_index()  #
        top20_author = temp.sort_values(["author_sum"], ascending=False).head(200)
        top20_author = top20_author.author_id.values
        temp = activity_region[activity_region['author_id'].isin(top20_author)]
        temp['author_id1'] = list(map(lambda x: 'AUTHOR' + str(x), temp.author_id))
        temp = pd.crosstab(temp.user_id, temp.author_id1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #



        #         30、该用户历史上总被浏览多少次
        temp = activity_region.groupby(['author_id'])['day'].agg({'h30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region['action_type_new2'] = list(map(lambda x: 'action2_' + str(x), activity_region.action_type))
        temp = pd.crosstab(activity_region.author_id, activity_region.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'],how='left')
        del activity_region['action_type_new2']
        #         37、该用户到今天为止已经注册了多少天了
        result['h37'] = result['day'] - result['register_day']
        result['h37'] = result['h37']+1
        result['h37_1'] = list(map(lambda x:1 if x>7 else 0,result.h37))

        # max continue len day
        temp = activity_region[['user_id', 'day']]
        temp['day'] = temp['day'].astype('str')
        temp = temp.groupby(['user_id'])['day'].agg(lambda x: ':'.join(x)).reset_index()
        temp = temp.drop_duplicates(['user_id', 'day'])  # 去重
        temp.rename(columns={'day': 'str_day'}, inplace=True)
        temp['max_continue_day'] = temp.apply(get_av_time_dis, axis=1)#apply
        temp = temp[['user_id','max_continue_day']]
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #该用户历史有多少天action了
        temp = activity_region[activity_region.action_type >= 0]
        temp = temp.drop_duplicates(['user_id', 'day'])  # 去重
        temp = temp.groupby(['user_id'])['day'].agg({'h40': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # h40/7
        result['h41'] = result['h40']/7
        # week crosstab
        activity_region['week'] = list(map(lambda x:x%7,activity_region.day))
        temp = pd.crosstab(activity_region.user_id, activity_region.week).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')



        # average
        result['av1'] = result['h1']/7
        result['av2'] = result['h2']/7
        result['av3'] = result['h3']/7
        result['av4'] = result['h4']/7
        result['av5'] = result['h5']/7
        result['av30'] = result['h30']/7


        # std

        #该用户历史的拍摄次数std
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_num': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_num'].agg({'h43': np.std}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #该用户历史行为次数std
        temp = activity_region.groupby(['user_id','day'])['action_type'].agg({'today_action_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_action_sum'].agg({'h44': np.std}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #该用户历史上被浏览多少次
        temp = activity_region.groupby(['author_id'])['day'].agg({'today_author_sum': np.size}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # 该用户历史的拍摄次数mean
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h47': np.mean}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数mean
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg({'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h48': np.mean}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次mean
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg({'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h49': np.mean}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # min
        # 该用户历史的拍摄次数min
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h50': np.min}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数min
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg({'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h51': np.min}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次min
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h52': np.min}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # max
        # 该用户历史的拍摄次数max
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h53': np.max}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数max
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg(
            {'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h54': np.max}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次max
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h55': np.max}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # median
        # 该用户历史的拍摄次数median
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h56': np.median}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数median
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg(
            {'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h57': np.median}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次median
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h58': np.median}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # var
        # 该用户历史的拍摄次数var
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h59': np.var}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数var
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg(
            {'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h60': np.var}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次var
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h61': np.var}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # skew  sp.stats.skew
        # 该用户历史的拍摄次数var
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h62': sp.stats.skew}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数var
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg(
            {'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h63': sp.stats.skew}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次var
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h64': sp.stats.skew}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #


        # kurt sp.stats.kurtosis
        # 该用户历史的拍摄次数var
        temp = create_region.groupby(['user_id', 'day'])['user_id'].agg({'today_create_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_create_sum'].agg({'h65': sp.stats.kurtosis}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史行为次数var
        temp = activity_region.groupby(['user_id', 'day'])['user_id'].agg(
            {'today_activity_sum': np.size}).reset_index()  #
        temp = temp.groupby(['user_id'])['today_activity_sum'].agg({'h66': sp.stats.kurtosis}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 该用户历史上被浏览多少次var
        temp = activity_region.groupby(['author_id', 'day'])['user_id'].agg(
            {'today_author_sum': np.size}).reset_index()  #
        temp = temp.groupby(['author_id'])['today_author_sum'].agg({'h67': sp.stats.kurtosis}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
#######################################################################################################################
        # 100、该用户的注册日期是周几
        result['regis_week'] = list(map(lambda x: x % 7, result.register_day))
        # 101、该用户在哪个星期最活跃
        temp = activity_region.groupby(['user_id', 'week'])['user_id'].agg({'temp1': np.size}).reset_index()  #
        temp2 = temp.groupby(['user_id'])['temp1'].agg({'temp2': np.max}).reset_index()  #
        temp = pd.merge(temp, temp2, on=['user_id'], how='left')  #
        temp = temp[temp.temp1==temp.temp2]
        temp = temp.drop_duplicates(['user_id'])
        temp = temp[['user_id','week']]
        temp.rename(columns={'week': 'most_week'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 102、该用户的平均，最大，最小，拍摄时间间隔
        temp = activity_region.drop_duplicates(['user_id', 'day'])
        temp = temp.sort_values(['day'], ascending=True)
        temp['next_time'] = temp.groupby(['user_id'])['day'].diff(1)
        temp2 = temp
        # average
        temp = temp.groupby(['user_id'])['next_time'].agg({'avg_time': np.mean})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     max
        temp = temp2.groupby(['user_id'])['next_time'].agg({'max_time': np.max})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     min
        temp = temp2.groupby(['user_id'])['next_time'].agg({'min_time': np.min})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     std
        temp = temp2.groupby(['user_id'])['next_time'].agg({'std_time': np.std})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     var
        temp = temp2.groupby(['user_id'])['next_time'].agg({'var_time': np.var})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #

        # 104、该用户的这种来源渠道在历史区间的活跃次数、拍摄次数
        temp = pd.merge(activity_region, register, on=['user_id'], how='left')  #
        temp = temp.groupby(['register_type'])['user_id'].agg({'regis_sum': np.size}).reset_index()
        result = pd.merge(result, temp, on=['register_type'], how='left')  #
        # 105、该用户这种设备类型在历史区间的活跃次数、拍摄次数
        temp = pd.merge(activity_region, register, on=['user_id'], how='left')  #
        temp = temp.groupby(['device_type'])['user_id'].agg({'device_sum': np.size}).reset_index()
        result = pd.merge(result, temp, on=['device_type'], how='left')  #
        # 106、该用户访问最多的page
        temp = activity_region.groupby(['user_id', 'page'])['user_id'].agg({'temp1': np.size}).reset_index()  #
        temp2 = temp.groupby(['user_id'])['temp1'].agg({'temp2': np.max}).reset_index()  #
        temp = pd.merge(temp, temp2, on=['user_id'], how='left')  #
        temp = temp[temp.temp1 == temp.temp2]
        temp = temp.drop_duplicates(['user_id'])
        temp = temp[['user_id', 'page']]
        temp.rename(columns={'page': 'most_page'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 107、该用户访问最多的action_type
        temp = activity_region.groupby(['user_id', 'action_type'])['user_id'].agg({'temp1': np.size}).reset_index()  #
        temp2 = temp.groupby(['user_id'])['temp1'].agg({'temp2': np.max}).reset_index()  #
        temp = pd.merge(temp, temp2, on=['user_id'], how='left')  #
        temp = temp[temp.temp1 == temp.temp2]
        temp = temp.drop_duplicates(['user_id'])
        temp = temp[['user_id', 'action_type']]
        temp.rename(columns={'action_type': 'most_action_type'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 108、对历史区间video和author word2vec，然后按照用户所有的访问记录叠加生成他的词向量

        # 109、对当天提取统计特征###########################################################################################
        activity_region_today = activity_region[activity_region.day == today]
        launch_region_today = launch_region[launch_region.day == today]
        create_region_today = create_region[create_region.day == today]
        #         1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'today_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'today_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'today_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action3_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page2' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'today_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action4_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        # 110、前一周的统计值，第二周的统计值，两周的差值，
        # 对1week提取统计特征###########################################################################################
        activity_region_today = activity_region[
            (activity_region.day <= (today - 7)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 7)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 7)) & (create_region.day > (today - 14))]
        #         1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'first_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'first_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'first_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action13_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page12' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'first_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action14_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']
        # 对2week提取统计特征###########################################################################################
        activity_region_today = activity_region[
            (activity_region.day <=today) & (activity_region.day > (today -7))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 7))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 7))]
        #         1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'two_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'two_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #         5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'two_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action23_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page22' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'two_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action24_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']
        #week2-week1#####################################################################
        result['week_sub1'] = result['two_1']-result['first_1']
        result['week_sub2'] = result['two_2'] - result['first_2']
        result['week_sub5'] = result['two_5'] - result['first_5']
        result['week_sub30'] = result['two_30'] - result['first_30']
        result['action31'] = result['action23_0'] - result['action13_0']
        result['action32'] = result['action23_1'] - result['action13_1']
        result['action33'] = result['action23_2'] - result['action13_2']
        result['action34'] = result['action23_3'] - result['action13_3']
        result['action35'] = result['action23_4'] - result['action13_4']
        result['action36'] = result['action23_5'] - result['action13_5']

        result['action41'] = result['action24_0'] - result['action14_0']
        result['action42'] = result['action24_1'] - result['action14_1']
        result['action43'] = result['action24_2'] - result['action14_2']
        result['action44'] = result['action24_3'] - result['action14_3']
        result['action45'] = result['action24_4'] - result['action14_4']
        result['action46'] = result['action24_5'] - result['action14_5']

        result['page41'] = result['page220'] - result['page120']
        result['page42'] = result['page221'] - result['page121']
        result['page43'] = result['page222'] - result['page122']
        result['page44'] = result['page223'] - result['page123']
        result['page45'] = result['page224'] - result['page124']


        # 111、该用户在历史区间既操作也拍摄的天数
        acti_user_list = activity_region.user_id.values
        create_user_list = create_region.user_id.values
        result['is_create_action'] = list(map(lambda x:1 if x in acti_user_list and x in create_user_list else 0,result.user_id))

        # ###########################################################################################
        # 112、该用户的各种行为次数对该用户的所有行为的占比
        result['pro_user_action_0'] = result['action_0'] / result['h5']
        result['pro_user_action_1'] = result['action_1'] / result['h5']
        result['pro_user_action_2'] = result['action_2'] / result['h5']
        result['pro_user_action_3'] = result['action_3'] / result['h5']
        result['pro_user_action_4'] = result['action_4'] / result['h5']
        result['pro_user_action_5'] = result['action_5'] / result['h5']
        # 113、该用户在各个page的行为次数对该用户的所有行为的占比
        result['pro_user_page_0'] = result['page0'] / result['h5']
        result['pro_user_page_1'] = result['page1'] / result['h5']
        result['pro_user_page_2'] = result['page2'] / result['h5']
        result['pro_user_page_3'] = result['page3'] / result['h5']
        result['pro_user_page_4'] = result['page4'] / result['h5']
        # # 114、该用户在各个week的行为次数对该用户的所有行为的占比
        # result['pro_user_week_0'] = result['0'] / result['h5']
        # result['pro_user_week_1'] = result['1'] / result['h5']
        # result['pro_user_week_2'] = result['2'] / result['h5']
        # result['pro_user_week_3'] = result['3'] / result['h5']
        # result['pro_user_week_4'] = result['4'] / result['h5']
        # result['pro_user_week_5'] = result['5'] / result['h5']
        # result['pro_user_week_6'] = result['6'] / result['h5']
##############################################################################################################################
        # 115、2天累加从前到后，7个区间，提取关于用户的统计特征，作差，差的平均，方差，标准差
        # 1 Fregion
        activity_region_today = activity_region[(activity_region.day <= (today-12)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today-12)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today-12)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'1_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'1_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'1_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action123_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page122' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'1_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action124_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 2 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 10)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 10)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 10)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'2_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'2_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'2_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action223_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page222' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'2_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(map(lambda x: 'action224_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 3 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 8)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 8)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 8)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'3_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'3_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'3_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action323_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page322' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'3_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(map(lambda x: 'action324_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 4 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 6)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 6)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 6)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'4_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'4_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'4_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action423_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page422' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'4_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action424_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']
        ###############################
        # 5 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 4)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 4)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 4)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'5_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'5_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'5_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action523_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page522' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'5_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'action524_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 6 Fregion
        activity_region_today = activity_region[(activity_region.day <= (today - 2)) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= (today - 2)) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= (today - 2)) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'6_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'6_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'6_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action623_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page622' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'6_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(map(lambda x: 'action624_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 7 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 14))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 14))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 14))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'7_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'7_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'7_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'action723_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'page722' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'7_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(map(lambda x: 'action724_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        # zuo cha 2-1
        result['2_1_sub_1'] = result['2_1']-result['1_1']
        result['2_1_sub_2'] = result['2_2']-result['1_2']
        result['2_1_sub_5'] = result['2_5'] - result['1_5']
        result['2_1_sub_30'] = result['2_30'] - result['1_30']
        # result['2_1_sub_action1_0'] = result['action223_0'] - result['action123_0']
        # result['2_1_sub_action1_1'] = result['action223_1'] - result['action123_1']
        # result['2_1_sub_action1_2'] = result['action223_2'] - result['action123_2']
        # result['2_1_sub_action1_3'] = result['action223_3'] - result['action123_3']
        # result['2_1_sub_action1_4'] = result['action223_4'] - result['action123_4']
        # result['2_1_sub_action1_5'] = result['action223_5'] - result['action123_5']
        #
        # result['2_1_sub_action2_0'] = result['action224_0'] - result['action124_0']
        # result['2_1_sub_action2_1'] = result['action224_1'] - result['action124_1']
        # result['2_1_sub_action2_2'] = result['action224_2'] - result['action124_2']
        # result['2_1_sub_action2_3'] = result['action224_3'] - result['action124_3']
        # result['2_1_sub_action2_4'] = result['action224_4'] - result['action124_4']
        # result['2_1_sub_action2_5'] = result['action224_5'] - result['action124_5']
        #
        # result['2_1_sub_page_0'] = result['page2220'] - result['page1220']
        # result['2_1_sub_page_1'] = result['page2221'] - result['page1221']
        # result['2_1_sub_page_2'] = result['page2222'] - result['page1222']
        # result['2_1_sub_page_3'] = result['page2223'] - result['page1223']
        # result['2_1_sub_page_4'] = result['page2224'] - result['page1224']

        # zuo cha 3-2
        result['3_2_sub_1'] = result['3_1'] - result['2_1']
        result['3_2_sub_2'] = result['3_2'] - result['2_2']
        result['3_2_sub_5'] = result['3_5'] - result['2_5']
        result['3_2_sub_30'] = result['3_30'] - result['2_30']
        # result['3_2_sub_action1_0'] = result['action323_0'] - result['action223_0']
        # result['3_2_sub_action1_1'] = result['action323_1'] - result['action223_1']
        # result['3_2_sub_action1_2'] = result['action323_2'] - result['action223_2']
        # result['3_2_sub_action1_3'] = result['action323_3'] - result['action223_3']
        # result['3_2_sub_action1_4'] = result['action323_4'] - result['action223_4']
        # result['3_2_sub_action1_5'] = result['action323_5'] - result['action223_5']
        #
        # result['3_2_sub_action2_0'] = result['action324_0'] - result['action224_0']
        # result['3_2_sub_action2_1'] = result['action324_1'] - result['action224_1']
        # result['3_2_sub_action2_2'] = result['action324_2'] - result['action224_2']
        # result['3_2_sub_action2_3'] = result['action324_3'] - result['action224_3']
        # result['3_2_sub_action2_4'] = result['action324_4'] - result['action224_4']
        # result['3_2_sub_action2_5'] = result['action324_5'] - result['action224_5']
        #
        # result['3_2_sub_page_0'] = result['page3220'] - result['page2220']
        # result['3_2_sub_page_1'] = result['page3221'] - result['page2221']
        # result['3_2_sub_page_2'] = result['page3222'] - result['page2222']
        # result['3_2_sub_page_3'] = result['page3223'] - result['page2223']
        # result['3_2_sub_page_4'] = result['page3224'] - result['page2224']

        # zuo cha 4-3
        result['4_3_sub_1'] = result['4_1'] - result['3_1']
        result['4_3_sub_2'] = result['4_2'] - result['3_2']
        result['4_3_sub_5'] = result['4_5'] - result['3_5']
        result['4_3_sub_30'] = result['4_30'] - result['3_30']
        # result['4_3_sub_action1_0'] = result['action423_0'] - result['action323_0']
        # result['4_3_sub_action1_1'] = result['action423_1'] - result['action323_1']
        # result['4_3_sub_action1_2'] = result['action423_2'] - result['action323_2']
        # result['4_3_sub_action1_3'] = result['action423_3'] - result['action323_3']
        # result['4_3_sub_action1_4'] = result['action423_4'] - result['action323_4']
        # result['4_3_sub_action1_5'] = result['action423_5'] - result['action323_5']
        #
        # result['4_3_sub_action2_0'] = result['action424_0'] - result['action324_0']
        # result['4_3_sub_action2_1'] = result['action424_1'] - result['action324_1']
        # result['4_3_sub_action2_2'] = result['action424_2'] - result['action324_2']
        # result['4_3_sub_action2_3'] = result['action424_3'] - result['action324_3']
        # result['4_3_sub_action2_4'] = result['action424_4'] - result['action324_4']
        # result['4_3_sub_action2_5'] = result['action424_5'] - result['action324_5']
        #
        # result['4_3_sub_page_0'] = result['page4220'] - result['page3220']
        # result['4_3_sub_page_1'] = result['page4221'] - result['page3221']
        # result['4_3_sub_page_2'] = result['page4222'] - result['page3222']
        # result['4_3_sub_page_3'] = result['page4223'] - result['page3223']
        # result['4_3_sub_page_4'] = result['page4224'] - result['page3224']

        # zuo cha 5-4
        result['5_4_sub_1'] = result['5_1'] - result['4_1']
        result['5_4_sub_2'] = result['5_2'] - result['4_2']
        result['5_4_sub_5'] = result['5_5'] - result['4_5']
        result['5_4_sub_30'] = result['5_30'] - result['4_30']
        # result['5_4_sub_action1_0'] = result['action523_0'] - result['action423_0']
        # result['5_4_sub_action1_1'] = result['action523_1'] - result['action423_1']
        # result['5_4_sub_action1_2'] = result['action523_2'] - result['action423_2']
        # result['5_4_sub_action1_3'] = result['action523_3'] - result['action423_3']
        # result['5_4_sub_action1_4'] = result['action523_4'] - result['action423_4']
        # result['5_4_sub_action1_5'] = result['action523_5'] - result['action423_5']
        #
        # result['5_4_sub_action2_0'] = result['action524_0'] - result['action424_0']
        # result['5_4_sub_action2_1'] = result['action524_1'] - result['action424_1']
        # result['5_4_sub_action2_2'] = result['action524_2'] - result['action424_2']
        # result['5_4_sub_action2_3'] = result['action524_3'] - result['action424_3']
        # result['5_4_sub_action2_4'] = result['action524_4'] - result['action424_4']
        # result['5_4_sub_action2_5'] = result['action524_5'] - result['action424_5']
        #
        # result['5_4_sub_page_0'] = result['page5220'] - result['page4220']
        # result['5_4_sub_page_1'] = result['page5221'] - result['page4221']
        # result['5_4_sub_page_2'] = result['page5222'] - result['page4222']
        # result['5_4_sub_page_3'] = result['page5223'] - result['page4223']
        # result['5_4_sub_page_4'] = result['page5224'] - result['page4224']

        # zuo cha 6-5
        result['6_5_sub_1'] = result['6_1'] - result['5_1']
        result['6_5_sub_2'] = result['6_2'] - result['5_2']
        result['6_5_sub_5'] = result['6_5'] - result['5_5']
        result['6_5_sub_30'] = result['6_30'] - result['5_30']
        # result['6_5_sub_action1_0'] = result['action623_0'] - result['action523_0']
        # result['6_5_sub_action1_1'] = result['action623_1'] - result['action523_1']
        # result['6_5_sub_action1_2'] = result['action623_2'] - result['action523_2']
        # result['6_5_sub_action1_3'] = result['action623_3'] - result['action523_3']
        # result['6_5_sub_action1_4'] = result['action623_4'] - result['action523_4']
        # result['6_5_sub_action1_5'] = result['action623_5'] - result['action523_5']
        #
        # result['6_5_sub_action2_0'] = result['action624_0'] - result['action524_0']
        # result['6_5_sub_action2_1'] = result['action624_1'] - result['action524_1']
        # result['6_5_sub_action2_2'] = result['action624_2'] - result['action524_2']
        # result['6_5_sub_action2_3'] = result['action624_3'] - result['action524_3']
        # result['6_5_sub_action2_4'] = result['action624_4'] - result['action524_4']
        # result['6_5_sub_action2_5'] = result['action624_5'] - result['action524_5']
        #
        # result['6_5_sub_page_0'] = result['page6220'] - result['page5220']
        # result['6_5_sub_page_1'] = result['page6221'] - result['page5221']
        # result['6_5_sub_page_2'] = result['page6222'] - result['page5222']
        # result['6_5_sub_page_3'] = result['page6223'] - result['page5223']
        # result['6_5_sub_page_4'] = result['page6224'] - result['page5224']

        # zuo cha 7-6
        result['7_6_sub_1'] = result['7_1'] - result['6_1']
        result['7_6_sub_2'] = result['7_2'] - result['6_2']
        result['7_6_sub_5'] = result['7_5'] - result['6_5']
        result['7_6_sub_30'] = result['7_30'] - result['6_30']
        # result['7_6_sub_action1_0'] = result['action723_0'] - result['action623_0']
        # result['7_6_sub_action1_1'] = result['action723_1'] - result['action623_1']
        # result['7_6_sub_action1_2'] = result['action723_2'] - result['action623_2']
        # result['7_6_sub_action1_3'] = result['action723_3'] - result['action623_3']
        # result['7_6_sub_action1_4'] = result['action723_4'] - result['action623_4']
        # result['7_6_sub_action1_5'] = result['action723_5'] - result['action623_5']
        #
        # result['7_6_sub_action2_0'] = result['action724_0'] - result['action624_0']
        # result['7_6_sub_action2_1'] = result['action724_1'] - result['action624_1']
        # result['7_6_sub_action2_2'] = result['action724_2'] - result['action624_2']
        # result['7_6_sub_action2_3'] = result['action724_3'] - result['action624_3']
        # result['7_6_sub_action2_4'] = result['action724_4'] - result['action624_4']
        # result['7_6_sub_action2_5'] = result['action724_5'] - result['action624_5']
        #
        # result['7_6_sub_page_0'] = result['page7220'] - result['page6220']
        # result['7_6_sub_page_1'] = result['page7221'] - result['page6221']
        # result['7_6_sub_page_2'] = result['page7222'] - result['page6222']
        # result['7_6_sub_page_3'] = result['page7223'] - result['page6223']
        # result['7_6_sub_page_4'] = result['page7224'] - result['page6224']

        # # cha sum
        # result['sub_sum_1'] = result['7_6_sub_1'] + result['6_5_sub_1'] + result['5_4_sub_1'] + result['4_3_sub_1'] \
        #                       +result['3_2_sub_1'] + result['2_1_sub_1']
        # result['sub_sum_2'] = result['7_6_sub_2'] + result['6_5_sub_2'] + result['5_4_sub_2'] + result['4_3_sub_2'] \
        #                       + result['3_2_sub_2'] + result['2_1_sub_2']
        # result['sub_sum_5'] = result['7_6_sub_5'] + result['6_5_sub_5'] + result['5_4_sub_5'] + result['4_3_sub_5'] \
        #                       + result['3_2_sub_5'] + result['2_1_sub_5']
        # result['sub_sum_30'] = result['7_6_sub_30'] + result['6_5_sub_30'] + result['5_4_sub_30'] + result['4_3_sub_30'] \
        #                        + result['3_2_sub_30'] + result['2_1_sub_30']
        #
        # result['sum_sub_action1_0'] = result['7_6_sub_action1_0'] + result['6_5_sub_action1_0'] + result['5_4_sub_action1_0'] \
        #                               + result['4_3_sub_action1_0'] + result['3_2_sub_action1_0'] + result['2_1_sub_action1_0']
        # result['sum_sub_action1_1'] = result['7_6_sub_action1_1'] + result['6_5_sub_action1_1'] + result['5_4_sub_action1_1'] \
        #                               + result['4_3_sub_action1_1'] + result['3_2_sub_action1_1'] + result['2_1_sub_action1_1']
        # result['sum_sub_action1_2'] = result['7_6_sub_action1_2'] + result['6_5_sub_action1_2'] + result['5_4_sub_action1_2'] \
        #                               + result['4_3_sub_action1_2'] + result['3_2_sub_action1_2'] + result['2_1_sub_action1_2']
        # result['sum_sub_action1_3'] = result['7_6_sub_action1_3'] + result['6_5_sub_action1_3'] + result['5_4_sub_action1_3'] \
        #                               + result['4_3_sub_action1_3'] + result['3_2_sub_action1_3'] + result['2_1_sub_action1_3']
        # result['sum_sub_action1_4'] = result['7_6_sub_action1_4'] + result['6_5_sub_action1_4'] + result['5_4_sub_action1_4'] \
        #                               + result['4_3_sub_action1_4'] + result['3_2_sub_action1_4'] + result['2_1_sub_action1_4']
        # result['sum_sub_action1_5'] = result['7_6_sub_action1_5'] + result['6_5_sub_action1_5'] + result['5_4_sub_action1_5'] \
        #                               + result['4_3_sub_action1_5'] + result['3_2_sub_action1_5'] + result['2_1_sub_action1_5']
        #
        # result['sum_sub_action2_0'] = result['7_6_sub_action2_0'] + result['6_5_sub_action2_0'] + result['5_4_sub_action2_0'] \
        #                               + result['4_3_sub_action2_0'] + result['3_2_sub_action2_0'] + result['2_1_sub_action2_0']
        # result['sum_sub_action2_1'] = result['7_6_sub_action2_1'] + result['6_5_sub_action2_1'] + result['5_4_sub_action2_1'] \
        #                               + result['4_3_sub_action2_1'] + result['3_2_sub_action2_1'] + result['2_1_sub_action2_1']
        # result['sum_sub_action2_2'] = result['7_6_sub_action2_2'] + result['6_5_sub_action2_2'] + result['5_4_sub_action2_2'] \
        #                               + result['4_3_sub_action2_2'] + result['3_2_sub_action2_2'] + result['2_1_sub_action2_2']
        # result['sum_sub_action2_3'] = result['7_6_sub_action2_3'] + result['6_5_sub_action2_3'] + result['5_4_sub_action2_3'] \
        #                               + result['4_3_sub_action2_3'] + result['3_2_sub_action2_3'] + result['2_1_sub_action2_3']
        # result['sum_sub_action2_4'] = result['7_6_sub_action2_4'] + result['6_5_sub_action2_4'] + result['5_4_sub_action2_4'] \
        #                               + result['4_3_sub_action2_4'] + result['3_2_sub_action2_4'] + result['2_1_sub_action2_4']
        # result['sum_sub_action2_5'] = result['7_6_sub_action2_5'] + result['6_5_sub_action2_5'] + result['5_4_sub_action2_5'] \
        #                               + result['4_3_sub_action2_5'] + result['3_2_sub_action2_5'] + result['2_1_sub_action2_5']
        #
        # result['sum_sub_page_0'] = result['7_6_sub_page_0'] + result['6_5_sub_page_0'] + result['5_4_sub_page_0'] + result['4_3_sub_page_0'] \
        #                            + result['3_2_sub_page_0'] + result['2_1_sub_page_0']
        # result['sum_sub_page_1'] = result['7_6_sub_page_1'] + result['6_5_sub_page_1'] + result['5_4_sub_page_1'] + result['4_3_sub_page_1'] \
        #                            + result['3_2_sub_page_1'] + result['2_1_sub_page_1']
        # result['sum_sub_page_2'] = result['7_6_sub_page_2'] + result['6_5_sub_page_2'] + result['5_4_sub_page_2'] + result['4_3_sub_page_2'] \
        #                            + result['3_2_sub_page_2'] + result['2_1_sub_page_2']
        # result['sum_sub_page_3'] = result['7_6_sub_page_3'] + result['6_5_sub_page_3'] + result['5_4_sub_page_3'] + result['4_3_sub_page_3'] \
        #                            + result['3_2_sub_page_3'] + result['2_1_sub_page_3']
        # result['sum_sub_page_4'] = result['7_6_sub_page_4'] + result['6_5_sub_page_4'] + result['5_4_sub_page_4'] + result['4_3_sub_page_4'] \
        #                            + result['3_2_sub_page_4'] + result['2_1_sub_page_4']


#############################################################################################################
        # 116、2天累加从后到前，7个区间，同上
        # 1 Fregion
        activity_region_today = activity_region[(activity_region.day <= today) & (activity_region.day > (today - 2))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 2))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 2))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_1_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_1_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_1_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action123_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page122' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_1_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action124_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 2 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today ) & (activity_region.day > (today - 4))]
        launch_region_today = launch_region[(launch_region.day <= today ) & (launch_region.day > (today - 4))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 4))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_2_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_2_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_2_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action223_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page222' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_2_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action224_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 3 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 6))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 6))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 6))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_3_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_3_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_3_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action323_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page322' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_3_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action324_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 4 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 8))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 8))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 8))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_4_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_4_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_4_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action423_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page422' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_4_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action424_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']
        ###############################
        # 5 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 10))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 10))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 10))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_5_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_5_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_5_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action523_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page522' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_5_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action524_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 6 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 12))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 12))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 12))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'back_6_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'back_6_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'back_6_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'back_action623_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'back_page622' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'back_6_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'back_action624_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']





##############################################################################################################
        # 117、2天不累加，7个区间，作差后直接求和


        ###############################
        # 2 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 10)) & (activity_region.day > (today - 12))]
        launch_region_today = launch_region[(launch_region.day <= (today - 10)) & (launch_region.day > (today - 12))]
        create_region_today = create_region[(create_region.day <= (today - 10)) & (create_region.day > (today - 12))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_2_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_2_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_2_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action223_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page222' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_2_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action224_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today.action_type_new2).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 3 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 8)) & (activity_region.day > (today - 10))]
        launch_region_today = launch_region[(launch_region.day <= (today - 8)) & (launch_region.day > (today - 10))]
        create_region_today = create_region[(create_region.day <= (today - 8)) & (create_region.day > (today - 10))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_3_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_3_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_3_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action323_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.action_type_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page322' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_3_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action324_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today['action_type_new2']).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 4 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 6)) & (activity_region.day > (today - 8))]
        launch_region_today = launch_region[(launch_region.day <= (today - 6)) & (launch_region.day > (today - 8))]
        create_region_today = create_region[(create_region.day <= (today - 6)) & (create_region.day > (today - 8))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_4_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_4_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_4_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action423_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today['action_type_new1']).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page422' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_4_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action424_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today['action_type_new2']).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']
        ###############################
        # 5 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 4)) & (activity_region.day > (today - 6))]
        launch_region_today = launch_region[(launch_region.day <= (today - 4)) & (launch_region.day > (today - 6))]
        create_region_today = create_region[(create_region.day <= (today - 4)) & (create_region.day > (today - 6))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_5_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_5_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_5_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action523_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today['action_type_new1']).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page522' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_5_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action524_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today['action_type_new2']).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 6 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= (today - 2)) & (activity_region.day > (today - 4))]
        launch_region_today = launch_region[(launch_region.day <= (today - 2)) & (launch_region.day > (today - 4))]
        create_region_today = create_region[(create_region.day <= (today - 2)) & (create_region.day > (today - 4))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_6_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_6_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_6_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action623_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today['action_type_new1']).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page622' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_6_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action624_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today['action_type_new2']).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        ###############################
        # 7 Fregion
        activity_region_today = activity_region[
            (activity_region.day <= today) & (activity_region.day > (today - 2))]
        launch_region_today = launch_region[(launch_region.day <= today) & (launch_region.day > (today - 2))]
        create_region_today = create_region[(create_region.day <= today) & (create_region.day > (today - 2))]
        # 1、该用户当天的总启动次数
        temp = launch_region_today.groupby(['user_id'])['day'].agg({'no_7_1': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 2、该用户当天的总拍摄次数
        temp = create_region_today.groupby(['user_id'])['day'].agg({'no_7_2': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 5、该用户当天行为次数
        temp = activity_region_today.groupby(['user_id'])['action_type'].agg({'no_7_5': np.size})  #
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab action_type
        activity_region_today['action_type_new1'] = list(
            map(lambda x: 'no_action723_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today['action_type_new1']).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['action_type_new1']
        # crosstab page
        activity_region_today['page_new1'] = list(map(lambda x: 'no_page722' + str(x), activity_region_today.page))
        temp = pd.crosstab(activity_region_today.user_id, activity_region_today.page_new1).reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        del activity_region_today['page_new1']

        #         30、该用户当天总被浏览多少次
        temp = activity_region_today.groupby(['author_id'])['day'].agg({'no_7_30': np.size})  #
        temp = temp.reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # crosstab AUTHOR action_type
        activity_region_today['action_type_new2'] = list(
            map(lambda x: 'no_action724_' + str(x), activity_region_today.action_type))
        temp = pd.crosstab(activity_region_today.author_id, activity_region_today['action_type_new2']).reset_index()
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')
        del activity_region_today['action_type_new2']

        # zuo cha 2-1
        result['no_2_1_sub_1'] = result['no_2_1'] - result['1_1']
        result['no_2_1_sub_2'] = result['no_2_2'] - result['1_2']
        result['no_2_1_sub_5'] = result['no_2_5'] - result['1_5']
        result['no_2_1_sub_30'] = result['no_2_30'] - result['1_30']
        # result['no_2_1_sub_action1_0'] = result['no_action223_0'] - result['action123_0']
        # result['no_2_1_sub_action1_1'] = result['no_action223_1'] - result['action123_1']
        # result['no_2_1_sub_action1_2'] = result['no_action223_2'] - result['action123_2']
        # result['no_2_1_sub_action1_3'] = result['no_action223_3'] - result['action123_3']
        # result['no_2_1_sub_action1_4'] = result['no_action223_4'] - result['action123_4']
        # result['no_2_1_sub_action1_5'] = result['no_action223_5'] - result['action123_5']
        #
        # result['no_2_1_sub_action2_0'] = result['no_action224_0'] - result['action124_0']
        # result['no_2_1_sub_action2_1'] = result['no_action224_1'] - result['action124_1']
        # result['no_2_1_sub_action2_2'] = result['no_action224_2'] - result['action124_2']
        # result['no_2_1_sub_action2_3'] = result['no_action224_3'] - result['action124_3']
        # result['no_2_1_sub_action2_4'] = result['no_action224_4'] - result['action124_4']
        # result['no_2_1_sub_action2_5'] = result['no_action224_5'] - result['action124_5']
        #
        # result['no_2_1_sub_page_0'] = result['no_page2220'] - result['page1220']
        # result['no_2_1_sub_page_1'] = result['no_page2221'] - result['page1221']
        # result['no_2_1_sub_page_2'] = result['no_page2222'] - result['page1222']
        # result['no_2_1_sub_page_3'] = result['no_page2223'] - result['page1223']
        # result['no_2_1_sub_page_4'] = result['no_page2224'] - result['page1224']

        # zuo cha 3-2
        result['no_3_2_sub_1'] = result['no_3_1'] - result['no_2_1']
        result['no_3_2_sub_2'] = result['no_3_2'] - result['no_2_2']
        result['no_3_2_sub_5'] = result['no_3_5'] - result['no_2_5']
        result['no_3_2_sub_30'] = result['no_3_30'] - result['no_2_30']
        # result['no_3_2_sub_action1_0'] = result['no_action323_0'] - result['no_action223_0']
        # result['no_3_2_sub_action1_1'] = result['no_action323_1'] - result['no_action223_1']
        # result['no_3_2_sub_action1_2'] = result['no_action323_2'] - result['no_action223_2']
        # result['no_3_2_sub_action1_3'] = result['no_action323_3'] - result['no_action223_3']
        # result['no_3_2_sub_action1_4'] = result['no_action323_4'] - result['no_action223_4']
        # result['no_3_2_sub_action1_5'] = result['no_action323_5'] - result['no_action223_5']
        #
        # result['no_3_2_sub_action2_0'] = result['no_action324_0'] - result['no_action224_0']
        # result['no_3_2_sub_action2_1'] = result['no_action324_1'] - result['no_action224_1']
        # result['no_3_2_sub_action2_2'] = result['no_action324_2'] - result['no_action224_2']
        # result['no_3_2_sub_action2_3'] = result['no_action324_3'] - result['no_action224_3']
        # result['no_3_2_sub_action2_4'] = result['no_action324_4'] - result['no_action224_4']
        # result['no_3_2_sub_action2_5'] = result['no_action324_5'] - result['no_action224_5']
        #
        # result['no_3_2_sub_page_0'] = result['no_page3220'] - result['no_page2220']
        # result['no_3_2_sub_page_1'] = result['no_page3221'] - result['no_page2221']
        # result['no_3_2_sub_page_2'] = result['no_page3222'] - result['no_page2222']
        # result['no_3_2_sub_page_3'] = result['no_page3223'] - result['no_page2223']
        # result['no_3_2_sub_page_4'] = result['no_page3224'] - result['no_page2224']

        # zuo cha 4-3
        result['no_4_3_sub_1'] = result['no_4_1'] - result['no_3_1']
        result['no_4_3_sub_2'] = result['no_4_2'] - result['no_3_2']
        result['no_4_3_sub_5'] = result['no_4_5'] - result['no_3_5']
        result['no_4_3_sub_30'] = result['no_4_30'] - result['no_3_30']
        # result['no_4_3_sub_action1_0'] = result['no_action423_0'] - result['no_action323_0']
        # result['no_4_3_sub_action1_1'] = result['no_action423_1'] - result['no_action323_1']
        # result['no_4_3_sub_action1_2'] = result['no_action423_2'] - result['no_action323_2']
        # result['no_4_3_sub_action1_3'] = result['no_action423_3'] - result['no_action323_3']
        # result['no_4_3_sub_action1_4'] = result['no_action423_4'] - result['no_action323_4']
        # result['no_4_3_sub_action1_5'] = result['no_action423_5'] - result['no_action323_5']
        #
        # result['no_4_3_sub_action2_0'] = result['no_action424_0'] - result['no_action324_0']
        # result['no_4_3_sub_action2_1'] = result['no_action424_1'] - result['no_action324_1']
        # result['no_4_3_sub_action2_2'] = result['no_action424_2'] - result['no_action324_2']
        # result['no_4_3_sub_action2_3'] = result['no_action424_3'] - result['no_action324_3']
        # result['no_4_3_sub_action2_4'] = result['no_action424_4'] - result['no_action324_4']
        # result['no_4_3_sub_action2_5'] = result['no_action424_5'] - result['no_action324_5']
        #
        # result['no_4_3_sub_page_0'] = result['no_page4220'] - result['no_page3220']
        # result['no_4_3_sub_page_1'] = result['no_page4221'] - result['no_page3221']
        # result['no_4_3_sub_page_2'] = result['no_page4222'] - result['no_page3222']
        # result['no_4_3_sub_page_3'] = result['no_page4223'] - result['no_page3223']
        # result['no_4_3_sub_page_4'] = result['no_page4224'] - result['no_page3224']

        # zuo cha 5-4
        result['no_5_4_sub_1'] = result['no_5_1'] - result['no_4_1']
        result['no_5_4_sub_2'] = result['no_5_2'] - result['no_4_2']
        result['no_5_4_sub_5'] = result['no_5_5'] - result['no_4_5']
        result['no_5_4_sub_30'] = result['no_5_30'] - result['no_4_30']
        # result['no_5_4_sub_action1_0'] = result['no_action523_0'] - result['no_action423_0']
        # result['no_5_4_sub_action1_1'] = result['no_action523_1'] - result['no_action423_1']
        # result['no_5_4_sub_action1_2'] = result['no_action523_2'] - result['no_action423_2']
        # result['no_5_4_sub_action1_3'] = result['no_action523_3'] - result['no_action423_3']
        # result['no_5_4_sub_action1_4'] = result['no_action523_4'] - result['no_action423_4']
        # result['no_5_4_sub_action1_5'] = result['no_action523_5'] - result['no_action423_5']
        #
        # result['no_5_4_sub_action2_0'] = result['no_action524_0'] - result['no_action424_0']
        # result['no_5_4_sub_action2_1'] = result['no_action524_1'] - result['no_action424_1']
        # result['no_5_4_sub_action2_2'] = result['no_action524_2'] - result['no_action424_2']
        # result['no_5_4_sub_action2_3'] = result['no_action524_3'] - result['no_action424_3']
        # result['no_5_4_sub_action2_4'] = result['no_action524_4'] - result['no_action424_4']
        # result['no_5_4_sub_action2_5'] = result['no_action524_5'] - result['no_action424_5']
        #
        # result['no_5_4_sub_page_0'] = result['no_page5220'] - result['no_page4220']
        # result['no_5_4_sub_page_1'] = result['no_page5221'] - result['no_page4221']
        # result['no_5_4_sub_page_2'] = result['no_page5222'] - result['no_page4222']
        # result['no_5_4_sub_page_3'] = result['no_page5223'] - result['no_page4223']
        # result['no_5_4_sub_page_4'] = result['no_page5224'] - result['no_page4224']

        # zuo cha 6-5
        result['no_6_5_sub_1'] = result['no_6_1'] - result['no_5_1']
        result['no_6_5_sub_2'] = result['no_6_2'] - result['no_5_2']
        result['no_6_5_sub_5'] = result['no_6_5'] - result['no_5_5']
        result['no_6_5_sub_30'] = result['no_6_30'] - result['no_5_30']
        # result['no_6_5_sub_action1_0'] = result['no_action623_0'] - result['no_action523_0']
        # result['no_6_5_sub_action1_1'] = result['no_action623_1'] - result['no_action523_1']
        # result['no_6_5_sub_action1_2'] = result['no_action623_2'] - result['no_action523_2']
        # result['no_6_5_sub_action1_3'] = result['no_action623_3'] - result['no_action523_3']
        # result['no_6_5_sub_action1_4'] = result['no_action623_4'] - result['no_action523_4']
        # result['no_6_5_sub_action1_5'] = result['no_action623_5'] - result['no_action523_5']
        #
        # result['no_6_5_sub_action2_0'] = result['no_action624_0'] - result['no_action524_0']
        # result['no_6_5_sub_action2_1'] = result['no_action624_1'] - result['no_action524_1']
        # result['no_6_5_sub_action2_2'] = result['no_action624_2'] - result['no_action524_2']
        # result['no_6_5_sub_action2_3'] = result['no_action624_3'] - result['no_action524_3']
        # result['no_6_5_sub_action2_4'] = result['no_action624_4'] - result['no_action524_4']
        # result['no_6_5_sub_action2_5'] = result['no_action624_5'] - result['no_action524_5']
        #
        # result['no_6_5_sub_page_0'] = result['no_page6220'] - result['no_page5220']
        # result['no_6_5_sub_page_1'] = result['no_page6221'] - result['no_page5221']
        # result['no_6_5_sub_page_2'] = result['no_page6222'] - result['no_page5222']
        # result['no_6_5_sub_page_3'] = result['no_page6223'] - result['no_page5223']
        # result['no_6_5_sub_page_4'] = result['no_page6224'] - result['no_page5224']

        # zuo cha 7-6
        result['no_7_6_sub_1'] = result['no_7_1'] - result['no_6_1']
        result['no_7_6_sub_2'] = result['no_7_2'] - result['no_6_2']
        result['no_7_6_sub_5'] = result['no_7_5'] - result['no_6_5']
        result['no_7_6_sub_30'] = result['no_7_30'] - result['no_6_30']
        # result['no_7_6_sub_action1_0'] = result['no_action723_0'] - result['no_action623_0']
        # result['no_7_6_sub_action1_1'] = result['no_action723_1'] - result['no_action623_1']
        # result['no_7_6_sub_action1_2'] = result['no_action723_2'] - result['no_action623_2']
        # result['no_7_6_sub_action1_3'] = result['no_action723_3'] - result['no_action623_3']
        # result['no_7_6_sub_action1_4'] = result['no_action723_4'] - result['no_action623_4']
        # result['no_7_6_sub_action1_5'] = result['no_action723_5'] - result['no_action623_5']
        #
        # result['no_7_6_sub_action2_0'] = result['no_action724_0'] - result['no_action624_0']
        # result['no_7_6_sub_action2_1'] = result['no_action724_1'] - result['no_action624_1']
        # result['no_7_6_sub_action2_2'] = result['no_action724_2'] - result['no_action624_2']
        # result['no_7_6_sub_action2_3'] = result['no_action724_3'] - result['no_action624_3']
        # result['no_7_6_sub_action2_4'] = result['no_action724_4'] - result['no_action624_4']
        # result['no_7_6_sub_action2_5'] = result['no_action724_5'] - result['no_action624_5']
        #
        # result['no_7_6_sub_page_0'] = result['no_page7220'] - result['no_page6220']
        # result['no_7_6_sub_page_1'] = result['no_page7221'] - result['no_page6221']
        # result['no_7_6_sub_page_2'] = result['no_page7222'] - result['no_page6222']
        # result['no_7_6_sub_page_3'] = result['no_page7223'] - result['no_page6223']
        # result['no_7_6_sub_page_4'] = result['no_page7224'] - result['no_page6224']

        # cha sum
        result['no_sub_sum_1'] = result['no_7_6_sub_1'] + result['no_6_5_sub_1'] + result['no_5_4_sub_1'] + result['no_4_3_sub_1'] \
                              + result['no_3_2_sub_1'] + result['no_2_1_sub_1']
        result['no_sub_sum_2'] = result['no_7_6_sub_2'] + result['no_6_5_sub_2'] + result['no_5_4_sub_2'] + result['no_4_3_sub_2'] \
                              + result['no_3_2_sub_2'] + result['no_2_1_sub_2']
        result['no_sub_sum_5'] = result['no_7_6_sub_5'] + result['no_6_5_sub_5'] + result['no_5_4_sub_5'] + result['no_4_3_sub_5'] \
                              + result['no_3_2_sub_5'] + result['no_2_1_sub_5']
        result['no_sub_sum_30'] = result['no_7_6_sub_30'] + result['no_6_5_sub_30'] + result['no_5_4_sub_30'] + result['no_4_3_sub_30'] \
                               + result['no_3_2_sub_30'] + result['no_2_1_sub_30']

        # result['no_sum_sub_action1_0'] = result['no_7_6_sub_action1_0'] + result['no_6_5_sub_action1_0'] + result[
        #     'no_5_4_sub_action1_0'] \
        #                               + result['no_4_3_sub_action1_0'] + result['no_3_2_sub_action1_0'] + result[
        #                                   'no_2_1_sub_action1_0']
        # result['no_sum_sub_action1_1'] = result['no_7_6_sub_action1_1'] + result['no_6_5_sub_action1_1'] + result[
        #     'no_5_4_sub_action1_1'] \
        #                               + result['no_4_3_sub_action1_1'] + result['no_3_2_sub_action1_1'] + result[
        #                                   'no_2_1_sub_action1_1']
        # result['no_sum_sub_action1_2'] = result['no_7_6_sub_action1_2'] + result['no_6_5_sub_action1_2'] + result[
        #     'no_5_4_sub_action1_2'] \
        #                               + result['no_4_3_sub_action1_2'] + result['no_3_2_sub_action1_2'] + result[
        #                                   'no_2_1_sub_action1_2']
        # result['no_sum_sub_action1_3'] = result['no_7_6_sub_action1_3'] + result['no_6_5_sub_action1_3'] + result[
        #     'no_5_4_sub_action1_3'] \
        #                               + result['no_4_3_sub_action1_3'] + result['no_3_2_sub_action1_3'] + result[
        #                                   'no_2_1_sub_action1_3']
        # result['no_sum_sub_action1_4'] = result['no_7_6_sub_action1_4'] + result['no_6_5_sub_action1_4'] + result[
        #     'no_5_4_sub_action1_4'] \
        #                               + result['no_4_3_sub_action1_4'] + result['no_3_2_sub_action1_4'] + result[
        #                                   'no_2_1_sub_action1_4']
        # result['no_sum_sub_action1_5'] = result['no_7_6_sub_action1_5'] + result['no_6_5_sub_action1_5'] + result[
        #     'no_5_4_sub_action1_5'] \
        #                               + result['no_4_3_sub_action1_5'] + result['no_3_2_sub_action1_5'] + result[
        #                                   'no_2_1_sub_action1_5']
        #
        # result['no_sum_sub_action2_0'] = result['no_7_6_sub_action2_0'] + result['no_6_5_sub_action2_0'] + result[
        #     'no_5_4_sub_action2_0'] \
        #                               + result['no_4_3_sub_action2_0'] + result['no_3_2_sub_action2_0'] + result[
        #                                   'no_2_1_sub_action2_0']
        # result['no_sum_sub_action2_1'] = result['no_7_6_sub_action2_1'] + result['no_6_5_sub_action2_1'] + result[
        #     'no_5_4_sub_action2_1'] \
        #                               + result['no_4_3_sub_action2_1'] + result['no_3_2_sub_action2_1'] + result[
        #                                   'no_2_1_sub_action2_1']
        # result['no_sum_sub_action2_2'] = result['no_7_6_sub_action2_2'] + result['no_6_5_sub_action2_2'] + result[
        #     'no_5_4_sub_action2_2'] \
        #                               + result['no_4_3_sub_action2_2'] + result['no_3_2_sub_action2_2'] + result[
        #                                   'no_2_1_sub_action2_2']
        # result['no_sum_sub_action2_3'] = result['no_7_6_sub_action2_3'] + result['no_6_5_sub_action2_3'] + result[
        #     'no_5_4_sub_action2_3'] \
        #                               + result['no_4_3_sub_action2_3'] + result['no_3_2_sub_action2_3'] + result[
        #                                   'no_2_1_sub_action2_3']
        # result['no_sum_sub_action2_4'] = result['no_7_6_sub_action2_4'] + result['no_6_5_sub_action2_4'] + result[
        #     'no_5_4_sub_action2_4'] \
        #                               + result['no_4_3_sub_action2_4'] + result['no_3_2_sub_action2_4'] + result[
        #                                   'no_2_1_sub_action2_4']
        # result['no_sum_sub_action2_5'] = result['no_7_6_sub_action2_5'] + result['no_6_5_sub_action2_5'] + result[
        #     'no_5_4_sub_action2_5'] \
        #                               + result['no_4_3_sub_action2_5'] + result['no_3_2_sub_action2_5'] + result[
        #                                   'no_2_1_sub_action2_5']
        #
        # result['no_sum_sub_page_0'] = result['no_7_6_sub_page_0'] + result['no_6_5_sub_page_0'] + result['no_5_4_sub_page_0'] + \
        #                            result['no_4_3_sub_page_0'] \
        #                            + result['no_3_2_sub_page_0'] + result['no_2_1_sub_page_0']
        # result['no_sum_sub_page_1'] = result['no_7_6_sub_page_1'] + result['no_6_5_sub_page_1'] + result['no_5_4_sub_page_1'] + \
        #                            result['no_4_3_sub_page_1'] \
        #                            + result['no_3_2_sub_page_1'] + result['no_2_1_sub_page_1']
        # result['no_sum_sub_page_2'] = result['no_7_6_sub_page_2'] + result['no_6_5_sub_page_2'] + result['no_5_4_sub_page_2'] + \
        #                            result['no_4_3_sub_page_2'] \
        #                            + result['no_3_2_sub_page_2'] + result['no_2_1_sub_page_2']
        # result['no_sum_sub_page_3'] = result['no_7_6_sub_page_3'] + result['no_6_5_sub_page_3'] + result['no_5_4_sub_page_3'] + \
        #                            result['no_4_3_sub_page_3'] \
        #                            + result['no_3_2_sub_page_3'] + result['no_2_1_sub_page_3']
        # result['no_sum_sub_page_4'] = result['no_7_6_sub_page_4'] + result['no_6_5_sub_page_4'] + result['no_5_4_sub_page_4'] + \
        #                            result['no_4_3_sub_page_4'] \
        #                            + result['no_3_2_sub_page_4'] + result['no_2_1_sub_page_4']
        # 118、区间内最后一次活跃距离区间末端的天数


        # 119、区间内median day距离区间末端的天数
        temp = activity_region.drop_duplicates(['user_id','day'])
        temp = temp.groupby(['user_id'])['day'].agg({'median_day': np.median}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        result['median_to_today'] = result['day'] - result['median_day']
        del result['median_day']
        # 120、区间内mean day距离区间末端的天数
        temp = activity_region.drop_duplicates(['user_id', 'day'])
        temp = temp.groupby(['user_id'])['day'].agg({'mean_day': np.mean}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        result['mean_to_today'] = result['day'] - result['mean_day']
        del result['mean_day']

        # 121,author action 0 ,1 rate
        result['author_action0_rate'] = result['action2_0'] /result['h30']
        result['author_action1_rate'] = result['action2_1'] / result['h30']
        # 122,the all region last_day to today
        action_all = activity[activity.day<today]
        temp = action_all.groupby(['user_id'])['day'].agg({'last_day': np.max}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        result['last_to_today'] = result['day'] - result['last_day']

        # 123,
        create_all = create[create.day<today]
        temp = create_all.groupby(['user_id'])['day'].agg({'last_create_day': np.max}).reset_index()  #
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        result['last_create_to_today'] = result['day'] - result['last_create_day']
        del result['last_create_day']
        # 124, is author?
        temp = activity.groupby(['author_id'])['day'].agg({'temp': np.size}).reset_index()  #
        temp.rename(columns={'author_id': 'user_id'}, inplace=True)
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        result['is_author'] = list(map(lambda x:1 if x>0 else 0,result.temp))
        del result['temp']
        # 125,
        # 、该用户的平均，最大，最小，拍摄时间间隔
        temp = action_all.drop_duplicates(['user_id', 'day'])
        temp = temp.sort_values(['day'], ascending=True)
        temp['next_time'] = temp.groupby(['user_id'])['day'].diff(1)
        temp2 = temp
        # average
        temp = temp.groupby(['user_id'])['next_time'].agg({'avg_time_all': np.mean})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     max
        temp = temp2.groupby(['user_id'])['next_time'].agg({'max_time_all': np.max})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     min
        temp = temp2.groupby(['user_id'])['next_time'].agg({'min_time_all': np.min})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     std
        temp = temp2.groupby(['user_id'])['next_time'].agg({'std_time_all': np.std})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        #     var
        temp = temp2.groupby(['user_id'])['next_time'].agg({'var_time_all': np.var})
        temp = temp.reset_index()
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 126,shangci action
        temp = action_all.groupby(['user_id'])['day'].agg({'last_day': np.max}).reset_index()  #
        action_all = pd.merge(action_all, temp, on=['user_id'], how='left')  #
        temp = action_all[action_all.day==action_all.last_day]
        temp = temp.groupby(['user_id'])['last_day'].agg({'last_day_sum': np.size}).reset_index()  #
        del action_all['last_day']
        result = pd.merge(result, temp, on=['user_id'], how='left')  #
        # 127,last one is which week?
        result['last_day_week'] = list(map(lambda x: x % 7, result.last_day))
        del result['last_day']


        F = result.columns
        print(F)


        result_new = result.drop_duplicates(['user_id'])  # 去重
        if train is None:
            train=result_new
        else:
            frames = [train, result_new]  # 添加当日数据到结果集
            train = pd.concat(frames, axis=0)


    # lisan register_type
    print('lisan..')
    temp = pd.get_dummies(train['register_type'], prefix='register_type')
    train = pd.concat([train, temp], axis=1)
    # print(train)
    print('saving...')
    BigTable = train[(train.day<24)]
    BigTable.to_csv('all_train_set.csv', encoding='utf-8', index=None)
    BigTable = train[(train.day <23)]
    BigTable.to_csv('train_set.csv', encoding='utf-8', index=None)
    BigTable = train[train.day==30]
    BigTable.to_csv('test_set.csv', encoding='utf-8', index=None)
    BigTable = train[train.day==23]
    BigTable.to_csv('eval_set.csv', encoding='utf-8', index=None)







# 训练
# xgboost
def xgboosts(df_train,df_test,df_eval):


    print('xgb---training')
    # XGB  'shop_star_level','shop_review_num_level','context_page_id','item_pv_level','item_collected_level','item_sales_level','item_price_level','user_star_level','user_occupation_id','user_age_level','item_category_list3','item_category_list2','item_category_list1','item_city_id','item_brand_id','context_id',
    feature1 = [x for x in df_train.columns if x not in ['register_type','week','today_create_num','today_launch_num','video_id','user_id','register_day','day','author_id','action_type','page','label']]
    feature2 = [x for x in df_test.columns if x not in ['register_type','week','today_create_num', 'today_launch_num', 'video_id', 'user_id', 'register_day','day', 'author_id', 'action_type', 'page', 'label']]

    # feature = list(set(feature1).intersection(set(feature2)))
    feature = [v for v in feature1 if v in feature2]
    print('F len :%s'%len(feature))
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
    predict.to_csv('result_all_inter.csv', encoding='utf-8', index=None)

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
    result.to_csv('result_add_inter.csv', encoding='utf-8', index=None)
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


# mark_new()
# split_F_train()

df_train = pd.read_csv(r'train_set.csv')
df_all_train = pd.read_csv(r'all_train_set.csv')
df_test = pd.read_csv(r'test_set.csv')
df_eval = pd.read_csv(r'eval_set.csv')

xgboosts(df_train,df_test,df_eval)
xgboosts(df_all_train,df_test,df_eval)
