# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 22:21:57 2021

@author: huangzehan
"""

import pandas as pd
import numpy as np
import random as r
import math
from numpy import isnan, isinf
import time

'''Step1: 导入按行业划分的非致命伤害和疾病的总病例（NAICS代码）'''
df_industry_nature = pd.read_excel(r'.\data\allocation_data\Allocation.xlsx', sheet_name = 1, usecols='A:O') # 2018年
# df_industry_nature = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 0, usecols='A:O') # 2017年
# df_industry_nature = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 2, usecols='A:O') # 2016年
# df_industry_nature = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 4, usecols='A:O') # 2015年
# df_industry_nature = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 6, usecols='A:O') # 2014年

'''Step2.1: 导入按年龄层划分的非致命伤害和疾病的总病例（BLS划分）'''
df_industry_age = pd.read_excel(r'.\data\allocation_data\Allocation.xlsx', sheet_name = 2, usecols='A:K') # 2018年
# df_industry_age = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 1, usecols='A:K') # 2017年
# df_industry_age = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 3, usecols='A:K') # 2016年
# df_industry_age = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 5, usecols='A:K') # 2015年
# df_industry_age = pd.read_excel(r'.\data\allocation_data\Allocation_2014-2017.xlsx', sheet_name = 7, usecols='A:K') # 2015年

'''Step2.2：将BLS划分的年龄区间与WHO的年龄区间向匹配'''
df_industry_age_WHO = df_industry_age.iloc[:,0:3]
df_industry_age_WHO['15 to 44'] = df_industry_age['14 to 15'] + df_industry_age['16 to 19'] + df_industry_age['20 to 24'] + df_industry_age['25 to 34'] + df_industry_age['35 to 44']
# 假设病例数在每个年龄区间内的分布是沿中位数对称的
df_industry_age_WHO['45 to 59'] = df_industry_age['45 to 54'] + df_industry_age['55 to 64'] * 0.5
df_industry_age_WHO['60 to 80'] = df_industry_age['55 to 64'] * 0.5 + df_industry_age['65 and over']

'''Step3：导入不同伤残的长短期分配系数'''
long_short_split = pd.read_excel(r'.\data\support_data\support_data.xlsx', sheet_name=2)

'''Step4：计算针对每个产业（per BLS code），每个年龄区间的每种非致命伤害的疾病病例''' # 使用3维数组进行表示
np_industry_injury_age = np.zeros((df_industry_nature.shape[0],df_industry_nature.shape[1]-2,df_industry_age_WHO.shape[1]-3))
for i in range(df_industry_nature.shape[0]):   # 每个产业
    for j in range(df_industry_nature.shape[1]-2): # 每种非致命伤害
        for k in range(df_industry_age_WHO.shape[1]-3): # 每个年龄区间
            np_industry_injury_age[i,j,k] = df_industry_age_WHO.iloc[i,3] / df_industry_age_WHO.iloc[i,k+2] * df_industry_nature.iloc[i,j+2]
            
'''Step4.1：计算针对每个年龄区间，每种伤害的短期非致命伤害的总病例（short-term nonfatal injuries per injury）'''
short_injury_coefficient = long_short_split.loc[:,'Short-term']
np_industry_shortinj_age = np.zeros((df_industry_nature.shape[0],df_industry_nature.shape[1]-2,df_industry_age_WHO.shape[1]-3))
for i in range(df_industry_nature.shape[0]): # 每个产业
    for k in range(df_industry_age_WHO.shape[1]-3): # 每个年龄区间
        np_industry_shortinj_age[i,:,k] = short_injury_coefficient * np_industry_injury_age[i,:,k]
        
'''Step4.2：计算针对每个年龄区间，每种伤害的长期非致命伤害的总病例（long-term nonfatal injuries per injury）'''
long_injury_coefficient = long_short_split.loc[:,'Life-long']
np_industry_longinj_age = np.zeros((df_industry_nature.shape[0],df_industry_nature.shape[1]-2,df_industry_age_WHO.shape[1]-3))
for i in range(df_industry_nature.shape[0]): # 每个产业
    for k in range(df_industry_age_WHO.shape[1]-3): # 每个年龄区间
        np_industry_longinj_age[i,:,k] = long_injury_coefficient * np_industry_injury_age[i,:,k]

'''MCS'''
cf_CMS = np.zeros((1000,397,5))

short_duration = pd.read_excel(r'.\data\support_data\support_data.xlsx', sheet_name=4, usecols='A:C') #BLS
np_short_duration = short_duration.loc[:13,'Median days away from work'] / 250 # 单位为年，一年250个工作日
np_short_duration = pd.DataFrame(np_short_duration)

long_duration = pd.read_excel(r'.\data\support_data\support_data.xlsx', sheet_name=3, usecols='A:D')
np_long_duration = long_duration.iloc[:,-1] # 使用不区分性别的剩余生命作为持续时间
np_long_duration = pd.DataFrame(np_long_duration)

short_weight = pd.read_excel(r'.\data\support_data\support_data.xlsx', sheet_name=0)
long_weight = pd.read_excel(r'.\data\support_data\support_data.xlsx', sheet_name=1)

g = pd.read_excel(r'.\data\original_data\USEEIOv2.0.xlsx', sheet_name=23, usecols='A:B')
g['USEEIO_Code'] = g['USEEIO_Code'].map(lambda x: str(x)[:-3])
g['USEEIO_Code'] = g['USEEIO_Code'].astype(str)

make_mat = pd.read_excel(r'.\data\original_data\USEEIOv2.0.xlsx', sheet_name=6, usecols='A:OV')
make_mat['USEEIO_Code'] = make_mat['USEEIO_Code'].astype(str)
make_mat['USEEIO_Code'] = make_mat['USEEIO_Code'].apply(lambda x:x[:6]).tolist()
make_mat_T = pd.DataFrame(make_mat.values.T, index=make_mat.columns, columns=make_mat.index)
make_mat_T[0] = make_mat_T[0].apply(lambda x:x[:6]).tolist()

q = pd.read_excel(r'.\data\original_data\USEEIOv2.0.xlsx', sheet_name=22, usecols='A:B')
q['USEEIO_Code'] = q['USEEIO_Code'].map(lambda x: str(x)[:-3])
q['USEEIO_Code'] = q['USEEIO_Code'].astype(str)

requ_mat = pd.read_excel(r'.\data\original_data\USEEIOv2.0.xlsx', sheet_name=9, usecols='A:OV')
requ_mat['USEEIO_Code'] = requ_mat['USEEIO_Code'].map(lambda x: str(x)[:-3])
requ_mat['USEEIO_Code'] = requ_mat['USEEIO_Code'].astype(str)

total_con = pd.read_excel(r'.\data\original_data\USEEIOv2.0.xlsx', sheet_name=24, usecols='A:B')

Conv = pd.read_excel(r'.\data\support_data\CONV.xlsx', sheet_name=2, usecols='B:OR')
Conv['Category code'] = Conv['Category code'].astype(str)
Conv['Category code'] = Conv['Category code'].map(lambda x: str(x)[:-2])

   
for t in range(1000):
    start = time.time()
    
    '''Step5.1 导入短期伤害和疾病的持续时间'''
    random_short_duration = np_short_duration.copy()
    random_short_duration = pd.DataFrame(random_short_duration)
    l = 0
    for l in range(13):
        random_short_duration.iloc[l,0] = r.uniform((0.9*np_short_duration.iloc[l,0]), (1.1*np_short_duration.iloc[l,0]))

    '''Step5.2 导入长期伤害和疾病的持续时间'''    
    random_long_duration = np_long_duration.copy()
    random_long_duration = pd.DataFrame(random_long_duration)
    l = 0
    for l in range(3):
        random_long_duration.iloc[l,0] = r.uniform((0.9*np_long_duration.iloc[l,0]), (1.1*np_long_duration.iloc[l,0]))

    random_long_duration = pd.Series(random_long_duration.iloc[:,0])
    '''Step6.1 导入短期伤害和疾病的权重'''    
    random_short_weight = short_weight.iloc[:,1]
    random_short_weight = pd.DataFrame(random_short_weight)
    m = 0
    for m in range(13):
        if short_weight.iloc[m,1] == 0:
            random_short_weight.iloc[m,0] = 0
        else:
            random_short_weight.iloc[m,0] = np.random.lognormal(math.log(short_weight.iloc[m,1]), short_weight.iloc[m,2], 1)

    '''Step6.2 导入长期伤害和疾病的权重'''     
    random_long_weight = long_weight.iloc[:,1]
    random_long_weight = pd.DataFrame(random_long_weight)
    n = 0
    for n in range(13):
        if long_weight.iloc[n,1] == 0:
            random_long_weight.iloc[n,0] = 0
        else:
            random_long_weight.iloc[n,0] = np.random.lognormal(math.log(long_weight.iloc[n,1]), long_weight.iloc[n,2], 1) 
   
    '''Step7.1 计算短期伤害和疾病的YLD'''
    short_injuries = pd.DataFrame(np_industry_shortinj_age.sum(axis=2)).fillna(0) #沿年龄区间叠加
    yld_short = pd.DataFrame(short_injuries * random_short_weight.iloc[:,0] * random_short_duration.iloc[:,0])
    yld_short.columns = df_industry_nature.columns[2:]
    yld_short.insert(0,'Industry',df_industry_nature.iloc[:,0])
    yld_short.insert(1,'NAICS code',df_industry_nature.iloc[:,1])

    '''Step7.2 计算长期伤害和疾病的YLD'''
    yld_long = df_industry_nature.copy()
    i = 0
    j = 0
    for i in range(df_industry_nature.shape[0]): # 每个产业
        for j in range(df_industry_nature.shape[1]-2): # 每种非致命伤害
            yld_long.iloc[i,j+2] = sum(np_industry_longinj_age[i,j,:] * random_long_duration * random_long_weight.iloc[j,0])

    yld_total = yld_short.iloc[:,2:] + yld_long.iloc[:,2:]
    yld_total.insert(0,'Industry',df_industry_nature.iloc[:,0])
    yld_total.insert(1,'NAICS code',df_industry_nature.iloc[:,1])

    '''CFs计算'''
    '''Step1：导入DALY数据'''
    DALY_total = yld_total
    DALY_total = DALY_total.rename(columns={'NAICS code':'Category code'})
    DALY_total['Category code'] = DALY_total['Category code'].astype(str)

    '''Step2：导入转换矩阵'''    
    Conv_cut = pd.merge(DALY_total, Conv, on = ['Category code'])
    Conv_cut = Conv_cut.drop(columns = Conv_cut.columns[2:16])
    Conv_cut = Conv[0:1].append(Conv_cut, ignore_index=True)
    Conv_cut = Conv_cut.drop(columns = [Conv_cut.columns[0], Conv_cut.columns[-1]])
    Conv_cut.insert(0,'Category',Conv.iloc[:,0])
    Conv_cut[0:1] = Conv_cut[0:1].astype(str)
    Conv_cut[0:1] = Conv_cut.iloc[0,:].apply(lambda x:x[:6]).tolist()
    DALY_cut = pd.merge(DALY_total, Conv, on = ['Category code'])
    DALY_cut = DALY_cut.drop(columns = DALY_cut.columns[15:])
    list_nacis = DALY_cut.iloc[:,1]

    # 求转换矩阵的转置
    Conv_T = pd.DataFrame(Conv_cut.values.T, index=Conv_cut.columns, columns=Conv_cut.index)
    list_io_c = Conv_T.iloc[2:,0]
    list_io_c.columns = ['USEEIO_Code']

    '''Step3：导入产业总产出(Industry output)'''    
    # 计算g与转换矩阵中的IO编码的交集
    g_cut = pd.merge(list_io_c, g, left_on = [0], right_on = ['USEEIO_Code'])
    g_cut = g_cut.drop(columns = g_cut.columns[0])
    list_io_g = pd.DataFrame(g_cut.iloc[:,0])

    '''Step4：将NAICS编码转换为IO编码'''
    Conv_matrix = np.array(Conv_cut.iloc[1:,2:])
    DALY_matrix = np.array(DALY_cut.iloc[:,2:])
    Conv_matrix = Conv_matrix.astype(float)
    DALY_matrix = DALY_matrix.astype(float)
    DALY_matrix[isnan(DALY_matrix)] = 0
    DALY_matrix[isinf(DALY_matrix)] = 0    
    Conv_matrix[isnan(Conv_matrix)] = 0
    DALY_io_ini_np = np.dot(Conv_matrix.T, DALY_matrix)
    DALY_io_ini_df = pd.DataFrame(DALY_io_ini_np, index = list_io_c)
    DALY_io_ini_df['USEEIO_Code'] = DALY_io_ini_df.index
    DALY_io_fin_df = pd.merge(list_io_g, DALY_io_ini_df, on = ['USEEIO_Code'])

    '''Step5；根据公式将g纳入计算'''
    process_1 = DALY_io_fin_df.copy()
    i = 0
    for i in range(DALY_io_fin_df.shape[1]-1):
        process_1.iloc[:,i+1] = DALY_io_fin_df.iloc[:,i+1] / g_cut.iloc[:,1]

    '''Step6：导入生产矩阵(make matrix)'''    
    make_mat_T_fin = pd.merge(list_io_g, make_mat_T, left_on = ['USEEIO_Code'], right_on = [0])
    make_mat_T_fin = make_mat_T[0:1].append(make_mat_T_fin, ignore_index=True)
    make_mat_T_fin = make_mat_T_fin.drop(columns = make_mat_T_fin.columns[0])
    list_io_v1 = pd.DataFrame(make_mat_T_fin.iloc[1:,-1])
    make_mat_fin = pd.DataFrame(make_mat_T_fin.values.T, index=make_mat_T_fin.columns, columns=make_mat_T_fin.index)
    list_io_v2 = pd.DataFrame(make_mat_fin.iloc[:411,0])

    '''Step7：将生产矩阵(make matrix)纳入计算，将产业产出与产品产出相转化'''
    process_1 = pd.merge(process_1, list_io_v1, on = ['USEEIO_Code'])
    process_1_np = np.array(process_1.iloc[:,1:])
    make_mat_np = np.array(make_mat_fin.iloc[:411,1:])
    process_2_np = np.matmul(make_mat_np, process_1_np)
    process_2_df = pd.DataFrame(process_2_np, index = list_io_v2.iloc[:,0])
    process_2_df['USEEIO_Code'] = process_2_df.index

    '''Step8：导入商品总产出(Commodity output)'''    
    q_cut = pd.merge(list_io_v2, q, left_on = [0], right_on = ['USEEIO_Code'])
    q_cut = q_cut.drop(columns = q_cut.columns[0])
    list_io_q = pd.DataFrame(q_cut.iloc[:,0])

    '''Step9：将q纳入计算'''
    process_2_df = pd.merge(list_io_q, process_2_df, on = ['USEEIO_Code'])
    process_3 = process_2_df.copy()
    j = 0
    for j in range(process_2_df.shape[1]-1):
        process_3.iloc[:,j+1] = process_2_df.iloc[:,j+1] / q_cut.iloc[:,1]
    process_3_np = np.array(process_3.iloc[:,1:])
    process_3_df = pd.DataFrame(process_3_np, index = list_io_q.iloc[:,0])
    process_3_cut = pd.merge(process_3_df, list_io_v1, on = ['USEEIO_Code'])
    process_3_cut = np.array(process_3_cut.iloc[:,1:])
    
    '''Step10：导入直接需求矩阵(Direct requirements matrix)'''    
    requ_mat_fin = pd.merge(list_io_q, requ_mat, on = ['USEEIO_Code'])
    requ_mat_fin = requ_mat[0:1].append(requ_mat_fin, ignore_index=True)
    requ_mat_T = pd.DataFrame(requ_mat_fin.values.T, index=requ_mat_fin.columns, columns=requ_mat_fin.index)
    requ_mat_T[0] = requ_mat_T[0].map(lambda x: str(x)[:-3])
    requ_mat_T[0] = requ_mat_T[0].astype(str)
    requ_mat_T_fin = pd.merge(list_io_q, requ_mat_T, left_on = ['USEEIO_Code'], right_on = [0])
    requ_mat_T_fin = requ_mat_T_fin.drop(columns = requ_mat_T_fin.columns[1])
    requ_mat_T_fin = requ_mat_T[0:1].append(requ_mat_T_fin, ignore_index=True)
    requ_mat_T_fin = requ_mat_T_fin.drop(columns = requ_mat_T_fin.columns[0])
    requ_mat_real = pd.DataFrame(requ_mat_T_fin.values.T, index=requ_mat_T_fin.columns, columns=requ_mat_T_fin.index)

    '''Step11：将A纳入计算'''
    A_np = np.array(requ_mat_real.iloc[:404,1:])
    i_A = np.eye(404)-A_np
    i_A = np.array(i_A, dtype='float')
    i_A_inv = np.linalg.inv(i_A)
    process_4_np = np.dot(i_A_inv.T, process_3_np)
    
    indirect_in_sec = np.zeros((404,2))
    x = 0
    for x in range(404):
        indirect_in_sec[x][0] = i_A_inv[x][x]
        indirect_in_sec[x][1] = 1
    
    indirect_in_sec = pd.DataFrame(indirect_in_sec)
    indirect_in_sec.iloc[:,0] = indirect_in_sec.iloc[:,0] - indirect_in_sec.iloc[:,1]
    
    indirect_in_sec = pd.concat([indirect_in_sec, list_io_q], axis=1)
    indirect_in_sec_cut = pd.merge(indirect_in_sec, list_io_v1, on = ['USEEIO_Code'])
    
    process_4_df = pd.DataFrame(process_4_np, index = list_io_q.iloc[:,0])
    process_4_1 = pd.merge(process_4_df, list_io_v1, on = ['USEEIO_Code'])
    process_4_1_np = np.array(process_4_1.iloc[:,1:])

    '''Step 12: calculate CFs'''
    cf_supply = process_4_1_np.sum(axis=1)
    #cf_supply = process_4_np.sum(axis=1)  # DALY per $ over the supply chain 
    cf_direct = process_3_cut.sum(axis=1)  # DALY per $ in the sector
    cf_indirect = cf_supply - cf_direct
    cf_indirect_in_sector = cf_indirect * indirect_in_sec_cut.iloc[:,0]
    cf_indirect_out_sector = cf_indirect - cf_indirect_in_sector
    
    end = time.time()
    print(str((end-start)/60) + ' minutes')
    
    cf_supply = pd.DataFrame(cf_supply)
    cf_direct = pd.DataFrame(cf_direct)
    cf_indirect = pd.DataFrame(cf_indirect)
    cf_indirect_in_sector = pd.DataFrame(cf_indirect_in_sector)
    cf_indirect_out_sector = pd.DataFrame(cf_indirect_out_sector)
    cf_total = list_io_v1.copy()
    cf_total.reset_index(drop=True, inplace=True)
    cf_total.insert(1,'cf_supply',cf_supply.iloc[:,0])
    cf_total.insert(2,'Direct Impact',cf_direct.iloc[:,0])
    cf_total.insert(3,'Indirect Impact',cf_indirect.iloc[:,0])
    cf_total.insert(4,'Supply chain impact in producer secotr',cf_indirect_in_sector.iloc[:,0])
    cf_total.insert(5,'Supply chaion impact in other sectors than producer sector',cf_indirect_out_sector.iloc[:,0])
    
    total_con['USEEIO_Code'] = total_con['USEEIO_Code'].map(lambda x: str(x)[:-3])
    total_con['USEEIO_Code'] = total_con['USEEIO_Code'].astype(str)
    total_cut = pd.merge(list_io_v1, total_con, on = ['USEEIO_Code'])

    
    # 依次为'USEEIO Code','cf_supply','Direct Impact','Indirect Impact'
    np_cf_total = np.array(cf_total.iloc[:,1:])
    cf_CMS[t] = np_cf_total