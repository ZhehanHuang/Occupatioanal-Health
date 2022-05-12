# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 02:01:48 2021

@author: huangzehan
"""

import pandas as pd

'''Step1：已有数据的NAICS全部代码导入'''
# coding_exist = pd.read_excel(r'.\data\original_data\2017\cd_r37_2017.xlsx', skiprows=2, usecols='A:K')
coding_exist = pd.read_excel(r'.\data\original_data\2016\cd_r37_2016.xlsx', skiprows=2, usecols='A:K')
coding_exist = coding_exist.replace('-',0)
coding_exist = coding_exist.replace('–',0)

# 移除NAICS编码为空值的行
coding_exist = coding_exist.dropna(subset=['NAICS code(2)'])
coding_exist['NAICS code(2)'] = coding_exist['NAICS code(2)'].astype(str)

# 导入全部6级产业
coding_total = pd.read_excel(r'.\data\support_data\total_6-level_industry.xlsx', sheet_name = 0, usecols='A:B')
coding_total['2012 NAICS Code'] = coding_total['2012 NAICS Code'].astype(str)

# 插入对产业的评级
coding_exist.insert(2,'NAICS coding level',0)
for z in range(coding_exist.shape[0]):
    if (coding_exist.iloc[z,1])[-4:] == '0000':
        coding_exist.iloc[z,2] = 2
    elif len(coding_exist.iloc[z,1]) != 6:
        coding_exist.iloc[z,2] = 2
    elif (coding_exist.iloc[z,1])[-3:] == '000':
        coding_exist.iloc[z,2] = 3
    elif (coding_exist.iloc[z,1])[-2:] == '00':
        coding_exist.iloc[z,2] = 4
    elif coding_total.iloc[:,0].str.contains(coding_exist.iloc[z,1]).any():
        coding_exist.iloc[z,2] = 6
    else: 
        coding_exist.iloc[z,2] = 5

# 仅保留等级为2和6的产业
coding_exist = coding_exist[-coding_exist.iloc[:,2].isin([1,3,4,5])]

# 分别取出2级和6级产业
coding_level2 = coding_exist[-coding_exist.iloc[:,2].isin([6])]
coding_level6 = coding_exist[-coding_exist.iloc[:,2].isin([2])]

'''Step2：对比全部的6级产业'''
coding_level6['NAICS code(2)'] = coding_level6['NAICS code(2)'].astype(str)

# 筛选没有数据的6级产业
coding_merge = pd.merge(coding_total, coding_level6, how = 'left', left_on = ['2012 NAICS Code'], right_on = ['NAICS code(2)'])

coding_level6_left = coding_merge.copy()
j = 0
for k in range(coding_merge.shape[0]):
    if coding_merge.iloc[k,2] != coding_merge.iloc[k,2]:
        coding_level6_left.iloc[j,:] = coding_merge.iloc[k,:]
        j = j + 1
        
coding_level6_left = coding_level6_left.iloc[:j-1,:]
coding_level6_left = coding_level6_left.drop(columns = [coding_level6_left.columns[2], coding_level6_left.columns[3], coding_level6_left.columns[4]])

'''Step3：计数分配填补空缺值'''
coding_counting = coding_level2.iloc[:,0:2]
coding_counting['exist'] = ''
coding_counting['left'] = ''
coding_counting['sum'] = ''
coding_counting = coding_counting.reset_index()
coding_counting = coding_counting.drop(columns = [coding_counting.columns[0]])

def counting1(a,b,c,d,column):
    e = 0
    for l in range(a.shape[0]):
        string = a.iloc[l,column]
        if string[:2] == b:
            e = e + 1
    coding_counting.iloc[c,d] = e
    
def counting2(a,b,c,d,f,column):
    e = 0
    for l in range(a.shape[0]):
        string = a.iloc[l,column]
        if string[:2] == b or string[:2] == c:
            e = e + 1
    coding_counting.iloc[d,f] = e
    
def counting3(a,b,c,d,f,g,column):
    e = 0
    for l in range(a.shape[0]):
        string = a.iloc[l,column]
        if string[:2] == b or string[:2] == c or string[:2] == d:
            e = e + 1
    coding_counting.iloc[f,g] = e

# 已有数据的6级产业分类计数
def counting_all(a,b,c,d):
    counting1(a, '11', b, c, d)
    counting1(a, '21', b+1, c, d)
    counting1(a, '23', b+2, c, d)
    counting3(a, '31', '32', '33', b+3, c, d)
    counting1(a, '22', b+4, c, d)
    counting1(a, '42', b+5, c, d)
    counting2(a, '44', '45', b+6, c, d)
    counting2(a, '48', '49', b+7, c, d)
    counting1(a, '51', b+8, c, d)
    counting1(a, '52', b+9, c, d)
    counting1(a, '53', b+10, c, d)
    counting1(a, '54', b+11, c, d)
    counting1(a, '55', b+12, c, d)
    counting1(a, '56', b+13, c, d)
    counting1(a, '61', b+14, c, d)
    counting1(a, '62', b+15, c, d)
    counting1(a, '71', b+16, c, d)
    counting1(a, '72', b+17, c, d)
    counting1(a, '81', b+18, c, d)
    
counting_all(coding_level6, 0, 2, 1)

# 空缺数据的6级产业分类计数
counting_all(coding_level6_left, 0, 3, 0)

coding_counting.iloc[:,4] = coding_counting.iloc[:,3] + coding_counting.iloc[:,2] # 完善表格

# 计算已有数据分类数据总和
coding_data = coding_level2.copy()
for l in range(coding_counting.shape[0]):
    if l == 0:
        m = 0
    else:
        m = coding_counting.iloc[:l,2].sum()
    n = coding_counting.iloc[l,2]
    coding_data.iloc[l,3:] = coding_level6.iloc[m:m+n,3:].sum()
coding_data = coding_data.drop(columns = [coding_data.columns[2]])

# 计算遗漏分类数据总和
coding_data_left = coding_data.copy()
coding_data_left.iloc[:,2:] = coding_level2.iloc[:,3:] - coding_data.iloc[:,2:]

# 方法1：将遗漏数据总和平均分配给遗漏6级产业分类
'''coding_average = coding_data.copy()
for o in range(coding_average.shape[0]):
    coding_average.iloc[o,2:] = coding_data_left.iloc[o,2:] / coding_counting.iloc[o,3]
coding_average.iloc[:,2:] = coding_average.iloc[:,2:].astype(int)    

## 将计算平均后的遗漏数据回填
coding_level6_left_fix = coding_level6_left.copy()
for l in range(coding_counting.shape[0]):
    if l == 0:
        m = 0
    else:
        m = coding_counting.iloc[:l,3].sum()
    n = coding_counting.iloc[l,3]
    coding_level6_left_fix.iloc[m:m+n,2:] = coding_average.iloc[l,2:]

## 将已有数据与填补数据合并
coding_level6 = coding_level6.drop(columns = [coding_level6.columns[2]])
coding_fix = coding_level6.append(coding_level6_left_fix, ignore_index = True)
coding_fix.iloc[518:,0] = coding_level6_left_fix.iloc[:,1]
coding_fix.iloc[518:,1] = coding_level6_left_fix.iloc[:,0]
coding_fix.sort_values("NAICS code[2]", inplace = True)'''

#方法2：将遗漏数据总和平均分配给全部产业
coding_average_all = coding_data.copy()
for o in range(coding_average_all.shape[0]):
    coding_average_all.iloc[o,2:] = coding_data_left.iloc[o,2:] / coding_counting.iloc[o,4]
coding_average_all.iloc[:,2:] = coding_average_all.iloc[:,2:].astype(int)

## 将计算平均后的遗漏数据回填
coding_level6_left_fix_all = coding_level6_left.copy()
for l in range(coding_counting.shape[0]):
    if l == 0:
        m = 0
    else:
        m = coding_counting.iloc[:l,3].sum()
    n = coding_counting.iloc[l,3]
    coding_level6_left_fix_all.iloc[m:m+n,2:] = coding_average_all.iloc[l,2:]

coding_level6_fix_all = coding_level6.copy()
coding_level6_fix_all = coding_level6_fix_all.drop(columns=[coding_level6_fix_all.columns[2]])
for p in range(coding_counting.shape[0]):
    if p == 0:
        x = 0
    else:
        x = coding_counting.iloc[:p,2].sum()
    y = coding_counting.iloc[p,2]
    coding_level6_fix_all.iloc[x:x+y,2:] = coding_level6.iloc[x:x+y,3:] + coding_average_all.iloc[p,2:]

## 将已有数据与填补数据合并
coding_fix_all = coding_level6_fix_all.append(coding_level6_left_fix_all, ignore_index = True)
a = pd.DataFrame(coding_level6.shape)
b = a.iloc[0,0]
coding_fix_all.iloc[b:,0] = coding_level6_left_fix_all.iloc[:,1]
coding_fix_all.iloc[b:,1] = coding_level6_left_fix_all.iloc[:,0]
