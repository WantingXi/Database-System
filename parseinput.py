#!/usr/bin/env python
# -*- coding: utf-8 -*-
# wx641 & kw2669

import numpy as np
import re
import time
from statistics import mean
#from BTrees.OIBTree import OIBTree
from BTrees.OOBTree import OOBTree

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename',  default='input.txt', help='filename.txt ')

opt = parser.parse_args()


def txt_to_matrix(filename):
    file = open(filename)
    lines = file.readlines()
    record = []
    for line in lines:
        line = line.strip().split('|')  # strip()默认移除字符串首尾空格或换行符
        #print(line)
        record.append(line)
    datamat = np.array(record)
    return datamat


def find_col(arr,s):
    for i in range(len(arr[0])):
        if arr[0][i] == s:
            return arr[:,i]


def find_col_num(arr,s):
    for i in range(len(arr[0])):
        if arr[0][i] == s:
            return i

def find_rows(col, value):
    res = []
    for i in range(1,len(col)):
        if col[i] == value:
            res.append(i)
    return res

def join_rows(i,j,t1,t2):
    res = []
    for items in t1[i]:
        res.append(items)
    for items in t2[j]:
        res.append(items)
    return res


def running_mean(l, N):
    sum = 0
    result = list( 0 for x in l)
    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum / (i+1)
    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum / N
    return result

def moving_sum(l, N):
    sum = 0
    result = list( 0 for x in l)

    for i in range( 0, N ):
        sum = sum + l[i]
        result[i] = sum

    for i in range( N, len(l) ):
        sum = sum - l[i-N] + l[i]
        result[i] = sum

    return result

def check_if_index(hash_dict, btree_dict, col_dict, col, arr):
    table_col = arr + '_' + col
    if table_col not in col_dict:
        return 'null'
    if col_dict[table_col] == 'hash':
        return hash_dict[table_col]
    if col_dict[table_col] == 'Btree':
        return btree_dict[table_col]

def relop(s,loc):
    if loc == 0:
        if re.search(r'=', s):
            if re.search(r'<=', s):
                return 1
            elif re.search(r'>=', s):
                return 2
            elif re.search(r'!=', s):
                return 3
            else:
                return 4
        if re.search(r'<', s):
            if re.search(r'<=', s):
                return 1
            else:
                return 5
        if re.search(r'>', s):
            if re.search(r'>=', s):
                return 2
            else:
                return 6
    else:
        if re.search(r'=', s):
            if re.search(r'<=', s):
                return 2
            elif re.search(r'>=', s):
                return 1
            elif re.search(r'!=', s):
                return 3
            else:
                return 4
        if re.search(r'<', s):
            if re.search(r'<=', s):
                return 2
            else:
                return 6
        if re.search(r'>', s):
            if re.search(r'>=', s):
                return 1
            else:
                return 5

def parse_terms(terms):
    if re.search('[a-z]',terms[0]):
        loc = 0
        constant = terms[1]
        if re.search(r'[\+\-\*\/]+', terms[0]):
            item = re.split(r'[\+\-\*\/]+', terms[0])
            col = item[0]
            const = item[1]
            if re.search(r'\+', terms[0]):
                arithop = 1
            if re.search(r'\-', terms[0]):
                arithop = 2
            if re.search(r'\*', terms[0]):
                arithop = 3
            if re.search(r'\/', terms[0]):
                arithop = 4
        else:
            col = terms[0]
            const = 0
            arithop = 1
    if re.search('[a-z]',terms[1]):
        loc = 1
        constant = terms[0]
        if re.search(r'[\+\-\*\/]+', terms[1]):
            item = re.split(r'[\+\-\*\/]+', terms[1])
            col = item[0]
            const = item[1]
            if re.search(r'\+', terms[1]):
                arithop = 1
            if re.search(r'\-', terms[1]):
                arithop = 2
            if re.search(r'\*', terms[1]):
                arithop = 3
            if re.search(r'\/', terms[1]):
                arithop = 4
        else:
            col = terms[1]
            const = 0
            arithop = 1
    return loc, col, const, arithop, constant

def parse_condition(ori_table,s,res, data):
    terms = re.split(r'[\=\>\<\!]+', s)
    loc, col, const, arithop, constant = parse_terms(terms)
    op = relop(s, loc)
    print(op)
    # if op is <=
    if op == 1:
        #col_num = find_col_num(data, col)
        for i in range(1, len(data)):
            left = col_content(data, arithop, col, const, i)
            if left <= float(constant):
                res.append(list(data[i]))

    # if op is >=
    if op == 2:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            left = col_content(data, arithop, col, const, i)
            if left >= float(constant):
                res.append(list(data[i]))

    # if op is !=
    if op == 3:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            left = col_content(data, arithop, col, const, i)
            if left != float(constant):
                res.append(list(data[i]))

    # if op is =
    if op == 4:
        # index
        tab_col = ori_table + '_' + col
        if tab_col in col_dict.keys():
            if col_dict[tab_col] == 'hash':
                temp_dict = hash_index_dict[tab_col]
                # ??????? how to define
                row_list = temp_dict[constant]
                for row in row_list:
                    res.append(list(data[row]))
            if col_dict[tab_col] == 'Btree':
                temp_dict = Btree_index_dict[tab_col]
                # ??????? how to define
                row_list = temp_dict[constant]
                for row in row_list:
                    res.append(list(data[row]))
                res.append(list(data[row]))
        # no index
        else:
            # identify column
            col_num = find_col_num(data, terms[0])
            for i in range(1, len(data)):
                if data[i][col_num].isdigit():
                    left = col_content(data, arithop, col, const, i)
                    if left == float(constant):
                        res.append(list(data[i]))
                else:
                    if data[i][col_num] == constant:
                        res.append(list(data[i]))

    # if op is <
    if op == 5:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            left = col_content(data, arithop, col, const, i)
            if left < float(constant):
                res.append(list(data[i]))

    # if op is >
    if op == 6:
        col_num = find_col_num(data, terms[0])
        for i in range(1, len(data)):
            left = col_content(data, arithop, col, const, i)
            if left > float(constant):
                res.append(list(data[i]))

    res = list(set([tuple(t) for t in res]))
    return res

def sort_by_col(data,col_name):
    order = []
    for i in range(0, len(data)):
        order.append(list(data[i]))
        k = order[0].index(col_name)
    for i in range(1, len(order)):
        order[i][k] = int(order[i][k])
    sort = []
    sort.append(order[0])
    temp = (sorted(order[1:], key=(lambda x: [x[k]])))
    for i in range(0, len(temp)):
        sort.append(temp[i])
    res_table = np.asarray(sort)
    return res_table

def join_equal(all_table_dict,table1,table2,term):
    table1_cols = all_table_dict[table1][0]
    table2_cols = all_table_dict[table2][0]

    res = []
    for i in range(0, len(table1_cols)):
        col_name = table1 + '_' + table1_cols[i]
        res.append(col_name)
    for i in range(0, len(table2_cols)):
        col_name = table2 + '_' + table2_cols[i]
        res.append(col_name)
    mat = []
    mat.append(res)
    terms = term.split('=')
    t1, col1, const1, arithop1, t2, col2, const2, arithop2 = parse_join_terms(terms[0], terms[1])
    data = all_table_dict[t1]
    data2 = all_table_dict[t2]
    arr1 = find_col(data, col1)
    arr2 = find_col(data2, col2)
    col_num1 = find_col_num(data, col1)
    col_num2 = find_col_num(data2, col2)
    tab_col1 = t1 + '_' + col1
    tab_col2 = t2 + '_' + col2
    if tab_col1 in col_dict.keys():
        if col_dict[tab_col1] == 'hash':
            temp_dict = hash_index_dict[tab_col1]
            i = 1
            while i < len(data2):
                value = data2[i][col_num2]
                if value in temp_dict.keys():
                    row_list = temp_dict[value]
                    for row in row_list:
                        temp = join_rows(row, i, data, data2)
                        mat.append(temp)
                i = i + 1
        if col_dict[tab_col1] == 'Btree':
            temp_dict = Btree_index_dict[tab_col1]
            i = 1
            while i < len(data2):
                value = data2[i][col_num2]
                if value in temp_dict.keys():
                    row_list = temp_dict[value]
                    for row in row_list:
                        temp = join_rows(row, i, data, data2)
                        mat.append(temp)
                i = i + 1
    elif tab_col2 in col_dict.keys():
        if col_dict[tab_col2] == 'hash':
            temp_dict = hash_index_dict[tab_col2]
            i = 1
            while i < len(data):
                value = data[i][col_num1]
                if value in temp_dict.keys():
                    row_list = temp_dict[value]
                    for row in row_list:
                        temp = join_rows(i, row, data, data2)
                        mat.append(temp)
                i = i + 1
        if col_dict[tab_col2] == 'Btree':
            temp_dict = Btree_index_dict[tab_col2]
            i = 1
            while i < len(data):
                value = data[i][col_num1]
                if value in temp_dict.keys():
                    row_list = temp_dict[value]
                    for row in row_list:
                        temp = join_rows(i, row, data, data2)
                        mat.append(temp)
                i = i + 1
    else:
        checked = {}
        i = 1
        while i < len(data):
            value = data[i][col_num1]
            if value not in checked.keys():
                t2_num = []
                for m in range(1, len(arr2)):
                    if value == arr2[m]:
                        t2_num.append(m)
                checked[value] = t2_num
                for j in range(0, len(t2_num)):
                    k = t2_num[j]
                    temp = join_rows(i, k, data, data2)
                    mat.append(temp)
            else:
                t2_num = checked[value]
                for j in range(0, len(t2_num)):
                    k = t2_num[j]
                    temp = join_rows(i, k, data, data2)
                    mat.append(temp)
            i = i + 1
    datamat = np.array(mat)
    return datamat

def parse_join_terms(var1, var2):
    if re.search(r'[\+\-\*\/]+', var1):
        table1 = var1.split('.')[0]
        temp = var1.split('.')[1]
        item = re.split(r'[\+\-\*\/]+', temp)
        col1 = item[0]
        const1 = item[1]
        if re.search(r'\+', temp):
            arithop1 = 1
        if re.search(r'\-', temp):
            arithop1 = 2
        if re.search(r'\*', temp):
            arithop1 = 3
        if re.search(r'\/', temp):
            arithop1 = 4
    if not (re.search(r'[\+\-\*\/]+', var1)):
        table1 = var1.split('.')[0]
        col1 = var1.split('.')[1]
        const1 = 0
        arithop1 = 5
    if re.search(r'[\+\-\*\/]+', var2):
        table2 = var2.split('.')[0]
        temp = var2.split('.')[1]
        item = re.split(r'[\+\-\*\/]+', temp)
        col2 = item[0]
        const2 = item[1]
        if re.search(r'\+', temp):
            arithop2 = 1
        if re.search(r'\-', temp):
            arithop2 = 2
        if re.search(r'\*', temp):
            arithop2 = 3
        if re.search(r'\/', temp):
            arithop2 = 4
    if not (re.search(r'[\+\-\*\/]+', var2)):
        table2 = var2.split('.')[0]
        col2 = var2.split('.')[1]
        const2 = 0
        arithop2 = 5

    return table1, col1, const1, arithop1, table2, col2, const2, arithop2

def col_content(data,arithop,col,const,i):
    col_num = find_col_num(data, col)
    if arithop == 1:
        return float(data[i][col_num]) + float(const)
    if arithop == 2:
        return float(data[i][col_num]) - float(const)
    if arithop == 3:
        return float(data[i][col_num]) * float(const)
    if arithop == 4:
        return float(data[i][col_num]) / float(const)
    if arithop == 5:
        return data[i][col_num]

def join_terms(op,terms,mat):
    new_res = []
    new_res.append(list(mat[0]))
    t1, col1, const1, arithop1, t2, col2, const2, arithop2 = parse_join_terms(terms[0], terms[1])
    col_name1 = t1 + '_' + col1
    col_name2 = t2 + '_' + col2
    if op == 1:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left <= right:
                new_res.append(list(mat[i]))
    if op == 2:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left >= right:
                new_res.append(list(mat[i]))
    if op == 3:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left != right:
                new_res.append(list(mat[i]))
    if op == 4:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left == right:
                new_res.append(list(mat[i]))
    if op == 5:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left < right:
                new_res.append(list(mat[i]))
    if op == 6:
        for i in range(1, len(mat)):
            left = col_content(mat, arithop1, col_name1, const1, i)
            right = col_content(mat, arithop2, col_name2, const2, i)
            if left > right:
                new_res.append(list(mat[i]))
    datamat = np.array(new_res)
    print(datamat)
    return datamat

def Btree(items,dict,btree_dict, col_dict):
    var = re.split(r'[\s\,\(\)]+', items)
    while '' in var:
        var.remove('')
    ori_table = var[1]
    index_col = var[2]
    table_col = ori_table + '_' + index_col
    col_dict[table_col] = 'Btree'
    col = find_col(dict[ori_table], index_col)
    dict = {}
    for index in range(0, len(col)):
        if col[index] not in dict.keys():
            res = []
            res.append(index)
            dict[col[index]] = res
        else:
            res = dict[col[index]]
            res.append(index)
            dict[col[index]] = res
    t = OOBTree()
    t.update(dict)
    btree_dict[table_col] = t
    return 1


def hash(items,dict,hash_dict, col_dict):
    var = re.split(r'[\s\,\(\)]+', items)
    while '' in var:
        var.remove('')
    ori_table = var[1]
    index_col = var[2]
    table_col = ori_table + '_' + index_col
    col_dict[table_col] = 'hash'
    col = find_col(dict[ori_table], index_col)
    dict = {}
    for index in range(0, len(col)):
        if col[index] not in dict.keys():
            res = []
            res.append(index)
            dict[col[index]] = res
        else:
            res = dict[col[index]]
            res.append(index)
            dict[col[index]] = res
    hash_dict[table_col] = dict
    return 2

def outputtofile(items,dict):
    var = re.split(r'[\s\,\(\)]+', items)
    while '' in var:
        var.remove('')
    ori_table = var[1]
    name = var[2]
    np.savetxt(name+'.txt', dict[ori_table], fmt='%s')
    return 3


def inputfromfile(items):
    table_name = items[0].replace(' ', '')
    var = re.split(r'[\(\)]', items[1])
    filename = str(var[1]) + '.txt'
    table1 = txt_to_matrix(filename)
    all_table_dict[table_name] = table1
    return 4

def select(items,dict):
    res_table = items[0].replace(' ', '')
    if re.search(r'or', items[1]):
        temp = re.split(r'or', items[1])
        var = re.split(r'[\s\,\(\)]+', temp[0])
        var2 = re.split(r'[\s\,\(\)]+', temp[1])
        var = var+var2
        ori_table = var[1]
        data = dict[ori_table]
        res = []
        #res.append(list(data[0]))
        while '' in var:
            var.remove('')
        for i in range(2,len(var)):
            datamat = parse_condition(var[1], var[i], res, data)
            res = datamat
    elif re.search(r'and', items[1]):
        temp = re.split(r'and', items[1])
        var = re.split(r'[\s\,\(\)]+', temp[0])
        var2 = re.split(r'[\s\,\(\)]+', temp[1])
        var = var+var2
        while '' in var:
            var.remove('')
        ori_table = var[1]
        data = dict[ori_table]
        for i in range(2,len(var)):
            res = []
            #res.append(list(data[0]))
            datamat = parse_condition(var[1], var[i], res, data)
            data = np.array(datamat)
            print(data)
    else:
        var = re.split(r'[\s\,\(\)]+', items[1])
        while '' in var:
            var.remove('')
        ori_table = var[1]
        data = dict[ori_table]
        res = []
        #res.append(list(data[0]))
        datamat = parse_condition(var[1], var[2],res, data)
    datamat.insert(0,list(data[0]))
    mat = np.array(datamat)
    dict[res_table] = mat
    return 5

def project(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    #bcomma = items[1].split(',')[0]
    #ori_table = bcomma.split('(')[1]
    res_table = items[0].replace(' ', '')
    ori_table = var[1]
    record = []
    for i in range(2, len(var)):
        col = find_col(dict[ori_table], var[i])
        record.append(col)
    datamat = np.array(record)
    mat = datamat.T
    dict[res_table] = mat
    return 6

def avg(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table = items[0].replace(' ', '')
    col = find_col(dict[ori_table], var[2])
    name = 'avg(' + var[2] + ')'
    record = []
    record.append(name)
    num = list(map(int, col[1:]))
    record.append(mean(num))
    datamat = np.array(record)
    mat = datamat.reshape(datamat.shape[0], 1)
    dict[res_table] = mat
    return 7

def sumgroup(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table_name = items[0].replace(' ', '')
    data = dict[ori_table]
    if len(var) == 4:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])
        y = datalist[0].index(var[2])
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        a = []  # 存储着结果，按照pricerange对qty的测量
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a.append([res_table[0][y], res_table[0][k]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                a.append([sum, res_table[i - 1][k]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 5:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                a.append([sum, res_table[i - 1][k], res_table[i - 1][p]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 6:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] and res_table[i][d] == res_table[i - 1][d]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                a.append([sum, res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 7:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        q = datalist[0].index(var[6])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d], x[q]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d], res_table[0][q]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] \
                    and res_table[i][d] == res_table[i - 1][d] and res_table[i][q] == res_table[i - 1][q]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                a.append([sum, res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d], res_table[i - 1][q]])
                i = i + 1
        res_table = np.asarray(a)
    else:
        print('Error: Please input up to 4 valid columns to caculate sumgroup')
    dict[res_table_name] = res_table
    return 8

def avggroup(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table_name = items[0].replace(' ', '')
    data = dict[ori_table]
    if len(var) == 4:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])
        y = datalist[0].index(var[2])
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        a = []  # 存储着结果，按照pricerange对qty的测量
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a.append([res_table[0][y], res_table[0][k]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                avg = (sum / count)
                a.append([round(avg, 4), res_table[i - 1][k]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 5:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        print(res_table)
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        count = 0
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                avg = sum / count
                a.append([round(avg, 4), res_table[i - 1][k], res_table[i - 1][p]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 6:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        print(res_table)
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] and res_table[i][d] == res_table[i - 1][d]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                avg = (sum / count)
                a.append([round(avg, 2), res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 7:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        q = datalist[0].index(var[6])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d], x[q]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d], res_table[0][q]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] \
                    and res_table[i][d] == res_table[i - 1][d] and res_table[i][q] == res_table[i - 1][q]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                avg = (sum / count)
                a.append(
                    [round(avg, 4), res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d], res_table[i - 1][q]])
                i = i + 1
        res_table = np.asarray(a)
    else:
        print('Error: Please input up to 4 valid columns to caculate avggroup')
    dict[res_table_name]=res_table
    return 9

def join(items, all_table_dict):
    res_table = items[0].replace(' ', '')
    if re.search(r'and', items[1]):
        temp = re.split(r'and', items[1])
        var = re.split(r'[\s\,\(\)]+', temp[0])
        var2 = re.split(r'[\s\,\(\)]+', temp[1])
        var = var+var2
        while '' in var:
            var.remove('')
        table1 = var[1]
        table2 = var[2]
        for i in range(3,len(var)):
            if re.search('=',var[i]):
                mat = join_equal(all_table_dict, table1, table2, var[i])
                print(mat)
        for i in range(3,len(var)):
            op = relop(var[i], 0)
            terms = re.split(r'[\=\>\<\!]+', var[i])
            datamat = join_terms(op, terms, mat)
            mat = datamat
            # if op != 4:
            #     datamat = join_terms(op,terms,mat)
            #     mat = datamat
    else:
        var = re.split(r'[\s\,\(\)]+', items[1])
        while '' in var:
            var.remove('')
        table1 = var[1]
        table2 = var[2]
        mat = join_equal(all_table_dict,table1, table2, var[3])
    all_table_dict[res_table] = mat
    return 10

def sort(items,all_table_dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table_name = items[0].replace(' ', '')
    data = all_table_dict[ori_table]
    order = []
    if len(var) == 3:
        for i in range(0, len(data)):
            order.append(list(data[i]))
            k = order[0].index(var[2])
        for i in range(1, len(order)):
            order[i][k] = int(order[i][k])
        sort = []
        sort.append(order[0])
        temp = (sorted(order[1:], key=(lambda x: [x[k]])))
        for i in range(0, len(temp)):
            sort.append(temp[i])
        res_table = np.asarray(sort)
    elif len(var) == 4:
        for i in range(0, len(data)):
            order.append(list(data[i]))
        k = order[0].index(var[2])
        p = order[0].index(var[3])
        for i in range(1, len(order)):
            order[i][k] = int(order[i][k])
            order[i][p] = int(order[i][p])
        sort = []
        sort.append(order[0])
        temp = (sorted(order[1:], key=(lambda x: [x[k], x[p]])))
        for i in range(0, len(temp)):
            sort.append(temp[i])
        res_table = np.asarray(sort)
    elif len(var) == 5:
        for i in range(0, len(data)):
            order.append(list(data[i]))
        k = order[0].index(var[2])
        p = order[0].index(var[3])
        d = order[0].index(var[4])
        for i in range(1, len(order)):
            order[i][k] = int(order[i][k])
            order[i][p] = int(order[i][p])
            order[i][d] = int(order[i][d])
        sort = []
        sort.append(order[0])
        temp = (sorted(order[1:], key=(lambda x: [x[k], x[p], x[d]])))
        for i in range(0, len(temp)):
            sort.append(temp[i])
        res_table = np.asarray(sort)
    elif len(var) == 6:
        for i in range(0, len(data)):
            order.append(list(data[i]))
        k = order[0].index(var[2])
        p = order[0].index(var[3])
        d = order[0].index(var[4])
        q = order[0].index(var[5])
        for i in range(1, len(order)):
            order[i][k] = int(order[i][k])
            order[i][p] = int(order[i][p])
            order[i][d] = int(order[i][d])
            order[i][q] = int(order[i][q])
        sort = []
        sort.append(order[0])
        temp = (sorted(order[1:], key=(lambda x: [x[k], x[p], x[d], x[q]])))
        for i in range(0, len(temp)):
            sort.append(temp[i])
        res_table = np.asarray(sort)
    elif len(var) == 7:
        for i in range(0, len(data)):
            order.append(list(data[i]))
        k = order[0].index(var[2])
        p = order[0].index(var[3])
        d = order[0].index(var[4])
        q = order[0].index(var[5])
        for i in range(1, len(order)):
            order[i][k] = int(order[i][k])
            order[i][p] = int(order[i][p])
            order[i][d] = int(order[i][d])
            order[i][q] = int(order[i][q])
        sort = []
        sort.append(order[0])
        temp = (sorted(order[1:], key=(lambda x: [x[k], x[p], x[d], x[q]])))
        for i in range(0, len(temp)):
            sort.append(temp[i])
        res_table = np.asarray(sort)
    else:
        print('Error: Please input up to 5 valid columns for sorting')
    all_table_dict[res_table_name] = res_table
    return 11

def movavg(items, all_table_dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table = items[0].replace(' ', '')
    data = all_table_dict[ori_table]
    col = find_col(data, var[2])
    num = list(map(int, col[1:]))
    res = running_mean(num, int(var[3]))
    res.insert(0, 'mov_avg')
    data = np.insert(data, len(data[0]), values=res, axis=1)
    all_table_dict[res_table] = data
    return 12

def movsum(items, all_table_dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table = items[0].replace(' ', '')
    data = all_table_dict[ori_table]
    col = find_col(data, var[2])
    num = list(map(int, col[1:]))
    res = moving_sum(num, int(var[3]))
    res.insert(0, 'mov_sum')
    data = np.insert(data, len(data[0]), values=res, axis=1)
    all_table_dict[res_table] = data
    return 13

def concat(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    table1 = var[1]
    table2 = var[2]
    res_table = items[0].replace(' ', '')
    a= dict[table1]
    b= dict[table2]
    res = np.concatenate((a, b[1:]), axis=0)
    dict[res_table] = res
    return 14

def count(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table = items[0].replace(' ', '')
    col = find_col(dict[ori_table], var[2])
    name = 'count(' + var[2] + ')'
    record = []
    record.append(name)
    num = list(map(int, col[1:]))
    record.append(len(num))
    datamat = np.array(record)
    mat = datamat.reshape(datamat.shape[0], 1)
    dict[res_table] = mat
    return 15

def mysum(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table = items[0].replace(' ', '')
    col = find_col(dict[ori_table], var[2])
    name = 'avg(' + var[2] + ')'
    record = []
    record.append(name)
    num = list(map(int, col[1:]))
    record.append(sum(num))
    datamat = np.array(record)
    mat = datamat.reshape(datamat.shape[0], 1)
    dict[res_table] = mat
    return 16

def countgroup(items,dict):
    var = re.split(r'[\s\,\(\)]+', items[1])
    while '' in var:
        var.remove('')
    ori_table = var[1]
    res_table_name = items[0].replace(' ', '')
    data= dict[ori_table]
    if len(var) == 4:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])
        y = datalist[0].index(var[2])
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        a = []  # 存储着结果，按照pricerange对qty的测量
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a.append([res_table[0][y], res_table[0][k]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                a.append([count, res_table[i - 1][k]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 5:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                a.append([i - 1, res_table[i - 1][k], res_table[i - 1][p]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 6:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] and res_table[i][d] == res_table[i - 1][d]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                a.append([i - 1, res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d]])
                i = i + 1
        res_table = np.asarray(a)
    elif len(var) == 7:
        datalist = []
        for i in range(0, len(data)):
            datalist.append(list(data[i]))
        k = datalist[0].index(var[3])  # groupby的类别1
        p = datalist[0].index(var[4])  # groupby的类别2
        d = datalist[0].index(var[5])  # groupby的类别3
        q = datalist[0].index(var[6])  # groupby的类别3
        y = datalist[0].index(var[2])  # 要被累加的数字
        res_table = []
        res_table.append(datalist[0])
        temp = (sorted(datalist[1:], key=(lambda x: [x[k], x[p], x[d], x[q]])))
        for i in range(0, len(temp)):
            res_table.append(temp[i])
        q = []
        q.append(1)
        for c in range(1, len(res_table)):
            res_table[c][y] = int(res_table[c][y])
        b = 1
        i = 2
        a = []
        a.append([res_table[0][y], res_table[0][k], res_table[0][p], res_table[0][d], res_table[0][q]])
        while i < len(res_table):
            sum = res_table[i - 1][y]
            while i <= len(res_table) - 1 and res_table[i][k] == res_table[i - 1][k] and res_table[i][p] == \
                    res_table[i - 1][p] \
                    and res_table[i][d] == res_table[i - 1][d] and res_table[i][q] == res_table[i - 1][q]:
                sum = sum + res_table[i][y]  # 求具体的sum数值
                i = i + 1
            else:
                q.append(i)
                count = q[len(q) - 1] - q[len(q) - 2]
                a.append([i - 1, res_table[i - 1][k], res_table[i - 1][p], res_table[i - 1][d], res_table[i - 1][q]])
                i = i + 1
        res_table = np.asarray(a)
    else:
        print('Error: Please input up to 4 valid columns to caculate countgroup')
    dict[res_table_name] = res_table
    return 17

if __name__ == '__main__':
    file = open(opt.filename)
    all_table_dict = {}
    hash_index_dict = {}
    Btree_index_dict = {}
    col_dict = {}
    f = open('sample_output.txt', 'a')
    while 1:
        flag = 0
        line = file.readline()
        #print(line)
        if not line:
            break
        #operation = input(str(line)).content[0]
        operation = str(line).split('/')[0]

        if re.match(r'\s+', operation):
            print('do nothing')
        else:
            operation = operation.replace(' ', '')
            f.write('\n\n\n' + operation)
            print(operation)  # do something
            if re.match(r'Btree', operation):
                print('btree')
                start_d = time.time()
                flag = Btree(operation,all_table_dict,Btree_index_dict, col_dict)
                end_d = time.time()
                time_d = end_d - start_d
                f.write('\n' + 'Btree running time:' + str(time_d) + 's')
                print(flag)
            elif re.match(r'Hash', operation):
                print('hash')
                start_d = time.time()
                flag = hash(operation,all_table_dict,hash_index_dict, col_dict)
                end_d = time.time()
                time_d = end_d - start_d
                f.write('\n' + 'Hash running time:' + str(time_d) + 's')
                print(flag)
            elif re.match(r'outputtofile', operation):
                print('outputtofile')
                start_d = time.time()
                flag = outputtofile(operation,all_table_dict)
                end_d = time.time()
                time_d = end_d - start_d
                f.write('\n' + 'outputtofile running time:' + str(time_d) + 's')
                print(flag)
            else:
                items = re.split(r'[\:]+', operation)
                print(items[1])
                if re.match('\=\s*'+r'inputfromfile', items[1]):
                    print('inputfrom')
                    start_d = time.time()
                    flag = inputfromfile(items)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'inputfromfile running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'select', items[1]):
                    print('select')
                    start_d = time.time()
                    flag = select(items,all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'select running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'project', items[1]):
                    print('project')
                    start_d = time.time()
                    flag = project(items,all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'project running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'avg', items[1]):
                    if re.match('\=\s*' + r'avggroup', items[1]):
                        print('avggroup')
                        start_d = time.time()
                        flag = avggroup(items, all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'avggroup running time:' + str(time_d) + 's')
                        print(flag)
                    else:
                        print('avg')
                        start_d = time.time()
                        flag = avg(items,all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'avg running time:' + str(time_d) + 's')
                        print(flag)

                if re.match('\=\s*'+r'join', items[1]):
                    print('join')
                    start_d = time.time()
                    flag = join(items, all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'join running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'sort', items[1]):
                    print('sort')
                    start_d = time.time()
                    flag = sort(items,all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'sort running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'movavg', items[1]):
                    print('movavg')
                    start_d = time.time()
                    flag = movavg(items, all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'movavg running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'movsum', items[1]):
                    print('movsum')
                    start_d = time.time()
                    flag = movsum(items, all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'movsum running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'concat', items[1]):
                    print('concat')
                    start_d = time.time()
                    flag = concat(items,all_table_dict)
                    end_d = time.time()
                    time_d = end_d - start_d
                    f.write('\n' + 'concat running time:' + str(time_d) + 's')
                    print(flag)
                if re.match('\=\s*'+r'count', items[1]):
                    if re.match('\=\s*' + r'countgroup', items[1]):
                        print('countgroup')
                        start_d = time.time()
                        flag = sumgroup(items, all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'countgroup running time:' + str(time_d) + 's')
                        print(flag)
                    else:
                        print('count')
                        start_d = time.time()
                        flag = count(items,all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'count running time:' + str(time_d) + 's')
                        print(flag)
                if re.match('\=\s*'+r'sum', items[1]):
                    if re.match('\=\s*' + r'sumgroup', items[1]):
                        print('sumgroup')
                        start_d = time.time()
                        flag = sumgroup(items, all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'sumgroup running time:' + str(time_d) + 's')
                        print(flag)
                    else:
                        print('sum')
                        start_d = time.time()
                        flag = mysum(items,all_table_dict)
                        end_d = time.time()
                        time_d = end_d - start_d
                        f.write('\n' + 'sum running time:' + str(time_d) + 's')
                        print(flag)


    print(all_table_dict.keys())
