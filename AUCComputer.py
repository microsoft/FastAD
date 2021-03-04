# -*- coding: utf-8 -*-
"""
Created on 2017

E-mail:zhipluo@microsoft.com
@author: zhipluo
"""

import pandas as pd
import numpy as np

IYBucket = 100

def target(val, labels = []):
    T = []
    for x in val:
        t = []
        if x[0] in labels:
            t.append(1)
        else:
            t.append(0)
        t.append(x[1])
        T.append(t)
    return T


def cal_auc(val):
    df = pd.DataFrame(val, columns=['label', 'score'])
    df['score'] = df['score'].astype(float)
    df.sort_values(by=['score'], ascending=[0], inplace=True)
    x = df['label'].values
    cnt = 1.0
    totalImp = len(x)
    step = 1.0/IYBucket
    totalClicks = np.array(x).sum()
    # print ('AUC boundary:', 1. * totalClicks / totalImp / 2 , 1. - 1. * totalClicks / totalImp / 2)
    if totalClicks < 1:
        return -1
    
    for i in range(1, IYBucket):
        preImp = int(i*step*totalImp)
        tauc = np.array(x[0:preImp]).sum()/totalClicks
        cnt += tauc

    cnt -= 0.5
               
    return cnt


'''
def cal_auc(val):
    df = pd.DataFrame(val, columns=['label', 'score'])
    df.sort_values(by=['score'], ascending=[0], inplace=True)
    x = df['label'].values

    cnt = 1.0
    totalImp = len(x)
    step = 1.0/IYBucket
    
    imps = [x[0]]
    for i in range(1, len(x)):
        imps.append( x[i] + imps[i-1])
    totalClicks = imps[len(x)-1]
    
    if totalClicks < 1:
        return -1
    
    for i in range(1, IYBucket):
        indx = int(i*step*totalImp)
    
        tauc = imps[indx-1]/totalClicks
        cnt += tauc
        #print (i, tauc, preImp, np.array(x[0:preImp]).sum())
                        
    cnt -= 0.5
               
    return cnt    
'''
 
#m_AdCopyJudgment == "Good" OR m_AdCopyJudgment == "Bad" OR m_AdCopyJudgment == "Fair" OR m_AdCopyJudgment == "Excellent" OR m_AdCopyJudgment == "Perfect";   
def good_bad_auc(val1):  #输入两列:label,score
    val = []
    valid_label = ['Good', 'Bad', 'Fair', 'Excellent', 'Perfect']
    for i in range(len(val1)):
        if val1[i][0] in valid_label:
            val.append(val1[i])

    bad = target(val, ['Bad'])
    bad_auc = cal_auc(bad)
    
    #good = target(val, ['Good', 'Excellent'])
    good = target(val, ['Good', 'Excellent', 'Perfect'])
     
    good_auc = cal_auc(good)
    
    print ('val size:', len(val))
    
    return bad_auc, good_auc
    

def read_file(file, sep='\t'):
    f = open(file)
    X = []
    for line in f:
        x = line[:-1].split(sep)
        X.append(x)
    f.close()
    df = pd.DataFrame(X)
    return df

import datetime    
def main():
    print ('load file...')
    #df = pd.read_csv('validation.tsv', sep='\t', header = None)
    df = read_file('data/validation_qk.tsv')
    
    #print (df.shape[0], df.shape[1])
    
    df = df[[0, 1]]
    print ('start cal auc...')
    begin = datetime.datetime.now()
    bad_auc, good_auc = good_bad_auc(df.values)
    end = datetime.datetime.now()
    
    print ('run time:', end-begin)
    # print (set(df[2].values))
    print ('bad auc:', bad_auc, '  good auc:', good_auc)
    
    
if __name__ == "__main__":
    main()

