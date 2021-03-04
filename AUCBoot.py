# -*- coding: utf-8 -*-
"""
Created on 2018-02-26

E-mail:zhipluo@microsoft.com
@author: zhipluo
"""

import pandas as pd
import numpy as np

#np.random.seed(77)

def loadAndFormatData(path, df=None):
    labels = ['disjoint', 'aggregator', 'spam', 'overlap', 'subset', 'superset', 'same', 'detrimental', 'verybad', 'very bad', 'bad', 'fair', 'good', 'excellent', 'perfect']

    if len(path) > 0:
        df = pd.read_csv(path, sep='\t')
        
    names = df.columns
    cols = {}
    for i in range(len(names)):
        cols[names[i]] = names[i].replace('m:','')
    df = df.rename(columns=cols)
    
    df = df.fillna("")

    df['MatchType'] = df['MatchType'].apply(lambda x:str(x).lower())
    df['Rating'] = df['Rating'].apply(lambda x:str(x).lower())    
    df['AdCopyJudgment'] = df['AdCopyJudgment'].apply(lambda x:str(x).lower())

    colnames = df.columns
    if 'RankScore' in colnames:
        df['RelevanceScore'] = ( df['RankScore']/1e+6 ) - 1000
    elif 'Predicted' in colnames:
        df['RelevanceScore'] = df['Predicted']
    
    df = df[ (df['Rating'].isin(labels)) & (df['AdCopyJudgment']).isin(labels)]
    return df
    
    
lpDefect = ['disjoint', 'aggregator', 'spam', 'verybad', 'very bad', 'detrimental']
acDefect = ['disjoint', 'aggregator', 'spam', 'verybad', 'very bad', 'bad', 'detrimental']


def getDefectData(data):
    return data[ (data.isin(lpDefect)) | (data.isin(acDefect)) ]

def getNonDefectData(data):
    return data[ ~( (data.isin(lpDefect)) | (data.isin(acDefect)) ) ]



def gsub(ins, rs, text):
    for c in ins:
        text = text.replace(c, rs)
    return text
    

def starts_with(text, pattern):
    return text[1:len(pattern)] == pattern

#====
def removeRowsByRating(data, removeRatings):
    data = data[ (~(data['Rating'].isin(removeRatings))) & (~(data['AdCopyJudgment'].isin(removeRatings)))  ]
    
    allRemoveRows = []
    for ratingStr in removeRatings:
        if starts_with(ratingStr, 'lp='):
            rating = 1
    
            
def getRatings(text):
    text = text.lower()        



def myauc(x, y):
    #print('myauc:', 0.5*sum(np.diff(x) * (y[1:] + y[:-1])))
    return 0.5*sum(np.diff(x) * (y[1:] + y[:-1]))
    

def myaucPR(rec, prec):
    if rec[0] != 0.0:
        rec = np.insert(rec, 0, 0.0)
        prec = np.insert(prec, 0, 1.0)
    
    if rec[-1] != 1.0:
        rec = np.append(rec, 1.0)
        prec = np.append(prec, 0.0)
    
    return myauc(rec, prec)
    

def myaucROC(fpr, rec):
    if fpr[0] != 0.0:
        rec = np.insert(rec, 0, 0.0)
        fpr = np.insert(fpr, 0, 0.0)
        
    if fpr[-1] != 1.0:
        rec = np.append(rec, 1.0)
        fpr = np.append(fpr, 1.0)
    return myauc(fpr, rec)
    
from sklearn.metrics import roc_auc_score
def PRnROCAUC(data):
    N = data.shape[0]
    
    data = data.reset_index(drop=True)
    
    #print (data[['predictions', 'labels']])
    roc_auc = roc_auc_score(data['labels'].values, data['predictions'].values)
    #print ('auc:', )
    
    
    labOrdered = data.sort_values(by=['predictions'], ascending=[0])['labels'].values
    
                                  
    PosRes = np.arange(1, N+1)
    PosLab = labOrdered.sum()
    NegLab = N - PosLab
                                  
    TP = labOrdered.cumsum()
    FP = PosRes - TP
    
    Prec = TP / PosRes
    Rec = TP / PosLab
    FPR = FP / NegLab
    

    aucPR = myaucPR(Rec, Prec)
    #aucROC = myaucROC(FPR, Rec)
    aucROC = roc_auc
    
    #print({'aucPR':aucPR, 'aucROC':aucROC, 'Prec':Prec, 'Rec':Rec, 'FPR':FPR})
    return {'aucPR':aucPR, 'aucROC':aucROC, 'Prec':Prec, 'Rec':Rec, 'FPR':FPR}

    
def func(x):
    if x[0] in lpDefect or x[1] in acDefect:
        return 0
    return 1

def PRnROCAUCBootStrap(data, replicas):
    #N = data.shape[0]    
    #dataBare = pd.DataFrame([1]*N, columns=['labels'])
    dataBare = data[['RelevanceScore']].rename(columns={'RelevanceScore':'predictions'})
    dataBare['labels'] = data[['Rating', 'AdCopyJudgment']].apply(func, axis=1)

    #print ('dataBare:')
    #print (dataBare)
    
    res1 = PRnROCAUC(dataBare)
    #print (res1)
    
    resDF = {'aucPR':res1['aucPR'], 'aucROC':res1['aucROC']}
    
    ####
    #if replicas > 1:
    ####
    
    return {'resDF':resDF, 'labels':dataBare['labels'].values, 'aucPR':res1['aucPR'], 'aucROC':res1['aucROC'], 'Prec':res1['Prec'], 'Rec':res1['Rec'], 'FPR':res1['FPR']}

            
    
def plotPrecisionRecall(bslData, newData, matchType, replicas):
    AUC = {}
    #epsilon = 1e-10
    
    bslRes = PRnROCAUCBootStrap(bslData, replicas)
    AUC['BaselinePR'] = np.mean(bslRes['resDF']['aucPR'])
    AUC['Baseline'] = np.mean(bslRes['resDF']['aucROC'])
    
    
    newRes = PRnROCAUCBootStrap(newData, replicas)
    AUC['NewModelPR'] = np.mean(newRes['resDF']['aucPR'])
    AUC['Baseline'] = np.mean(newRes['resDF']['aucROC'])

    #print(AUC)
    return AUC
    
    
    
def run(baseFile, newFile, removeRatings, replicas):
    bslData = loadAndFormatData(baseFile)
    newData = loadAndFormatData(newFile)
    
    #print (bslData)
    print ('replicas: ', replicas, '\n')
    
    

    if bslData.shape[0] < 2 or newData.shape[0] < 2:
        return 0

    if len(removeRatings) > 0:
        bslData = removeRowsByRating(bslData, removeRatings)
        newData = removeRowsByRating(newData, removeRatings)
        
    AUC = {}
    AUC['AllMatches'] = plotPrecisionRecall(bslData, newData, 'AllMatches', replicas)  
    
    
    
    for matchType in set(bslData['MatchType'].values):
        print (matchType)
        mbsl = bslData[bslData['MatchType']== matchType]
        mnew = newData[newData['MatchType']== matchType]

        AUC[matchType] = plotPrecisionRecall(mbsl, mnew, matchType, replicas)

    print (AUC)
    
    

def cal_PRAUC(df):
    df = loadAndFormatData('', df)
    print ('replicas: ', replicas, '\n')

    if df.shape[0] < 2:
        return 0

    if len(removeRatings) > 0:
        df = removeRowsByRating(df, removeRatings)
        
    Res = PRnROCAUCBootStrap(df, replicas)
    all_pr = np.mean(Res['resDF']['aucPR'])

    dt = df[df['MatchType']== 'smartmatch']
    Res = PRnROCAUCBootStrap(dt, replicas)
    sm_pr = np.mean(Res['resDF']['aucPR'])
    
    return all_pr, sm_pr

def cal_AUC(df):
    df = loadAndFormatData('', df)
    print ('replicas: ', replicas, '\n')

    if df.shape[0] < 2:
        return 0

    if len(removeRatings) > 0:
        df = removeRowsByRating(df, removeRatings)
        
    Res = PRnROCAUCBootStrap(df, replicas)
    all_pr = np.mean(Res['resDF']['aucPR'])
    all_roc = np.mean(Res['resDF']['aucROC'])

    dt = df[df['MatchType']== 'smartmatch']
    Res = PRnROCAUCBootStrap(dt, replicas)
    sm_pr = np.mean(Res['resDF']['aucPR'])
    sm_roc = np.mean(Res['resDF']['aucROC'])
    
    return [all_pr, sm_pr], [all_roc, sm_roc]
    
removeRatings = []
replicas = 100


def main():
    baseFile = 'baseline.txt'
    newFile = 'newModel.txt'
    run(baseFile, newFile, removeRatings, replicas)
    
    #df = pd.read_csv('baseline.txt', sep='\t')
    #print (cal_PRAUC(df))

    #df = pd.read_csv('newModel.txt', sep='\t')
    #print (cal_PRAUC(df))
    
if __name__ == '__main__':
    main()


    