""" tbmlib.py
 i.		Data preprocessing for RNN training 
 ii.	Missing Data Handling (TBM, Missing Indicators, Forward-filling, Expert Imputation)
 iii.	Evaluation (k-fold cross validation, analyze the results files)

 Author: Yeo Jin Kim
 updated: May. 10, 2018
"""

import math
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix
import os

filepath = 'data/'
outpath = filepath+"out/"
pid = 'VisitIdentifier'       # primary id
label = 'ShockFlag'
timefeat = 'MinutesFromArrival'
idfeat = [pid] 
labels = [label]
DEBUG = False

# ----------------------------------------
# Data Generation for RNN training
# 1. cut the last observations
# 2. add all the cumulative rows for every timestep for each visit

# A. Visit-level sequence data generation
# make X as 2D array
def makeXY(df, feat, pid):
    X = []
    Y = []
    vids, pos_vid, neg_vid = get_vids(df, pid)
    if DEBUG:
        print("--- makeXY")
        print("pos: {0}, neg: {1}".format(len(pos_vid), len(neg_vid)))
    
    for vid in sorted(vids):
        X.append(np.array(df.loc[df[pid] == vid, feat])) 
        if vid in pos_vid:
            Y.append(1)
        else:
            Y.append(0)

    return X, Y
    
        
# B. Event-level sequence data generation
# predict the next label: shift the labels with 1 timestep backward
def makeXY_event(df, feat, pid, label): 
    X = []
    Y = []
    eids = df[pid].unique()
    for eid in eids:
        edf = df[df[pid] == eid]
        tmp = np.array(edf[feat])
        for i in range(len(tmp)-1):
            X.append(tmp[:i+1])
        Y += edf[1:len(edf)][label].tolist()
    return X, Y


# -------------------------------------  
# Dataframe: load & save    
def loaddf(filename):
    df = pd.read_csv(filename, header=0)
    return df
	
def loaddf_index(filename):
    df = pd.read_csv(filename, header=0, index_col = 0)
    return df

def savedf(df, filename):
    print('* save file: '+filename)
    time = datetime.now().strftime('%m%d%H') # %m%d_%H
    df.to_csv(filename+'_'+time+'.csv', index = False) 

def savedf_date(df, filename):
    print('* save file: '+filename)
    time = datetime.now().strftime('%m%d') # %m%d_%H
    df.to_csv(filename+'_'+time+'.csv', index = False) 
    
def savedf_index(df, filename):
    print('* save file: '+filename)
    time = datetime.now().strftime('%m%d_%H')
    df.to_csv(filename+'_'+time+'.csv', index = True)
    
def sortbytime(df):
    pd.to_numeric(df[timefeat], errors='coerce')
    df = df.sort_values(by=[label, timefeat])
    df = df.reset_index(drop=True)
    return df

# Get primary ids for datasets
def get_vids(df, pid):        
    id = df[pid].unique().tolist()
    posid = df[df[label]==1][pid].unique().tolist()
    negid = [x for x in id if x not in posid]
    return id, posid, negid

# merge labels to features
def mergeXY(Xdf, Ydf):
    Xdf = sortbytime(Xdf)
    Ydf = sortbytime(Ydf)
    Xdf[label] = Ydf[label]  
    return Xdf

#--------------------------------------------------------------
# Missing Data Handling
# baseline: 1) set missing indicator (option: set_mi = True)
#           2) standardize
#           3) carry fowrad + normal value filling

# ----------------------------------------------------------------
# Expert imputation : 8 hr for vital signs, 24 hr for lab tests
def clinic_ffill(traindf, valdf, testdf, feat, vitals, labs, pid):  
    # carry fowrad 
    traindf = carryFwd_allfeat(traindf, False, '', vitals, labs, pid)
    valdf = carryFwd_allfeat(valdf, False, '', vitals, labs, pid)
    testdf = carryFwd_allfeat(testdf, False, '', vitals, labs, pid)
    # normal value filling (normal value is 0 due to standardization)
    traindf[feat] = traindf[feat].fillna(0)
    valdf[feat] = valdf[feat].fillna(0)
    testdf[feat] = testdf[feat].fillna(0)    
    return traindf, valdf, testdf

# carry forward for all the features within clinical hours
def carryFwd_allfeat(df, savefile_mode, filename, vitals, labs, pid):
    #print('\n* carry forward features')
    # 1. Carryforward for vital signs until the new measurement happended for all the features 
    for f in vitals:
        df = carryFwd_minutes(df, f, 480, pid) # 8 hr for vital signs
    for f in labs:
        df = carryFwd_minutes(df, f, 1440, pid) # 24 hr for lab results
        
    if savefile_mode:
        savedf(df, filepath, filename)
    return df

def carryFwd_minutes(df, feat, cftime, pid):
    # for each feature, record the observation time
    df.loc[pd.notnull(df[feat]),feat+'_t'] = df.loc[:,timefeat] # timefeat = 'MinutesFromArrival'
    df.loc[:,feat] = df.groupby(pid)[feat].ffill()
    df.loc[:,feat+'_t'] = df.groupby(pid)[feat+'_t'].ffill()
    df.loc[df[timefeat]- df[feat+'_t'] > cftime, feat] = np.nan
    df = df.drop(feat+'_t', axis = 1)
    return df
    
# ----------------------------------------------------------------    
# zero-filling
def zero_fill(traindf,valdf, testdf,feat):
    traindf[feat] = traindf[feat].fillna(0)
    valdf[feat] = valdf[feat].fillna(0)
    testdf[feat] = testdf[feat].fillna(0)
    return traindf, valdf, testdf

# ----------------------------------------------------------------
# forward-filling + zero-filling
def ffill(traindf,valdf, testdf, pid, feat):
    traindf[feat] = traindf.groupby(pid)[feat].ffill()
    valdf[feat] = valdf.groupby(pid)[feat].ffill()
    testdf[feat] = testdf.groupby(pid)[feat].ffill()
    # set 0 (mean or normal value) for the remaining missing values
    traindf[feat] = traindf[feat].fillna(0)
    valdf[feat] = valdf[feat].fillna(0)
    testdf[feat] = testdf[feat].fillna(0)
    return traindf, valdf, testdf

# ----------------------------------------------------------------
# backward-filing + zero-filling
def bfill(traindf,valdf, testdf, pid, feat):
    traindf[feat] = traindf.groupby(pid)[feat].bfill()
    valdf[feat] = valdf.groupby(pid)[feat].bfill()
    testdf[feat] = testdf.groupby(pid)[feat].bfill()
    # set 0 (mean or normal value) for the remaining missing values
    traindf[feat] = traindf[feat].fillna(0)
    valdf[feat] = valdf[feat].fillna(0)
    testdf[feat] = testdf[feat].fillna(0)
    return traindf, valdf, testdf
   



# ----------------------------------------------------------------
# Temporal Belief Memory 
# according to the belief propagation mode: 
# .0 : forward only within a reliable time window
# .25: 25% of backward & 75% of forward within a reliable time window
# .5 : 50% of backward & 50% of forward within a reliable time window
# .75: 75% of backward & 25% of forward within a reliable time window
# 1  : backward only within a reliable time window
# 
# * Beta calculation
# Note that temporal belief function, belief_t = exp^(-beta * |delta_t| / tau) and we set the threshold to 0.5 for activation
# Thus, beta = ln(2)/propagation_portion because exp^(-beta * propagation_portion) = 0.5 (threshold)
# e.g. TBM.75 mode: tau = 120 minutes, back portion of delta_t = 90 min, forward portion of delta_t = 30 min
# exp^(-beta * 90/120) = exp^(-beta * 0.75 ) = 0.5 ==> beta = ln(2)/0.75 = 0.924196..
def make_belief(df, pid, feat, taus, mode):
    df = setmi(df, feat)# set missing indicators
    if mode == '0':    # forward
        df = ffill_belief(df, pid, feat, taus, beta = math.log(2))   
    elif mode == '1':  # backward
        df = bfill_belief(df, pid, feat, taus, beta = math.log(2))
    elif mode == '.25': # TBM.25: for overlap period, forward has a priority (forward -> backward propagation)
        df = ffill_belief(df, pid, feat, taus, beta = math.log(2)/0.75) 
        df = bfill_belief(df, pid, feat, taus, beta = math.log(2)/0.25) 
    elif mode == '.75': # TBM.75: for overlap period, backward has a priority (backward -> forward)
        df = bfill_belief(df, pid, feat, taus, beta = math.log(2)/0.25)
        df = ffill_belief(df, pid, feat, taus, beta = math.log(2)/0.75) 
    elif mode == '.5':  # TBM.5: forward has a priority
        df = ffill_belief(df, pid, feat, taus, beta = math.log(2)/0.5) 
        df = bfill_belief(df, pid, feat, taus, beta = math.log(2)/0.5)
    else:
        print("ERROR: there is no mode of '{0}'".format(mode))
        
    df[feat] = df[feat].fillna(0)          # fill 0 for remaining nulls (out of reliable windows)  
    return df

# ----------------------------------------------------------------
# TBM Belief mode: forward-filling with belief    
def ffill_belief(df, pid, feat, taus, beta):
    df[feat] = df.groupby(pid)[feat].ffill()
    i = 0
    for f in feat:
        df['last_obtime'] = np.nan # set the field of last_observed_time
        df.loc[pd.notnull(df[f]), 'last_obtime'] = df[pd.notnull(df[f])][timefeat] # copy observed_time
        df['last_obtime'] = df.groupby(pid)['last_obtime'].ffill() # carry forward observed_time until the next observation
        bi = step(np.e**(beta*(-(df[timefeat]-df.last_obtime)/taus[i])))
        df[f] = df[f] * bi 
        i += 1
    df = df.drop(['last_obtime'], axis = 1)
    return df

# ----------------------------------------------------------------
# TBM Belief mode: backward-filling with belief
def bfill_belief(df, pid, feat, taus, beta):
    df[feat] = df.groupby(pid)[feat].bfill()
    i = 0
    for f in feat:
        df['last_obtime'] = np.nan # set the field of last_observed_time
        df.loc[pd.notnull(df[f]), 'last_obtime'] = df[pd.notnull(df[f])][timefeat] # copy observed_time
        df['last_obtime'] = df.groupby(pid)['last_obtime'].bfill() # carry forward observed_time until the previous observation
        bi = step(np.e**(beta*(-(df.last_obtime-df[timefeat])/taus[i])))
        df[f] = df[f] * bi
        i += 1
        # belief for bfill = not based on cumulative stimula, but only temporal interval, which means it's same to ffill
    df = df.drop(['last_obtime'], axis = 1)
    return df

# ----------------------------------------------------------------
# Initialize Tau: the average sampling frequencies, and measuring counts for the features
def init_tau(traindf, pid, time_feat, feat, sample_size, trace):
    # get tau
    print("feat: ",feat) 
    tau, measureNum =  getSampleFreq(traindf, feat, sample_size, pid, time_feat, trace) 
    print('tau: {0}\n measureNum: {1}'.format(tau, measureNum))

    # Make time_dataframe
    timedf = pd.DataFrame(columns = feat, index = ['tau_init', 'tau'])
    timedf.loc['tau_init',feat]=tau   
    timedf.loc['tau',feat]=tau
    #savedf_index(timedf, filepath+'timedf')
    #df.to_csv(filepath+'out/'+filename+'_'+time+'.csv', index = False)
    return tau, timedf   
    
#----------------------------------
# Preprocessing

# standardization for training & test data
def standardize(traindf, testdf, numfeat, savefile_mode):
    trainmean = traindf[numfeat].mean()
    trainstd = traindf[numfeat].std()
    traindf.loc[:,numfeat] = (traindf.loc[:,numfeat] - trainmean[numfeat]) / trainstd[numfeat] # standardize the whole numeric data with train mean & std
    testdf.loc[:,numfeat]  = (testdf.loc[:,numfeat] - trainmean[numfeat]) / trainstd[numfeat]

    # save the data files
    if savefile_mode:
        savedf(traindf, filepath,'traindf_std')
        savedf(testdf, filepath,'testdf_std')
    return traindf, testdf

# standardization for training, validation & test data
# from sklearn.preprocessing import StandardScaler 
#     e = StandardScaler(with_mean=False)
#     traindf[numfeat] = e.fit_transform(traindf[numfeat])
#     valdf[numfeat] = e.transform(valdf[numfeat])
#     testdf[numfeat] = e.transform(testdf[numfeat])
# For sparse data, it should pass 'with_mean=False', but still does not accept null values. 
# if we use StandardScaler, we standardize test data with the mean and std of training data
def standardize_all(traindf, valdf, testdf, numfeat, savefile_mode):
    trainmean = traindf[numfeat].mean()
    trainstd = traindf[numfeat].std()
    traindf.loc[:,numfeat] = (traindf.loc[:,numfeat] - trainmean[numfeat]) / trainstd[numfeat] # standardize the whole numeric data with train mean & std
    valdf.loc[:,numfeat]  = (valdf.loc[:,numfeat] - trainmean[numfeat]) / trainstd[numfeat]
    testdf.loc[:,numfeat]  = (testdf.loc[:,numfeat] - trainmean[numfeat]) / trainstd[numfeat]

    # save the data files
    if savefile_mode:
        savedf(traindf, filepath,'traindf_std')
        savedf(valdf, filepath, 'valdf_std')
        savedf(testdf, filepath,'testdf_std')
    return traindf, valdf, testdf

# normalization for training & test data
def normalize(traindf, testdf, numfeat, savefile_mode):
    trainmax = traindf[numfeat].max()
    trainmin = traindf[numfeat].min()
    diff = trainmax-trainmin
    traindf.loc[:,numfeat] = (traindf.loc[:,numfeat] - trainmin) / diff # normalize the whole numeric data with train mean & std
    testdf.loc[:,numfeat]  = (testdf.loc[:,numfeat] - trainmin) / diff

    # save the data files
    if savefile_mode:
        savedf(traindf, filepath,'traindf_norm')
        savedf(testdf, filepath,'testdf_norm')
    return traindf, testdf
    

    
# ----------------------------------------------    
## Missing Data Handling

# 1. set the missing indicators
# 2. impute
#    1) set the currently observed time
#    2) observed_value
#    3) its belief: calcuate the belief of the previously observed value by feature
#    4) belief_flag: get the flag of belief with the threshold == 0.5
# 3. set 0 for remaining nulls

def step(x):
    return 1. * (x >= 0.5)

 # 1. set missing indicators
def setmi(df, feat):
    for f in feat:
        df.loc[pd.notnull(df[f]), f+'_mi'] = 0
        df.loc[pd.isnull(df[f]), f+'_mi'] = 1
    return df


# Get the average sampling(measuring) frequencies for the given columns
# Use them to initialize the taus. 

def getSampleFreq(df, col, sample_num, pid, time_feat, trace):
    numCol = np.size(col) 
    spfreq = np.zeros(numCol)          # sampling freq = the avg. time interval between measurements by feature
    visitNum = np.zeros(numCol)        # the number of visit hit by feature
                                       # (feautres not measured within an entire visit = missing variable)
    vids = df.sample(sample_num)[pid].unique()
    print("sample_num: {0}".format(len(vids)))
    
    for vid in sorted(vids):
        vdf = df.loc[df[pid] == vid]
        firstMeasure = np.zeros(numCol)    # the first time_feat by feature within each visit.
        lastMeasure = np.zeros(numCol)     # the last time_feat by feature within each visit.
        featEventNum = np.zeros(numCol) # the total number of measurement by feature within each visit.
        
        for i in range(numCol):
            obsrvidxs = vdf[pd.notnull(vdf[col[i]])].index
            featEventNum[i] = len(obsrvidxs)
            if featEventNum[i] > 0:
                firstMeasure[i] = vdf.loc[obsrvidxs[0]][time_feat]
                lastMeasure[i] = vdf.loc[obsrvidxs[-1]][time_feat]
            #print(firstMeasure[i], lastMeasure[i])
            if featEventNum[i] > 1: 
                curspfreq = (lastMeasure[i] - firstMeasure[i]) / featEventNum[i] # samplingfreq for the current visit
                spfreq[i] = (visitNum[i] * spfreq[i] + curspfreq) / (visitNum[i]+1) # spfreq for the whole visits
                visitNum[i] +=1
        
        if trace: 
            print("\n VisitID: {0}".format(vid))
            print("firstMeasure: {0}, lastMeasure: {1}".format(firstMeasure, lastMeasure))
            print(spfreq)

    spfreq[spfreq == 0] = 1 # avoid to divide by 0
    return spfreq, visitNum


# -------------------------
# Evaluation

# Stratified sampling for k-fold cross validation
# 1) set the missing indicators
# 2) split tratified kfold 
# 3) standardize the training and testing set of each fold, using the training set of each fold

def stratified_kfold(df, k, pid, DEBUG):
    print("Data splitting for a stratified ",k,"-fold cross validation")
    vids = df[pid].unique().tolist()
    posvid = df[df[label] == 1][pid].unique().tolist()
    negvid = [v for v in vids if v not in posvid]
    random.shuffle(posvid)
    random.shuffle(negvid)
    random.shuffle(vids)
    
    test_posnum = int(len(posvid) / k) #val_posnum = test_posnum
    test_negnum = int(len(negvid) / k) #val_negnum = test_negnum
    
    train_posnum = len(posvid) - test_posnum*2
    train_negnum = len(negvid) - test_negnum*2
    print('posvid:{0}, negvid:{1}, test/val_posvid:{2}, test/val_negvid:{3}, train_posvid:{4}, train_negvid:{5}'
          .format(len(posvid), len(negvid), test_posnum, test_negnum, train_posnum, train_negnum))

    test_k_vids = []
    val_k_vids = []
    train_k_vids = []
    for i in range(k):
        test_vids = posvid[i*test_posnum:(i+1)*test_posnum] + negvid[i*test_negnum:(i+1)*test_negnum]
        train_pos = [v for v in posvid if v not in test_vids]
        train_neg = [v for v in negvid if v not in test_vids]
        random.shuffle(train_pos)
        random.shuffle(train_neg)
        val_vids = train_pos[0:test_posnum] + train_neg[0:test_negnum]
        train_vids = [v for v in vids if v not in test_vids]
        train_vids = [v for v in train_vids if v not in val_vids]
        #print("train_pos:{0},train_neg:{1}, val_vids:{2}, test_vids:{3}".format(len(train_pos), len(train_neg), len(val_vids), len(test_vids)))
        test_k_vids.append(test_vids)
        val_k_vids.append(val_vids)
        train_k_vids.append(train_vids)
        if DEBUG:
            print("train: pos({0}), neg({1})".format(len(train_pos), len(train_neg)))
            test_pos = df[(df[label]==1) & (df[pid].isin(test_vids))][pid].unique().tolist()            
            print("test : pos({0}), neg({1})".format(len(test_pos), len(test_vids)-len(test_pos)))
        
    return train_k_vids, val_k_vids, test_k_vids

# training 70%, validation 10%, Test 20%
def stratified_kfold_traing70(df, k, pid, DEBUG):
    print("Data splitting for a stratified ",k,"-fold cross validation")
    vids = df[pid].unique().tolist()
    posvid = df[df[label] == 1][pid].unique().tolist()
    negvid = [v for v in vids if v not in posvid]
    random.shuffle(posvid)
    random.shuffle(negvid)
    random.shuffle(vids)
    
    test_posnum = int(len(posvid) / k) #val_posnum = test_posnum
    test_negnum = int(len(negvid) / k) #val_negnum = test_negnum
    val_posnum = int(test_posnum / 2)
    val_negnum = int (test_negnum / 2)
    
    train_posnum = len(posvid) - test_posnum - val_posnum
    train_negnum = len(negvid) - test_negnum - val_negnum
#     print('positive vid:{0}, negvid:{1}, test/val_posvid:{2}, test/val_negvid:{3}, train_posvid:{4}, train_negvid:{5}'
#           .format(len(posvid), len(negvid), test_posnum, test_negnum, train_posnum, train_negnum))
    print('Positive vid: {}, train: {}, val: {}, test: {}'.format(len(posvid), train_posnum, val_posnum, test_posnum))
    print('Negative vid: {}, train: {}, val: {}, test: {}'.format(len(negvid), train_negnum, val_negnum, test_negnum))

    test_k_vids = []
    val_k_vids = []
    train_k_vids = []
    for i in range(k):
        test_vids = posvid[i*test_posnum:(i+1)*test_posnum] + negvid[i*test_negnum:(i+1)*test_negnum]
        train_pos = [v for v in posvid if v not in test_vids]
        train_neg = [v for v in negvid if v not in test_vids]
        random.shuffle(train_pos)
        random.shuffle(train_neg)
        val_vids = train_pos[0:val_posnum] + train_neg[0:val_negnum]
        train_vids = [v for v in vids if v not in test_vids]
        train_vids = [v for v in train_vids if v not in val_vids]
        #print("train_pos:{0},train_neg:{1}, val_vids:{2}, test_vids:{3}".format(len(train_pos), len(train_neg), len(val_vids), len(test_vids)))
        test_k_vids.append(test_vids)
        val_k_vids.append(val_vids)
        train_k_vids.append(train_vids)
        if DEBUG:
            print("train: pos({0}), neg({1})".format(len(train_pos), len(train_neg)))
            test_pos = df[(df[label]==1) & (df[pid].isin(test_vids))][pid].unique().tolist()            
            print("test : pos({0}), neg({1})".format(len(test_pos), len(test_vids)-len(test_pos)))
        
    return train_k_vids, val_k_vids, test_k_vids
    
# Return only training and test set without a validation set
def stratified_kfold_noVal(df, k, pid, DEBUG):
    print("Data splitting for a stratified ",k,"-fold cross validation")
    vids = df[pid].unique().tolist()
    posvid = df[df[label] == 1][pid].unique().tolist()
    negvid = [v for v in vids if v not in posvid]
    random.shuffle(posvid)
    random.shuffle(negvid)
    random.shuffle(vids)
    
    test_posnum = int(len(posvid) / k) #val_posnum = test_posnum
    test_negnum = int(len(negvid) / k) #val_negnum = test_negnum
    
    train_posnum = len(posvid) - test_posnum*2
    train_negnum = len(negvid) - test_negnum*2
    print('posvid:{0}, negvid:{1}, test_posvid:{2}, test_negvid:{3}, train_posvid:{4}, train_negvid:{5}'
          .format(len(posvid), len(negvid), test_posnum, test_negnum, train_posnum, train_negnum))

    test_k_vids = []
    train_k_vids = []
    for i in range(k):
        test_vids = posvid[i*test_posnum:(i+1)*test_posnum] + negvid[i*test_negnum:(i+1)*test_negnum]
        train_pos = [v for v in vids if v not in test_vids]
        train_neg = [v for v in negvid if v not in test_vids]
        train_vids = [v for v in vids if v not in test_vids]
        print("train_pos:{0},train_neg:{1}, test_vids:{2}".format(len(train_pos), len(train_neg), len(test_vids)))
        test_k_vids.append(test_vids)
        train_k_vids.append(train_vids)
        if DEBUG:
            print("train: pos({0}), neg({1})".format(len(train_pos), len(train_neg)))
            test_pos = df[(df[label]==1) & (df[pid].isin(test_vids))][pid].unique().tolist()            
            print("test : pos({0}), neg({1})".format(len(test_pos), len(test_vids)-len(test_pos)))
        
    return train_k_vids, test_k_vids

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
# confusion matrix    
def conf_measures(orgY, predY):
    conf = confusion_matrix(orgY, predY)
    accuracy = accuracy_score(orgY, predY)
    precision = precision_score(orgY, predY)
    recall = recall_score(orgY, predY)
    f1 = f1_score(orgY, predY)
    return conf, accuracy, precision, recall, f1
    
import matplotlib.pyplot as plt

# Given missing indicator mi, get average results of 5 metrics
def getResult(df, mi):
    modes = df.fill_mode.unique().tolist()
    auc = []
    f1 = []
    rec = []
    prec = []
    acc = []
    for m in modes:
        auc.append(np.mean(df[(df.fill_mode == m) & (df.mi_mode == mi)].auc))
        acc.append(np.mean(df[(df.fill_mode == m) & (df.mi_mode == mi)].acc))
        f1.append(np.mean(df[(df.fill_mode == m) & (df.mi_mode == mi)].f1))
        prec.append(np.mean(df[(df.fill_mode == m) & (df.mi_mode == mi)].prec))
        rec.append(np.mean(df[(df.fill_mode == m) & (df.mi_mode == mi)].recall))
    return acc, rec, prec, f1, auc

# prepare the analysis table by methods
def makeAnaldf(acc, rec, prec, f1, auc):
    analdf = pd.DataFrame(columns = ['mode','acc', 'prec', 'recall', 'f1', 'auc'])
    analdf.loc[len(analdf)] = ['TBM.25', acc[1], prec[1], np.round(rec[1],4), np.round(f1[1],4),np.round(auc[1],4)]
    analdf.loc[len(analdf)] = ['TBM.50', acc[0], prec[0], np.round(rec[0],4), np.round(f1[0],4),np.round(auc[0],4)]
    analdf.loc[len(analdf)] = ['TBM.75', acc[2], prec[2], np.round(rec[2],4), np.round(f1[2],4), np.round(auc[2],4)]
    analdf.loc[len(analdf)] = ['forward', acc[3], prec[3], np.round(rec[3],4), np.round(f1[3],4), np.round(auc[3],4)]
    analdf.loc[len(analdf)] = ['mean', acc[4], prec[4], np.round(rec[4], 4), np.round(f1[4],4), np.round(auc[4],4)]
    return analdf

# draw a bar graph for the results
from matplotlib.font_manager import FontProperties
def drawAnaldf(analdf, mi):
    font = {'family' : 'DejaVu Sans','weight' : 'normal','size': 12}
    plt.rc('font', **font)
    metrics = ['acc',  'prec', 'recall','f1', 'auc']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mikey = ''
    if mi:
        mikey = '_MI'
    
    analdf[metrics].loc[0].plot(kind='bar', color='#0099ff', ax=ax, position=1, figsize=(6,4),width=0.1, legend=True, label='TBM.25'+mikey)
    analdf[metrics].loc[1].plot(kind='bar', color='#33ccff', ax=ax, position=2, figsize=(6,4),width=0.1, legend=True, label='TBM.50'+mikey)
    analdf[metrics].loc[2].plot(kind='bar', color='blue', ax=ax, position=3, figsize=(6,4),width=0.1, legend=True, label='TBM.75'+mikey)
    analdf[metrics].loc[3].plot(kind='bar', color='#ff6699', ax=ax, position=4, figsize=(6,4),width=0.1, legend=True, label='Forward'+mikey)
    analdf[metrics].loc[4].plot(kind='bar', color='#00e600', ax=ax, position=5, figsize=(6,4),width=0.1, legend=True, label='Mean'+mikey)
    plt.ylabel = ('AUC')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))#, mode='expand'
    plt.ylim(.4, 1.0)
    #plt.title(title)
    plt.show()
    
def loadResult(file, feat):
    lstm1_auc = analyze(file, 1, 'auc', feat)
    lstm0_auc = analyze(file, 0, 'auc', feat)
    lstm1_acc = analyze(file, 1, 'acc', feat)
    lstm0_acc = analyze(file, 0, 'acc', feat)
    lstm1_f1 = analyze(file, 1, 'f1', feat)
    lstm0_f1 = analyze(file, 0, 'f1', feat)
    lstm1_pre = analyze(file, 1, 'prec', feat)
    lstm0_pre = analyze(file, 0, 'prec', feat)
    lstm1_rec = analyze(file, 1, 'recall', feat)
    lstm0_rec = analyze(file, 0, 'recall', feat)
    return lstm1_auc, lstm0_auc, lstm1_acc, lstm0_acc, lstm1_f1, lstm0_f1, lstm1_pre, lstm0_pre, lstm1_rec, lstm0_rec


# Given missing indicator mi, get average results of 5 metrics, and draw a graph
def anal(df, mi):
    if mi:
        print("* with missing indicators")
    else:
        print("* without missing indicators")
    acc, rec, prec, f1, auc = getResult(df, mi)
    analdf = makeAnaldf(acc, rec, prec, f1, auc)
    print(analdf)
    
    havedisplay = "DISPLAY" in os.environ
    if havedisplay:
        drawAnaldf(analdf, mi)
    return analdf    
