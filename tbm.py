"""
 tbm.py (main)
 
 excution:
 
 python tbm.py -n [RNN_model] -hu [number of hidden units] -b [batch size] 
               -m [max sequence length] -g [GPU-ID] -e [early prediction hours] 
               -k [keyward for results] 
 
 RNN_model: RNN | LSTM | GRU
 GPU-ID: integer (0 | 1 | ...)
 early prediction hours: integer  
 
 input files: preprocessed data file
 output files: 
  - validataion log
  - test results with 5 metrics from 5-fold crossvalidation   
    (metrics: accuracy, precision, recall, f1 score, AUC)

 Author: Yeo Jin Kim
 updated: Nov. 13, 2018
 
"""

import os 
import math
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
import tensorflow as tf
import TLSTM
import tbmlib as tl

def set_rnn(nn_mode, X_train, Y_train, activation, loss,optimizer, metrics, batch, epoch, max_seq_length, hidden_units):    
    input_length = X_train.shape[1]
    input_dim = X_train.shape[2]
    input_shape = (input_length, input_dim) 
    batch_shape = (1, max_seq_length, input_dim)
    model = Sequential()
    
    if nn_mode == 'LSTM':        
        model.add(LSTM(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))              
    elif nn_mode == 'LSTM2':
        model.add(LSTM(hidden_units, return_sequences=True, input_shape = input_shape, batch_input_shape = batch_shape))
        model.add(LSTM(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))
    elif nn_mode == 'GRU':
        model.add(GRU(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))
    elif nn_mode == 'GRU2':
        model.add(GRU(hidden_units, return_sequences=True, input_shape = input_shape, batch_input_shape = batch_shape))
        model.add(GRU(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))        
    elif nn_mode == 'RNN':
        model.add(SimpleRNN(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))
    elif nn_mode == 'RNN2':
        model.add(SimpleRNN(hidden_units, return_sequences=True, input_shape = input_shape, batch_input_shape = batch_shape))
        model.add(SimpleRNN(hidden_units, input_shape = input_shape, batch_input_shape = batch_shape))

    if dropout:
        model.add(Dropout(do_rate, input_shape=(hidden_units,)))
    model.add(Dense(int(hidden_units/2), activation='relu'))  
    model.add(Dense(int(hidden_units/2), activation='relu'))
    model.add(Dense(1, activation=activation))
    model.compile(loss= loss, optimizer=optimizer, metrics = metrics)
    print ("{0}: input_dim = {1}, hidden_dim = {2}".format(nn_mode, np.shape(X_train), hidden_units))
    return model


def learn_batch(model, traindf, max_review_length):   
    trainX, trainY = tl.makeXY(traindf, totfeat, pid) # make 2D array data : take the post sequences with max_review_length
    trainX_pad = pad_sequences(trainX, maxlen = max_review_length, dtype='float')         # padding
    history = model.fit(trainX_pad, trainY, callbacks=[ResetStatesCallback()], batch_size=batch, shuffle=False, 
                        epochs=1, verbose = 0)
    return model 


def make_batches(df, batch_size):
    vids = df[pid].unique()
    batches_vid = []
    for i in range(int(len(vids)/ batch_size)):
        batches_vid.append(vids[batch_size*i: batch_size*(i+1)])
    
    return batches_vid

# Preprocessing for batch learning
def prepro_batches(df, pid, taus, batches_vid, totfeat, max_seq_length, bf_mode, fill_mode):
    batch_df = df[df[pid].isin(batches_vid)][:]
    if DEBUG:
        print("--- prepro_baches")
        print("pos: {0}, tot: {1}".format(len(batch_df[batch_df[label]==1][pid].unique()), len(batches_vid))) 
    
    if bf_mode: # belief mode
        batch_df = tl.make_belief(batch_df, pid, feat, taus, fill_mode)

    X_batch, Y_batch = tl.makeXY(batch_df, totfeat, pid) # make 2D array data : take the post sequences with max_review_length
    X_batch_pad = pad_sequences(X_batch, maxlen = max_seq_length, dtype='float')         # padding

    return X_batch_pad, Y_batch

# Batch training
def train_batch(pid, nn_mode, traindf, valdf, batch_size, alpha, MAX_EPOCH, taus, max_seq_length, totfeat,tr_posnum,te_posnum,
                impute_mode, bf_mode, mi_mode, fill_mode,hidden_units):
    print('Training:')
    batches_vid = make_batches(traindf, batch_size)   # split the batches of training data with batch_size     
    
    if mi_mode == True: # With missing indicators, get the indexes for features (except missing indicators) 
	    idx_feat = [x for x in range(len(totfeat)) if x%2 == 0]
    else: 				# Without missing indicators, use only original features
        totfeat = feat
        idx_feat = [x for x in range(len(totfeat))]
    
    if nn_mode == 'LSTM' or nn_mode == 'LSTM2':  # the input gate weight of LSTM is the first weight among "4" weights 
        w_dim = 4          # LSTM - the gate weight order: input, forget, candidate, output)
    elif nn_mode == 'GRU' or nn_mode == 'GRU2':  # GRU - 3 gates (2 gates, 1 candidate)
        w_dim = 3          
    elif nn_mode == 'RNN' or nn_mode == 'RNN2': # RNN
        w_dim = 1
    else:
        print("Unknown RNN model: Cannot get the input weight indexes!")
        return 

    # get the indexes of the input gate weights, if model is LSTM or GRU
    idx_hl = [x for x in range(hidden_units * w_dim) if x % w_dim == 0] 
    
    # Get the first batch information for model setting
    trainX_batch_pad, trainY_batch = prepro_batches(traindf, pid, taus, batches_vid[0], totfeat, max_seq_length, 
                                                    bf_mode = bf_mode, fill_mode=fill_mode)
    
    # Set the model
    model = set_rnn(nn_mode, trainX_batch_pad,trainY_batch,'sigmoid','binary_crossentropy','adam', 
                     ['binary_accuracy'],1,1,max_seq_length, hidden_units)  
    
    models = []
    aucs = []
    taus_pool = []
    for epoch in range(MAX_EPOCH):
        mean_tr_acc = []
        mean_tr_loss = []
        pre_weights = []
        cur_weights = []

        for i in range(len(batches_vid)):    
            # 1. Preprocess the batch
            if i>0: # if i==0, use the first batch generated above
                trainX_batch_pad, trainY_batch = prepro_batches(traindf, pid, taus, batches_vid[i], totfeat, 
                                                                max_seq_length, bf_mode = bf_mode, fill_mode=fill_mode)
            # 2. Learn RNN with the batch
            for j in range(batch_size):
                tr_loss, tr_acc = model.train_on_batch(np.expand_dims(trainX_batch_pad[j], axis=0), 
                                                       np.array([trainY_batch[j]]))
                mean_tr_acc.append(tr_acc)
                mean_tr_loss.append(tr_loss)

            # ---------------------------------------------------------------------------
			# TBM Learning, only for TBM mode
            if bf_mode: 
                pre_weights = cur_weights
                cur_weights = model.layers[0].get_weights()[0][idx_feat] # take W from the first layer 
                cur_weights = np.array([w[idx_hl] for w in cur_weights]) # take the (input gate) weights from W
                
                if i > 0: # update taus
                    #cost = np.mean((cur_weights - pre_weights)**2, 1) # grads = np.mean((cur_weights - pre_weights), 1)
                    grads = np.mean((cur_weights - pre_weights), 1) # average gradient of batch input weights
                    taus += grads * alpha # alpha = adjustment for learning step size of tau 
                    taus = [ t if  t>=0 else 0 for t in taus] #tau's lower bound = 0 (no negative reliable time window)
            # ---------------------------------------------------------------------------
            model.reset_states()

        #print(' ep {0}- acc: {1:.4f}, loss: {2:.4f} (a={3:.2f})'
        #      .format(epoch, np.mean(mean_tr_acc), np.mean(mean_tr_loss), alpha))
        
        # Validation
        conf, acc, prec, rec, f1, auc = test_batch(valdf, taus, batch_size, model, bf_mode,mi_mode, totfeat, fill_mode)
        models.append(model)
        aucs.append(auc)
        taus_pool.append(taus)
        
        # Save the validation results
        if impute_mode == 'rule':
            resultdf_val.loc[len(resultdf_val)]= [nn_mode, max_seq_length, hidden_units, batch_size, alpha, MAX_EPOCH, eh, tr_posnum, te_posnum,
                                                  impute_mode, fill_mode, mi_mode,conf, acc, prec, rec, f1,auc]+np.zeros(len(feat)).tolist()
        else:
            resultdf_val.loc[len(resultdf_val)]= [nn_mode, max_seq_length, hidden_units, batch_size, alpha, MAX_EPOCH, eh, tr_posnum, te_posnum,
                                                  impute_mode, fill_mode, mi_mode, conf, acc, prec, rec, f1,auc]+taus

        alpha = alpha * alpha_dec   # decrease alpha over time for convergence
        
        # Early stopping
        best_idx, best_val = tl.get_best_model_idx(aucs)
        print(' e {0}) tr - acc: {1:.4f}, loss: {2:.4f} / val - auc: {3:.4f} (best e:{4}, {5:.4f})'  
              .format(epoch, np.mean(mean_tr_acc), np.mean(mean_tr_loss), auc, best_idx, best_val))  #a={x:.2f} alpha
        if epoch - best_idx  >= patience and epoch >= MIN_EPOCH:
            print("Early stopped (patience {0})".format(patience))
            break

    resultdf_val.to_csv(valres_file+".csv", index = False) # save the validation results file

    return models[epoch], taus_pool[epoch], epoch 


# Test early prediction using the learned model
def test_batch(testdf, taus, batch_size, model, bf_mode, mi_mode, totfeat, fill_mode):
    if mi_mode == False:
        totfeat = feat

    batches_vid = make_batches(testdf, batch_size)
     
    if DEBUG:
        print("test_batch")
    testX, testY = tl.makeXY(testdf,totfeat,pid)

    predY = []
    trueY = []       
    pred = []
    for i in range(len(batches_vid)):    
        # 2. Preprocess the batch
        testX_batch_pad, testY_batch = prepro_batches(testdf, pid, taus, batches_vid[i], totfeat, max_seq_length, bf_mode, fill_mode)

        for j in range(batch_size):
            te_loss, te_acc = model.test_on_batch(np.expand_dims(testX_batch_pad[j], axis=0),
                                                  np.array([testY_batch[j]]))

            pred_val = model.predict_on_batch(np.expand_dims(testX_batch_pad[j], axis=0))[0][0]
            predY.append(int(round(pred_val)))
            pred.append(pred_val)
        model.reset_states()
        trueY += testY_batch

    if DEBUG:
        print("trueY: ", trueY)
    conf, acc, prec, rec, f1 = tl.conf_measures(trueY,predY)    
    auc = roc_auc_score(trueY, pred, average='micro', sample_weight=None) # micro : considering imbalanced / macro : not considering imbalanced
    #print('acc = {0:.4f}, loss = {1:.4f}'.format(np.mean(mean_te_acc), np.mean(mean_te_loss)))
    #print('{0}, acc:{1:.4f}, prec:{2:.4f}, rec:{3:.4f}, f1:{4:.4f}, auc:{5:.4f}'.format(conf, acc,prec, rec,f1,auc))
    #print('   auc:{0:.4f}'.format(auc))
    return conf, acc, prec, rec, f1, auc


# TBM learning
def belief_learning(pid, e_traindf, e_valdf, e_testdf, eh, hidden_units):
    print('----------------------------------')
    te_posnum = len(e_testdf[e_testdf[label] == 1][pid].unique())
    tr_posnum = len(e_traindf[e_traindf[label] == 1][pid].unique())
    impute_mode = 'belief'
    alpha = alpha_init
    for fm in fill_mode:
        for mi in mi_mode:
            taus = taus_org[:]
            print('\n * [{0}-hbs] belief: {1}-fill, mi = {2}'.format(eh, fm, mi))
            model, taus, model_idx = train_batch(pid, nn_mode, e_traindf, e_valdf, batch_size, alpha, MAX_EPOCH, taus, max_seq_length, 
                                                 totfeat, tr_posnum, te_posnum, impute_mode, bf_mode = True, 
                                                 mi_mode = mi, fill_mode=fm, hidden_units=hidden_units)
            
            conf, acc, prec, rec, f1, auc = test_batch(e_testdf, taus, batch_size, model, bf_mode = True, mi_mode = mi,
                                                       totfeat=totfeat, fill_mode=fm)
            print("  TEST - auc: {0:.4f}".format(auc))
            resultdf.loc[len(resultdf)]= [nn_mode, max_seq_length, hidden_units, batch_size, alpha, model_idx, eh, tr_posnum, te_posnum,impute_mode, fm, mi, 
                                          conf, acc, prec, rec, f1, auc] + taus
            resultdf.to_csv(testres_file+".csv", index = False) 
    print('----------------------------------')

# Baseline learning (rule-based)
def rule_learning(pid, e_traindf, e_valdf, e_testdf, eh, hidden_units):
    impute_mode = 'rule'
    alpha = ''
    taus = ''
    for fm in fill_mode_base:   
        if fm == 'e':	# Expert mode
            e_traindf, e_valdf, e_testdf = tl.clinic_ffill(e_traindf, e_valdf, e_testdf, feat, vitals, labs, pid)
        elif fm == 'f': # Forward-filling
            e_traindf, e_valdf, e_testdf = tl.ffill(e_traindf, e_valdf, e_testdf, pid, feat) # elif fm == 'b': e_traindf, e_testdf = bfill(e_traindf, e_testdf)
        elif fm =='0':  # Zero-filling
            e_traindf, e_valdf, e_testdf = tl.zero_fill(e_traindf, e_valdf, e_testdf, feat)

        te_posnum = len(e_testdf[e_testdf[label] ==1][pid].unique()) #e_valdf = e_testdf[e_testdf.VisitIdentifier.isin(idvaldf['0'])]
        tr_posnum = len(e_traindf[e_traindf[label] ==1][pid].unique())
        
        for mi in mi_mode:
            print('\n * [{0}-hbs] base: {1}-fill, mi = {2}'.format(eh, fm, mi))
            model, taus, model_idx = train_batch(pid, nn_mode, e_traindf, e_valdf, batch_size, 0, MAX_EPOCH, taus, max_seq_length, 
                                                 totfeat, tr_posnum, te_posnum, impute_mode, bf_mode = False, 
                                                 mi_mode = mi, fill_mode='none',hidden_units=hidden_units)
            conf, acc, prec, rec, f1, auc = test_batch(e_testdf, taus, batch_size, model, bf_mode = False, mi_mode = mi,
                                                  totfeat=totfeat, fill_mode='none')
            print("  TEST - auc: {0:.4f}".format(auc))
            resultdf.loc[len(resultdf)]= [nn_mode, max_seq_length, hidden_units, batch_size, alpha, model_idx, eh, tr_posnum, te_posnum, impute_mode, 
                                          fm, mi,conf, acc, prec, rec, f1,auc] + np.zeros(len(feat)).tolist()
            resultdf.to_csv(testres_file+".csv", index = False)   


# Initialization
# 1. make the directory for results
# 2. set the initial reliable time windows (timedf)
def initialize(nn_mode, args, pid, time_feat, label, file, feat):
    print("Initalize....")
    newdir = 'data/'+args.k+'_'+nn_mode
    if os.path.exists(newdir)==False: 
    	os.mkdir(newdir)
    	os.mkdir(newdir+'/test/')
    	os.mkdir(newdir+'/val/')
    	os.mkdir(newdir+'/models/')
    modelfile = newdir+'/models/model'
    print("creat output folder:", newdir)

    df = tl.loaddf(file)
    totfeat = []
    for f in feat:
        totfeat.append(f)
        totfeat.append(f+'_mi')
        
	# set dataframes for test & validation results
    columns = ['model', 'max_seq_length','hidden', 'batch_size', 'alpha', 'MAX_EPOCH', 'hbs', 'tr_posnum', 'te_posnum', 'bf_mode', 'fill_mode', 'mi_mode', 'confmat','acc', 'prec', 'recall', 'f1','auc'] + feat
    resultdf = pd.DataFrame(columns = columns)
    resultdf_val = pd.DataFrame(columns = columns)
        
    key_hyper = '_hbs'+str(earlyHours[0])+'_f'+str(len(feat))+'_k'+str(kfold)+'h'+str(hidden_units[0])+'m'+str(max_seq_lengths[0])+'b'+str(batch_size)
    if dropout == True:
        key_hyper += 'd'+str(do_rate)

    testres_file = newdir+'/test/'+nn_mode+key_hyper+'_test' # log file for test
    valres_file = newdir+'/val/'+nn_mode+key_hyper+'_val'    # log file for validation

    if INIT_TIMEDF: # initialize the reliable time windows with the average sampling frequency of features
        taus_org, timedf = tl.init_tau(df, pid, time_feat, feat, 1000, False)
        timedf.to_csv(newdir+"/timedf"+str(earlyHours[0])+"h.csv")
    else:
        timedf = tl.loaddf_index("data/timedf.csv" ) 
        taus_org = timedf.loc['tau_init'].values

    return resultdf, resultdf_val, testres_file, valres_file, key_hyper, modelfile, totfeat, key_hyper, taus_org, timedf



import argparse
if __name__ == '__main__':
    DEBUG = False
    # extract filename from command
    parser = argparse.ArgumentParser()
    parser.add_argument("-n")   # Network model
    parser.add_argument("-hu")  # num of hidden units
    parser.add_argument("-e")   # target early hours before septic shock
    parser.add_argument("-m")   # max_sequence_length for RNN learning
    parser.add_argument("-b")   # batch size
    parser.add_argument("-g")   # GPU ID#
    parser.add_argument("-k")   # keyword for models & results
    args = parser.parse_args()
    earlyHours = [args.e]  	    # e.g. 8 = 8 hour early prediction
    max_seq_lengths = [int(args.m)] # max sequence length for RNN 

    # GPU setup
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'            # Ignore detailed log massages for GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    # the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)  # GPU-ID "0" or "0, 1" for multiple
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    session = tf.Session(config=config)    
    
    # Define primary ID, target label, temporal feature
    pid = 'VisitIdentifier'				# primary ID
    label = 'ShockFlag' 				# target label
    time_feat = 'MinutesFromArrival' 	# temporal feature
    kfold = 5							# N-fold crossvalidation
    
    # Missing data handling mode setting
    fill_mode = ['0','.25','.5','.75','1'] 	# TBM modes: ['0','.25','.5','.75','1'] 
                        	# .0 : 100% back propagation / 0.25 : 25% back + 75% forward belief propagation / ...
    fill_mode_base = ['f','e','0'] 	# baseline modes: ['f','e','0'] f: forward-filling, e: expert, 0: zero-filling
    mi_mode = [True, False] # Missing Indicator mode
    INIT_TIMEDF = True      # True to general a new 'timedf' file for an initial reliable time, which is set with the average sampling frequency of each feature 
                            # False to use the existing 'timedf'
    # RNN model setting
    nn_mode = args.n		# Neural network models: 'LSTM', 'RNN', 'GRU'
    hidden_units = [int(args.hu)] # num of hidden units
    dropout, do_rate = True, 0.2
    
    # Learning hyper-parameters
    batch_size = int(args.b) 
    MAX_EPOCH = 2 
    MIN_EPOCH = 0
    patience = 7
    random.seed(1000) # for reproducing the results
    
    # TBM hyper-parameters
    alpha_init, alpha_dec = 1000, 0.98	# alpha * error = step size for tau learning

    # Features 
    feat = ['SystolicBP', 'MAP', 'Lactate', 'WBC', 'Platelet', 'Creatinine',
       'RespiratoryRate', 'FIO2', 'PulseOx', 'BiliRubin', 'BUN', 'HeartRate',
       'Temperature', 'Bands', 'Gender', 'Age'] # gender and age are static features (no missing values)
    vitals = ['SystolicBP', 'MAP','RespiratoryRate','PulseOx','HeartRate','Temperature','FIO2']
    labs = ['Lactate', 'WBC', 'Platelet', 'Creatinine','BiliRubin', 'BUN','Bands']
    numfeat = ['SystolicBP', 'MAP', 'Lactate', 'WBC', 'Platelet', 'Creatinine',
       'RespiratoryRate', 'FIO2', 'PulseOx', 'BiliRubin', 'BUN', 'HeartRate',
       'Temperature', 'Bands'] # static features should not be included here
       
    # For early prediction, 
    # i. if you use different data sets for training and test, load them separately. 
    # ii. Otherwise, you can use a same file for training and test.
    #     It will be automatically split for cross-validation during training 
    train_file = 'data/right_align/subgroups/gender_race_age/Trunc_8h.csv' # data before 1 hour 
    test_file = 'data/right_align/subgroups/gender_race_age/Trunc_'+str(earlyHours[0])+'h.csv' # data before the target hours (e.g. 4 hours)
    resultdf, resultdf_val, testres_file, valres_file, key_hyper, modelfile, totfeat, key_hyper, taus_org, timedf = initialize(nn_mode, args, pid, time_feat, label, train_file, feat)
    
	
    for eh in earlyHours:
        # Training data: sorting and setting missing indicators
        edf_org = tl.loaddf(train_file)
        edf_org = edf_org.sort_values([pid, time_feat])
        edf_org = tl.setmi(edf_org, feat) 
        # Test data: sorting and setting missing indicators 
        test_org = tl.loaddf(test_file) 
        test_org = edf_org.sort_values([pid, time_feat])
        test_org = tl.setmi(test_org, feat) 
        
        train_k_vids, val_k_vids, test_k_vids = tl.stratified_kfold_traing70(edf_org, kfold, pid, DEBUG) # training 70%, val: 10%, test: 20%

        for k in range(kfold):
            # print(len(train_k_vids[i]), len(val_k_vids[i]), len(test_k_vids[i]))
            traindf = edf_org[edf_org[pid].isin(train_k_vids[k])] 
            valdf = edf_org[edf_org[pid].isin(val_k_vids[k])] 
            testdf = test_org[edf_org[pid].isin(test_k_vids[k])]
            print("\n** fold {0} - training: {1}, validation: {2}, test: {3}".format(k, np.shape(traindf), np.shape(valdf), np.shape(testdf)))
            
            if False: # if the data has been not yet standardized, set True
                traindf, valdf, testdf = tl.standardize_all(traindf.copy(deep=True), valdf.copy(deep=True), 
                                                      testdf.copy(deep=True), numfeat, False) 
            for max_seq_length in max_seq_lengths:
                for hu in hidden_units:
                    print('# max_seq_length:{0}, hidden:{1}, batch:{2}, epoch:{3}, alpha:{4} ({5} / {6}) [k={7}]'.format(max_seq_length, hu, batch_size, MAX_EPOCH, alpha_init, nn_mode, key_hyper, k))
                    belief_learning(pid, traindf.copy(deep=True),valdf.copy(deep=True),testdf.copy(deep=True),eh,hu)
                    rule_learning(pid, traindf.copy(deep=True),valdf.copy(deep=True),testdf.copy(deep=True),eh,hu)



