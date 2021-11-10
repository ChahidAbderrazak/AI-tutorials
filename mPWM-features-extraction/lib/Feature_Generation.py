#%% -*- coding: utf-8 -*-

#% Packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lib.Shared_Functions import *

#%% #from __future__ import division
import matlab.engine
import io
out = io.StringIO()
err = io.StringIO()

#%% ##################################################################################################
def Set_levels_Sigma_py(k,M,mu,sigma):

    Levels= np.asarray([i for i in range(1,M+1)])
    N=(M-1)//2;
#    VECTOR=[-floor(N/2): floor(N/2)];
    VECTOR=[i for i in range(-N,N + 1)]
#    Level_intervals= mu+k*sigma*VECTOR;
    Level_intervals=[ mu+k*sigma*i for i in VECTOR]

    # method 2
    Level_intervals=np.linspace(mu-3*sigma, mu+3*sigma, num=M-1)

    return Levels , Level_intervals

def mapping_levels(X,Level_intervals, Levels):

    Q=np.zeros(shape = (X.shape[0],X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xn=X[i,j]
            Q[i,j]=Get_level(Xn,Level_intervals,Levels);

    return Q


def Get_level(Vx,Level_intervals,Levels):
#    idx=find(Vx<=Level_intervals);
    idx=[ i for i in range(len(Level_intervals)) if Vx <=Level_intervals[i] ]
#    print('idx',idx)
    if len(idx)==0:
        L=Levels[-1:];
    else:
       l=idx[0];
       L=Levels[l];

    return np.asscalar(np.array(L))



def Load_split_dataset(filepath, mat_filename,list_Subjets,list_Gestures,Trial_split):
    # load mat file where the data are saved
    #filepath, mat_filename = browse_file()
    filepath=filepath+mat_filename

    # Use matlab scripts
    eng = matlab.engine.start_matlab()
    eng.eval("addpath('./Matlab_lib');")
    eng.eval("addpath('./Matlab_lib/mPWM');")

    [Gesture, Subject, Trial, X, DB]=eng.Load_sEMG_data(filepath,nargout=5,stdout=out,stderr=err)

    eng.quit()

    X=np.asarray(X);
    Gesture=np.asarray(Gesture).ravel();Subject=np.asarray(Subject).ravel(); Trial=np.asarray(Trial).ravel()

    # Displays
    blcd, classes, data_dic=Explore_dataset(Gesture)
    # explore the dataset:  number f subjects, trails , gestures...
    trials_labels=np.unique(Trial); print(trials_labels)

    #list_Subjets=np.unique(Subject); print(list_Subjets)
    list_Gestures=np.unique(Gesture); print(list_Gestures)

    #%% Split the data into training and testing
    score=list();

    Trial_TR=list();Trial_TS=list();
    for gesture in list_Gestures:
        Trial_TR.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[0]]))
        Trial_TS.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[1]]))



    # Genrate the Training/Testing splits
    y=Gesture
    X_train= X[Trial_TR];  X_train_mat=matlab.double(X_train.tolist()); print(X_train)
    y_train= y[Trial_TR];  y_train_mat=matlab.double(y_train.tolist())
    X_test = X[Trial_TS];  X_test_mat=matlab.double(X_test.tolist())
    y_test = y[Trial_TS]

    # Displays
    print('\n\n==> Genrate the Training/Testing splits ')
    trials_labels_TR=np.unique(Trial[Trial_TR]); print('Training size', len(y_train) ,' ,  trials labels: ',trials_labels_TR)
    trials_labels_TS=np.unique(Trial[Trial_TS]); print('Testing size', len(y_test) ,' ,  trials labels: ',trials_labels_TS)

    y_test=y_test-1
    y_train=y_train-1

    return X, Gesture,Subject, Trial, X_train, y_train, X_test, y_test, mat_filename

def Quantization( k,M, X, mu="NA", sigma="NA", y="NA"):

    if  isinstance(mu,str):
        # Use matlab scripts
        eng = matlab.engine.start_matlab()
        eng.eval("addpath('.\Matlab_lib');")
        eng.eval("addpath('.\Matlab_lib\mPWM');")

        #  Quatization
        mu,sigma=eng.Split_Multi_classes_samples(matlab.double(X.tolist()),matlab.double(y.tolist()),nargout=2,stdout=out,stderr=err);
        eng.quit()

    Levels, Level_intervals=Set_levels_Sigma_py(k,M,mu,sigma);
    Q= mapping_levels(X,Level_intervals, Levels);

    return Q,mu,sigma,Level_intervals

def Build_QuPWM_matrices(m, X, y,Level_intervals):
    # Use matlab scripts
    eng = matlab.engine.start_matlab()
    eng.eval("addpath('.\Matlab_lib');")
    eng.eval("addpath('.\Matlab_lib\mPWM');")

    print('\n-->  Generate the mPWM matrices')
    X_mat=matlab.double(np.asarray(X).tolist());
    y_mat=matlab.double(np.asarray(y).tolist());
    Level_intervals_mat=matlab.double(np.asarray(Level_intervals).tolist());

    mPWM_structure=eng.Build_mPWMs_Structure(m, X_mat, y_mat, Level_intervals_mat,nargout=1,stdout=out,stderr=err);
    print(out.getvalue()); print(err.getvalue());
    eng.quit()

    return mPWM_structure

def Generate_QuPWM_features(mPWM_structure, Q):
    # Use matlab scripts
    eng = matlab.engine.start_matlab()
    eng.eval("addpath('.\Matlab_lib');")
    eng.eval("addpath('.\Matlab_lib\mPWM');")

    print('\n-->  Generate the QuPWM features of order ',mPWM_structure['m'] )
    Q_mat=matlab.double(np.asarray(Q).tolist());

    mPWM_feature=eng.Generate_mPWM_features(mPWM_structure, Q_mat,nargout=1,stdout=out,stderr=err);
    print(out.getvalue()); print(err.getvalue());
    eng.quit()

    return mPWM_feature


def QuPWM_feature_selection(mPWM_feature,selected_feature):
    name_features=list(mPWM_feature['C1'].keys())
    name_classes =list(mPWM_feature.keys())

    QuPWM_names=[]
    if selected_feature==-1:
        for k in name_features:
            QuPWM_names.append(k)
        selected_feature=[i for i in range(1,len(QuPWM_names)+1)]
    else:
        for k in selected_feature:
            QuPWM_names.append(name_features[k-1])


    QuPWM_sizes=[];QuPWM_sizes.append(0)

    for f in QuPWM_names:
        for C in name_classes:
             f_new=mPWM_feature[C][f]
             try:
                 QuPWM_f=np.concatenate((QuPWM_f,f_new), axis=1)
             except NameError:
                 QuPWM_f=np.asarray(f_new)

        QuPWM_sizes.append(QuPWM_f.shape[1])

    print('\n name=',QuPWM_names,'\n size=',QuPWM_sizes)

    return QuPWM_f, QuPWM_names, QuPWM_sizes,selected_feature


def Feature_selection_using_Tree(X,y):
    from sklearn.ensemble import ExtraTreesClassifier


    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    #% feature selection using threshold
    return  indices,importances

def Get_QuPWM_feature(QuPWM_f, Get_feature, selected_feature,QuPWM_sizes):
    indices=[]
    for k in Get_feature:
        start=selected_feature.index(k)

        for j in range(QuPWM_sizes[start],QuPWM_sizes[start+1]):
            indices.append(j)
    X=QuPWM_f[:,indices]

    return X

def Feature_selection_using_Scanning_kMers(mPWM_feature_train,y_train, mPWM_feature_test, y_test, names, classifiers):

    from itertools import combinations

    #Genrate all features
    selected_feature=[3,6,9]
    QuPWM_f_train, QuPWM_names, QuPWM_sizes, selected_feature=QuPWM_feature_selection(mPWM_feature_train,-1)
    QuPWM_f_test , QuPWM_names, QuPWM_sizes0, selected_feature0=QuPWM_feature_selection(mPWM_feature_test,selected_feature)


    acc_max=0
    for k in range(3,6):
        for Get_feature in combinations(selected_feature , k) :
            print(list(Get_feature))
    #        selected_feature=[1,5,8]# [0,1,2]#
            Xf_train=Get_QuPWM_feature(QuPWM_f_train, Get_feature, selected_feature,QuPWM_sizes);
            Xf_test =Get_QuPWM_feature(QuPWM_f_test , Get_feature, selected_feature,QuPWM_sizes);

            Mdl_score_op=Classification_Train_Test(names, classifiers, Xf_train, y_train, Xf_test, y_test )

            acc=Mdl_score_op['Logistic Regression'][0]
            print('Acc_max=',acc_max,'---  Acc=',acc)

            if acc>acc_max :
                selected_feature_op=selected_feature
                Xf_train_op=Xf_train
                Xf_test_op=Xf_test
                acc_max=acc

    return  selected_feature_op,Xf_train_op,Xf_test_op,acc_max