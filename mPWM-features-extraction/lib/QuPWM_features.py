#%% -*- coding: utf-8 -*-
#% Packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lib.Shared_Functions import *
from sklearn.preprocessing import normalize


class mPWM:
  
  def __init__(self, m, Levels):
    self.m = m
    self.Levels=Levels
    self.mPWM={}
    self.motifs={}
    self.classes=[]
    
  def Me(self):
    print('Hello, it mPWM features done @ KAUST 2019 ')

  def Build_QuPWM_matrices(self, Q_sequences, y):
    import itertools
    print('\n-->  Generate the mPWM matrices')
    classes=np.unique(y)
    self.classes=classes
    # Get all motifs of order m
    for m in range(1,self.m +1):            
            kmers=list(self.Levels)
            perm_k = [p for p in itertools.product(kmers, repeat=m)]#  permutations(kmers,m)#
            
            # Print the obtained permutations 
            motifs=[]
            for i in list(perm_k): 
                    
                motifs.append(i)
                #print(motifs)
            name_motifs='m'+str(m)
            self.motifs[name_motifs] = motifs
            
            
    # Get the PWM for each morif for each class        
    for C in classes:
        Idx=np.where(y==C)
        Q_input_Class=Q_sequences[Idx];
    
        ## Build the different kmers
        # print('\n   -->  Building mPWM matrices for the class ',C)     
    
        for m in range(1,self.m +1):
            # print('\n    - Extracting kmers', m)
            
            name_motifs='m'+str(m)
            motifs=self.motifs[name_motifs]
            PWM_C=np.zeros([len(motifs), np.size(Q_input_Class,1)])
    
            for i in range(0,len(motifs)):                          # Scan all kmers motifs
                pattern = motifs[i]                                 # given this m-mer
                for p in range(0,np.size(Q_input_Class,1)-m+1):                              #calculate k-mer frequency of order m at all posiitons p 
                    count_mers=Q_input_Class[:,p:p+m] == pattern
                    # print(' ------\n',i,p,count_mers)
                    PWM_C[i,p]=np.sum(count_mers.all(axis=1))
                    
                # print('PWM of C=',C,' using m=',m,' is:',PWM_C)
                PWM_name='C'+str(C)+'-m'+str(m)
                
                # normalize PWM rows to sum up to 1 ( kmers probabilities)
                # PWM_C_norm=PWM_C.T
                PWM_C_norm=normalize(PWM_C.T, axis=1, norm='l1')
                self.mPWM[PWM_name]=PWM_C_norm
            
        
def get_data_stats(X, y):
    
    from scipy.stats import norm
    import seaborn as sns


    classes=np.unique(y)
    mu=[];sigma=[]
    for C in classes:
            Idx=np.where(y==C)
            data=X[Idx].flatten();
            mu.append(np.mean(data))
            sigma.append(np.std(data))
            # sns.distplot(data, hist=False, rug=True);
            
    # print('mu=',mu); print('sigma=',sigma); 
    
    mu=np.mean(mu)
    sigma=np.min(sigma)
        
    # print('mu=',mu); print('sigma=',sigma)

    return mu, sigma
            
def Set_levels_Sigma_py(k,M,mu,sigma):

    Levels= np.asarray([i for i in range(0,M)])
    N=(M-1)//2;
#    VECTOR=[-floor(N/2): floor(N/2)];
    VECTOR=[i for i in range(-N,N + 1)]
#    Level_intervals= mu+k*sigma*VECTOR;
    # Level_intervals=[ mu+k*sigma*i for i in VECTOR]

    # method 2
    # Level_intervals=np.linspace(mu-3*sigma, mu+3*sigma, num=M-1)
    Level_intervals=np.linspace(mu-k*sigma, mu+k*sigma, num=M-1)

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

def Load_split_dataset(filepath, csv_filename,list_Subjets,list_Gestures,Trial_split):
    
    print('-->  Loading the original data ')
    # load mat file where the data are saved
    #filepath, csv_filename = browse_file()
    
    cname=filepath+csv_filename
    Data = pd.read_table(cname, sep=',')
    
    Gesture=Data['Gesture'];
    Subject=Data['Subject'];
    Trial=Data['Trial'];
    X=Data[Data.columns[Data.columns.str.contains(pat = 'sEMG')] ];

    X=np.asarray(X);
    Gesture=np.asarray(Gesture).ravel();Subject=np.asarray(Subject).ravel(); Trial=np.asarray(Trial).ravel()

    # Displays
    blcd, classes, data_dic=Explore_dataset(Gesture)
    # explore the dataset:  number f subjects, trails , gestures...
    trials_labels=np.unique(Trial); #print(trials_labels)

    #list_Subjets=np.unique(Subject); print(list_Subjets)
    # list_Gestures=np.unique(Gesture); #print(list_Gestures)

    #%% Split the data into training and testing
    score=list();

    Trial_TR=list();Trial_TS=list();
    for gesture in list_Gestures:
        Trial_TR.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[0]]))
        Trial_TS.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[1]]))



    # Genrate the Training/Testing splits
    y=Gesture
    X_train= X[Trial_TR];  #X_train_mat=matlab.double(X_train.tolist()); #print(X_train)
    y_train= y[Trial_TR];  #y_train_mat=matlab.double(y_train.tolist())
    X_test = X[Trial_TS];  #X_test_mat=matlab.double(X_test.tolist())
    y_test = y[Trial_TS]

    # Displays
    print('\n\n==> Genrate the Training/Testing splits. classes:', list_Gestures)
    trials_labels_TR=np.unique(Trial[Trial_TR]); print('Training size', len(y_train) ,' ,  trials labels: ',trials_labels_TR)
    trials_labels_TS=np.unique(Trial[Trial_TS]); print('Testing size', len(y_test) ,' ,  trials labels: ',trials_labels_TS)

    y_test =y_test  - np.min(y)
    y_train=y_train - np.min(y)

    ## Density Plot and Histogram of all arrival delays
    # import seaborn as sns
    # sns.distplot(X_train.flatten(), hist=True, kde=True, 
    #              bins=int(180/5), color = 'darkblue', 
    #              hist_kws={'edgecolor':'black'},
    #              kde_kws={'linewidth': 4})
    
    
    

    return X, Gesture,Subject, Trial, X_train, y_train, X_test, y_test, csv_filename


def Generate_QuPWM_features(QuPWM_object, Q_sequences, QuPWM_set=''):

    fPWM_dict={}
    fPWM = np.array([],dtype=float)
    
    if QuPWM_set=='':
        QuPWM_set=[]
        for m in range(1,QuPWM_object.m +1):
            QuPWM_set.append('m'+str(m))
            QuPWM_set.append('p'+str(m))
    
    # print('\n-->Selected QuPWM features',QuPWM_set)           
    
    # Get the PWM for each morif for each class        
    for C in QuPWM_object.classes:
        
        ##Build the different kmers
        # print('\n-->  Compute the features the class ',C) 
        
        for m in QuPWM_set:
            # print('\n     --> Features of kmers', m)
            PWM_name='C'+str(C)+'-m'+m[1]
            PWM_C=QuPWM_object.mPWM[PWM_name]
            name_motifs='m'+m[1]
            motifs=QuPWM_object.motifs[name_motifs]

            
            if m[0]=='m': # Replace each sample by its PWM weights
                name_motifs='m'+str(m[1])
                motifs=QuPWM_object.motifs[name_motifs]
                new_features=np.zeros([Q_sequences.shape[0],Q_sequences.shape[1]-int(m[1])+1]) 
            
                for i in range(0,PWM_C.shape[1]):                          # Scan all kmers motifs
                    pattern = motifs[i]                                 # given this m-mer
                    for p in range(0,np.size(new_features,1)):                              #calculate k-mer frequency of order m at all posiitons p
                        idx0=Q_sequences[:,p:p+int(m[1])] == pattern
                        idx=np.where(idx0.all(axis=1))
                        new_features[idx,i]=PWM_C[p,i]
                        
              
            
            if m[0]=='p': # Sum the motifs the weights seperatly
                new_features=np.zeros([Q_sequences.shape[0],PWM_C.shape[1]]) 
                for i in range(0,PWM_C.shape[1]):                          # Scan all kmers motifs
                    pattern = motifs[i]                                 # given this m-mer
                    for p in range(0,np.size(Q_sequences,1)-int(m[1])+1):                              #calculate k-mer frequency of order m at all posiitons p
                        # print(i,p)
                        idx0=Q_sequences[:,p:p+int(m[1])] == pattern
                        idx=np.where(idx0.all(axis=1))
                        new_features[idx,i]=new_features[idx,i] + PWM_C[p,i]
                        
              
                    
                    
            if m[0]=='s':   # sum all weights to build one features vector 
                new_features=np.zeros([Q_sequences.shape[0],PWM_C.shape[1]]) 
                
                
                new_features=np.zeros([Q_sequences.shape[0],1]) 
                for i in range(0,PWM_C.shape[1]):                          # Scan all kmers motifs
                    pattern = motifs[i]                                 # given this m-mer
                    for p in range(0,np.size(Q_sequences,1)-int(m[1])+1):                              #calculate k-mer frequency of order m at all posiitons p
                        idx0=Q_sequences[:,p:p+int(m[1])] == pattern
                        idx=np.where(idx0.all(axis=1))
                        new_features[idx]=new_features[idx] + PWM_C[p,i]
                
                new_features=np.reshape(new_features, (new_features.shape[0], 1))

                
            # Generate the features
            fPWM_name='C'+str(C)+'-'+m
                            
            # print(new_features.shape)

            fPWM_dict[fPWM_name]=new_features
            
            if len(fPWM)==0:
                fPWM=new_features
                
            else:
                fPWM=np.concatenate([fPWM,new_features], axis=1)
            
      
    return fPWM_dict, fPWM

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

