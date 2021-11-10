import sys
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#import h5py
from lib.Shared_Functions import *
from lib.QuPWM_features import *
from lib.pretty_confusion_matrix import *
from lib.utils import *
import warnings
warnings.filterwarnings("ignore")

print('################################################')
print('#   Classification using PWM features          #')
print('################################################')
# Input parameters
names = 'test-sequences'
m=1                                                                            # kmers order
QuPWM_set=['p1','s1']#,'p2','s2','p3','s3']#,'p4']

# The classifier pipeline
names = ["Logistic Regression",
            "Nearest Neighbors", "Linear SVM","RBF SVM",
            "Decision Tree", "Random Forest",
            "Neural Net", "Naive Bayes", "AdaBoost"]
classifiers = [LogisticRegression(),#(random_state=0, solver='lbfgs',multi_class='multinomial'),
    KNeighborsClassifier(3), SVC(kernel="linear",C=0.025),SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1), GaussianNB(), AdaBoostClassifier()]

# load input data saved in pkl file 
var_filename = 'data/Data_variables.pkl'
X, y = load_variables(var_filename) 
y = np.array( [int(i) for i in y] )
print('classes =', np.unique(y))
# X, y = X.cpu().numpy().reshape(1045,2800), y.cpu().numpy()
print(X.shape)
# show dataset infomation
Explore_dataset(X,y)
# split data to test/train
Q_train, Q_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
# Create the mPWM object
Levels = np.unique(Q_train)
mPWM_object=mPWM(m, Levels)
# Generate the Training/Testing features
#  Build the nPWM matrices for different kMers
mPWM_object.Build_QuPWM_matrices(Q_train,y_train)
# Generate the fatures  
fPWM_dict_train, fPWM_train=Generate_QuPWM_features(mPWM_object, Q_train, QuPWM_set)
fPWM_dict_test,  fPWM_test=Generate_QuPWM_features(mPWM_object, Q_test, QuPWM_set)
Mdl_score, acc_op, clf_name, clf_op=Classification_Train_Test(names, classifiers, fPWM_train, y_train, fPWM_test, y_test)
# save resut tables
Mdl_score.to_csv( 'results.csv', sep=',')
# Plot non-normalized confusion matrix
confusion_matrix_class(clf_op.predict(fPWM_test),y_test)
print('\n\n ########  END  ##########')