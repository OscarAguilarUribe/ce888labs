# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:45:49 2019
@author: oa18525 - Oscar Eduardo Aguilar Uribe
For the purpose of CE888-Data Science and Decision Making
Assignment 2 - Final Code

Function knowledge_dist(dataset,num_trees,num_nearneigh,num_bins)

-----DESCRIPTION-----
This function aims to identify if a decision tree that learns from the proba-
bilities of a random forest performs better or worse than the random forest
per se. In order to do so, the probabilities are used to generate a new data
set and plugged into a bining technique (histogram) to yield different classes.
At the same time, the function trains different classifiers such as: svm, knn,
decision tree (binary classification), random forest and the distilled 
decision tree.

To compare the performance of the 5 different classifiers, all the classifiers
are tunned with a validation set and tested with a testing set. Is important 
to mention that within this function there are two other ones that do not take 
any input. These are: 
    
    accuracies(): 
    calculates tuning (test set) and final (validation set) accuracies for 
    each classifier
    classreports(): 
    calculates the classification report (precision, recal, f1 score, and 
    support) for each classifier, as well as their confusion matrices.

Please change the path in line 68 to the one where the .CSV files are  
in your computer 

-----INPUTS-----
num_trees: 
    Number of decision trees for the random forest classifier

num_nearneigh: 
    Number of nearest neighbours for the k-nn classifier

num_bins:
    Number of bins for the multiclass classifier decision tree

-----OUTPUTS-----
a: A list of the tuning and final testing accuracies for each classifier

b: A list of the classification reports and confusion matrices of each 
   classifier
"""
#Import Libaries
import numpy as np
import pandas as pd
import graphviz
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report

def knowledge_dist(num_trees,num_nearneigh,num_bins):
    #Read Data Set file
    data = pd.read_csv("/Users/oscaraguilar/Desktop/dataset3_clean.csv", 
                        sep = ",", header = 0)
    attributes = ['BI-RADS','Age','Shape','Density']

    #Split in feature data and target data
    columns=data.shape[1]
    x=data.values[:,1:columns-1]
    y=data.values[:,-1]
    x=np.array(x, dtype=float)

    #Split in training, testing and validation sets
    #Generating train and test data sets, 80% for training and 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, 
                                                        random_state=10)
    #Splitting train set in training and validation sets, 75% for 
    #training and 25% for testing
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                      test_size=0.25, 
                                                      random_state=10)

    #Building of classifiers
    clf1 = RandomForestClassifier(n_estimators=num_trees, criterion='gini', 
                                  random_state=10) #Binary Class
    clf2 = DecisionTreeClassifier(criterion='gini',splitter='best',
                                  random_state=10, 
                                  min_impurity_decrease=0.0001) #Binary Class
    clf3 = DecisionTreeClassifier(criterion='gini',splitter='best',
                                  random_state=10, 
                                  min_impurity_decrease=0.0001) #Multi Class
    clf4 = SVC(kernel='linear')
    clf5 = KNeighborsClassifier(n_neighbors=num_nearneigh)

    #Training Classifiers for binary and multiclass classification problems
    clf1=clf1.fit(x_train,y_train)
    clf2=clf2.fit(x_train,y_train)
    clf4=clf4.fit(x_train,y_train)
    clf5=clf5.fit(x_train,y_train)

    #Get probabilities
    prob=clf1.predict_proba(x)
    #Convert to data frame
    df=pd.DataFrame(prob)
    #Drop '1' class probabilities
    p1=df.drop(1, axis=1)
    #Conver to numpy array
    p2=np.array(p1, dtype=float)
    #Bining process
    hist, bin_edges = np.histogram(p2, bins=num_bins)
    #Retrieve the bin number of each probability 
    bin_number= np.digitize(p2,bin_edges)
    #Create new data set for multiclass classification
    #prob_dataset=np.concatenate((x,bin_number), axis=1)
    #print(prob_dataset)

    #Generating train and test data sets, 70% for training and 30% for testing
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x,bin_number,
                                                            test_size=0.3, 
                                                            random_state=10)
    #Training multiclass classification decision tree
    clf3=clf3.fit(x_train2,y_train2)
    
    #Tuning Testing with test sets
    rf=clf1.predict(x_val)
    dt=clf2.predict(x_val)
    dt_multi=clf3.predict(x_test2)
    svm=clf4.predict(x_val)
    knn=clf5.predict(x_val)

    #Final testing with validation set (data never seen by the classifiers)
    rf_val=clf1.predict(x_test)
    dt_val=clf2.predict(x_test)
    dt_multi_test=clf3.predict(x_test)
    svm_val=clf4.predict(x_test)
    knn_val=clf5.predict(x_test)

    #Binarize clf3 output into 2 classes (binary)
    binarize=np.where(dt_multi_test >= (num_bins/2),0,1)

    #Accuracies and classification reports
    def accuracies():
        k_fold = KFold(n_splits=10)
        a1=accuracy_score(y_val,rf)*100
        a2=accuracy_score(y_val,dt)*100
        a3=accuracy_score(y_test2,dt_multi)*100
        a4=accuracy_score(y_val,svm)*100
        a5=accuracy_score(y_val,knn)*100
        a6=accuracy_score(y_test,rf_val)*100
        a7=accuracy_score(y_test,dt_val)*100
        a8=accuracy_score(y_test,binarize)*100
        a9=accuracy_score(y_test,svm_val)*100
        a10=accuracy_score(y_test,knn_val)*100
    
        print('Accuracies:')
        print("Tuning Accuracy random forest = %s" % str(a1))
        print("Final Accuracy random forest = %s" % str(a6))
        score_1 = cross_val_score(clf1, x, y, cv=k_fold, n_jobs=-1)
        print('Average random forest accuracy: {} %'.format(np.mean(score_1)*100))
        print('----------------------------------')
        print("Tuning Accuracy binary decision tree = %s" %str(a2))
        print("Final Accuracy binary decision tree = %s" %str(a7))
        score_2 = cross_val_score(clf2, x, y, cv=k_fold, n_jobs=-1)
        print('Average binary decision tree accuracy: {} %'.format(np.mean(score_2)*100))
        print('----------------------------------')
        print("Tuning Accuracy multi class decision tree = %s" %str(a3))
        print("Final Accuracy multi class decision tree = %s" %str(a8))
        score_3 = cross_val_score(clf3, x, y, cv=k_fold, n_jobs=-1)
        print('Average multi class decision tree accuracy: {} %'.format(np.mean(score_3)*100))
        print('----------------------------------')
        print("Tuning Accuracy binary SVM = %s" %str(a4))
        print("Final Accuracy binary SVM = %s" %str(a9))
        score_4 = cross_val_score(clf4, x, y, cv=k_fold, n_jobs=-1)
        print('Average SVM accuracy: {} %'.format(np.mean(score_4)*100))
        print('----------------------------------')
        print("Tuning Accuracy binary KNN (10NN) = %s" %str(a5))
        print("Final Accuracy binary KNN (10NN) = %s" %str(a10))
        score_5 = cross_val_score(clf5, x, y, cv=k_fold, n_jobs=-1)
        print('Average KNN accuracy: {} %'.format(np.mean(score_5)*100))
        print('----------------------------------')
        return;
    
    def classreports():
        classes=['0-Healthy','1-Unhealthy']
        print('Class reports and confusion matrices')
        print('RANDOM FOREST')
        print(classification_report(y_test,rf_val,target_names=classes))
        print(confusion_matrix(y_test,rf_val))
        print('----------------------------------')
        print('BINARY DECISION TREE')
        print(classification_report(y_test,dt_val,target_names=classes))
        print(confusion_matrix(y_test,dt_val))
        print('----------------------------------')
        print('BINARIZED MULTICLASS DECISION TREE')
        print(classification_report(y_test,binarize,target_names=classes))
        print(confusion_matrix(y_test,binarize))
        print('----------------------------------')
        print('SVM')
        print(classification_report(y_test,svm_val,target_names=classes))
        print(confusion_matrix(y_test,svm_val))
        print('----------------------------------')
        print('K-NEAREST NEIGHBOURS')
        print(classification_report(y_test,knn_val,target_names=classes))
        print(confusion_matrix(y_test,knn_val))
        return;

    #Plot Distilled and binary DTs
    classes2 = ['0','1']
    classes3 = ['0','1','2','3','4','5','6','7','8','9','10']
    
    dot_data2 = tree.export_graphviz(clf2, out_file=None, 
                                    feature_names=attributes, 
                                    class_names=classes2, 
                                    filled=True, rounded=True)
    
    graph2 = graphviz.Source(dot_data2) 
    graph2.render("Binary DT", directory='/Users/oscaraguilar/Desktop', 
                 format='png')
    
    
    dot_data = tree.export_graphviz(clf3, out_file=None, 
                                    feature_names=attributes, 
                                    class_names=classes3, 
                                    filled=True, rounded=True)
    
    graph = graphviz.Source(dot_data) 
    graph.render("Distilled tree", directory='/Users/oscaraguilar/Desktop', 
                 format='png')
    
    #Plot confusion matrices
    #cm= confusion_matrix(y_test,binarize)
    #plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    #classNames = ['Healthy','Unhealthy']
    #plt.title('DISTILLED DECISION TREE CONFUSION MATRIX')
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #tick_marks = np.arange(len(classNames))
    #plt.xticks(tick_marks, classNames, rotation=45)
    #plt.yticks(tick_marks, classNames)
    #s = [['TN','FP'], ['FN', 'TP']]
        #for i in range(2):
            #for j in range(2):
                #plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
        #plt.savefig('/Users/oscaraguilar/Desktop')
    
    a = accuracies()
    b = classreports()
    return a,b

print(knowledge_dist(500,10,5))
