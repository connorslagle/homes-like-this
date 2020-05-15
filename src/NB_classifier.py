import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from data_pipelines import ImagePipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def load_data(file_dir, use_filter=False):
    '''
    Load images from specified directory.

    Outputs featurized (raveled) images for NB Classification model.
    '''

    img_pipe = ImagePipeline(file_dir)
    img_pipe.read()
    if use_filter:
        img_pipe._filter_image()
    img_pipe.vectorize()
    X_from_pipe = img_pipe.features
    y_from_pipe = img_pipe.labels
    return X_from_pipe, y_from_pipe

def fname_to_city(df, X_in, y_in):
    '''
    Searches dataframe for filenames in y -> creates target with city as
    categories.
    
    Returns: city_target and matching X
    '''
    city = []
    idx = []
    for elem in y_in: 
        if elem in df.image_file.values: 
            city.append(df.city[df.image_file == elem].values[0])
            idx.append(y_in.index(elem))

    X_match = X_in[idx,:]

    return X_match, city

def k_folds_mnb(multi_NB_classifier, X_tt, y_tt, n_folds=5):
    '''
    Applies kfold cv on mnb. Was not able to stratify on tt/holdout split,
    trying to accomodate with averaging k-folds

    returns train acc, test acc
    '''

    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X_tt)

    test_acc = []
    train_acc = []
    for train_index, test_index in kf.split(X_tt):

        X_train, X_test = X_tt[train_index], X_tt[test_index]
        y_train, y_test = y_tt[train_index], y_tt[test_index]
    
        multi_NB_classifier.fit(X_train, y_train)
        y_hat = multi_NB_classifier.predict(X_test)
        y_hat_train = multi_NB_classifier.predict(X_train)

        test_acc.append(np.mean(y_test == y_hat))
        train_acc.append(np.mean(y_train == y_hat_train))
    return train_acc, test_acc

if __name__ =="__main__":
    img_size = 128

    # df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')
    # X_feat, y_target = load_data(f'../data/proc_images/{img_size}/')

    # X_new, target = fname_to_city(df, X_feat, y_target)

    # y = np.array(target)
    # # holdout
    # rand_state = 1
    # X_tt, X_holdout, y_tt, y_holdout = train_test_split(X_new, y, random_state=rand_state, test_size=0.20)

    # # test/train split
    # X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, random_state=rand_state, test_size=0.20)

    '''
    Single test/train split
    '''
    # mnb = MultinomialNB()
    # mnb.fit(X_train, y_train)

    # y_hat = mnb.predict(X_test)

    # cats = ['Denver','Arvada','Aurora','Lakewood','Centennial','Westminster','Thornton']
    # conf_mat = confusion_matrix(y_test,y_hat, labels=cats, normalize='true')

    # accuracy = mnb.score(X_test,y_test)


    '''
    3,5,7,9 Kfold split (no real change - sticking with 5)
    '''
    # mnb = MultinomialNB()
    # folds_list = np.arange(2,20,2)
    # test_acc_means = []
    # train_acc_means = []
    # for folds in folds_list:
    #     train_acc, test_acc = k_folds_mnb(mnb, X_tt, y_tt, folds)
    #     test_acc_means.append(np.mean(test_acc))
    #     train_acc_means.append(np.mean(train_acc))
    
    
    # print('\nTraining Accuracy:\n')
    # for folds, acc in zip(folds_list,train_acc_means):
    #     print(f'# Folds: {folds}\tTrain Acc: {acc:0.3f}')

    # print('\nTesting Accuracy:\n')
    # for folds, acc in zip(folds_list,test_acc_means):
    #     print(f'# Folds: {folds}\tTest Acc: {acc:0.3f}')