import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from data_pipelines import ImagePipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

def load_data(file_dir):
    img_pipe = ImagePipeline(file_dir)
    img_pipe.read()
    img_pipe.vectorize()
    X_from_pipe = img_pipe.features
    y_from_pipe = img_pipe.labels
    return X_from_pipe, y_from_pipe

def fname_to_city(df, X_in, y_in):
    city = []
    idx = []
    for elem in y_in: 
        if elem in df.image_file.values: 
            city.append(df.city[df.image_file == elem].values[0])
            idx.append(y_in.index(elem))

    X_match = X_in[idx,:]

    return X_match, city



if __name__ =="__main__":

    df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')
    X_feat, y_target = load_data('../data/proc_images/32/')

    X_new, target = fname_to_city(df, X_feat, y_target)
    # breakpoint()
    y = np.array(target)
    # holdout
    rand_state = 1
    X_tt, X_holdout, y_tt, y_holdout = train_test_split(X_new, y, random_state=rand_state, test_size=0.20)

    # test/train split
    X_train, X_test, y_train, y_test = train_test_split(X_tt, y_tt, random_state=rand_state, test_size=0.20)

    '''
    Single test/train split
    '''
    mnb = MultinomialNB()
    # mnb.fit(X_train, y_train)

    # y_hat = mnb.predict(X_test)

    # cats = ['Denver','Arvada','Aurora','Lakewood','Centennial','Westminster','Thornton']
    # conf_mat = confusion_matrix(y_test,y_hat, labels=cats, normalize='pred')

    # accuracy = mnb.score(X_test,y_test)

    '''
    3,5,7,9 Kfold split (no real change - sticking with 5)
    '''
    # folds_list = np.arange(2,20,2)
    # test_acc_means = []
    # train_acc_means = []
    # for folds in folds_list:
    kf = KFold()
    kf.get_n_splits(X_tt)
    test_acc = []
    train_acc = []
    for train_index, test_index in kf.split(X_tt):
        X_train, X_test = X_tt[train_index], X_tt[test_index]
        y_train, y_test = y_tt[train_index], y_tt[test_index]
    
        mnb.fit(X_train, y_train)
        y_hat = mnb.predict(X_test)      # Default predict_proba thresh
        y_hat_train = mnb.predict(X_train)

        test_acc.append(np.mean(y_test == y_hat))
        train_acc.append(np.mean(y_train == y_hat_train))

        # test_acc_means.append(np.mean(test_acc))
        # train_acc_means.append(np.mean(train_acc))
    # print('\nTraining Accuracy:\n')
    # for folds, acc in zip(folds_list,train_acc_means):
    #     print(f'# Folds: {folds}\tTrain Acc: {acc:0.3f}')

    # print('\nTesting Accuracy:\n')
    # for folds, acc in zip(folds_list,test_acc_means):
    #     print(f'# Folds: {folds}\tTest Acc: {acc:0.3f}')


    # Project Structure

    # I want to ultimately deploy the recommender to a web-application. For that purpose, the project will be split into two parts. In part one, a scalable data infrastructure (webscraper and pipelines) will be presented along with exploratory data analysis (EDA) and preliminary image featurization. In part two, a convolutional neural network (CNN) autoencoder will be explored for real  

    # - Develop a scalable webscraper (to be deployed on AWS)
    # - Develop robust data cleaning pipelines for image and image metadata
    # - Featurize images

    # Goals for Capstone 3: 
    # - Scrape more data (more cities/listings -> run on AWS)
    # - Combine image features with metadata -> predict where to look by 

    # This project will be presented in two parts.
    
