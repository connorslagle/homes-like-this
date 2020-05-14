import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB
from data_pipelines import ImagePipeline

def load_data(file_dir):
    img_pipe = ImagePipeline(file_dir)
    img_pipe.read()
    img_pipe.vectorize()
    X = img_pipe.features
    y = img_pipe.labels
    return X, y

def fname_to_city(df, y):
    city = [] 
    for elem in y: 
        if elem in df.image_file.values: 
            city.append(df.city[df.image_file == elem].values[0])
    return city

if __name__ =="__main__":

    df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')
    X, y = load_data('../data/proc_images/64/')

    target = fname_to_city(df,y)

    # holdout
    X_tt, X_holdout, y_tt, y_holdout = train_test_split(X, target, stratify=1, random_state=1)


    # print("\nsklearn's Implementation")
    # mnb = SKMultinomialNB()
    # mnb.fit(X_tr_vec, y_train)
    # print('Test Accuracy:', mnb.score(X_te_vec, y_test))
    # sklearn_predictions = mnb.predict(X_te_vec)

    # # Assert I get the same results as sklearn
    # # (will give an error if different)
    # assert np.all(sklearn_predictions == my_predictions)