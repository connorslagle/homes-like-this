from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as SKMultinomialNB



if __name__ =="__main__""

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


    X_tr_toks = [my_tokenizer(doc) for doc in X_train]
    X_te_toks = [my_tokenizer(doc) for doc in X_test]

    cv = CountVectorizer(tokenizer=my_tokenizer)
    X_tr_vec = cv.fit_transform(X_train)
    X_te_vec = cv.transform(X_test)




    print("\nsklearn's Implementation")
    mnb = SKMultinomialNB()
    mnb.fit(X_tr_vec, y_train)
    print('Test Accuracy:', mnb.score(X_te_vec, y_test))
    sklearn_predictions = mnb.predict(X_te_vec)

    # Assert I get the same results as sklearn
    # (will give an error if different)
    assert np.all(sklearn_predictions == my_predictions)