# ------------------------------------------------------
# Author : Naimish Mani B
# Date : 9th May 2021
# ------------------------------------------------------
# Contains all helper functions to ensure the server runs smoothly
# ------------------------------------------------------

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import shap
import io
import base64
import pickle


BASE = 'https://raw.githubusercontent.com/fractalego/subjectivity_classifier/'
vectorizer = None
clf = None
x_train = None


def load_data():
    text = []
    labels = []

    urls = [
        BASE + 'master/data/subj_dataset/objective_cv_train.txt',
        BASE + 'master/data/subj_dataset/subjective_cv_train.txt',
        BASE + 'master/data/subj_dataset/objective_test.txt',
        BASE + 'master/data/subj_dataset/subjective_test.txt',
    ]

    for url in urls:
        # Because requests returns byte object
        data = requests.get(url).content.decode()
        # Splitting it line wise
        data = data.split('\n')
        for line in data:
            text.append(line)
            if 'objective' in url:
                labels.append(0)
            else:
                labels.append(1)

    df = pd.DataFrame()
    df['text'] = text
    df['labels'] = labels

    # # Print the head and shape, to verify if dataset is loaded properly
    # print(df.head())
    # print(df.shape)
    return df


def train_model(df):
    global vectorizer, clf, x_train

    # TF-IDF to get word vectors
    vectorizer = TfidfVectorizer(max_features=500)
    vectors = vectorizer.fit_transform(df.text)
    words_df = pd.DataFrame(
        vectors.toarray(),
        columns=vectorizer.get_feature_names())

    x = words_df
    y = df.labels

    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    # Build Classifier
    clf = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    clf.fit(x_train, y_train)

    # Evaluate the Classifier
    eval_clf(clf, x_train, y_train)

    return clf


def eval_clf(clf, X, y):

    # Confusion matrix
    # Get Predictions
    y_pred = clf.predict(X)
    confMat = confusion_matrix(y, y_pred)

    # Calculate Values
    pr = precision_score(y, y_pred)
    rc = recall_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    # Print them
    print("Confusion Matrix:\n", confMat)
    print("Precision: ", pr)
    print("Recall: ", rc)
    print("Accuracy: ", acc)


def save_clf():
    with open("model", 'wb') as fh:
        pickle.dump(clf, fh)


def load_clf():
    with open("model", 'rb') as fh:
        c = pickle.load(fh)
    return c


def get_prediction(text):
    global vectorizer, clf
    x = vectorizer.transform([text])
    words_df = pd.DataFrame(
        x.toarray(),
        columns=vectorizer.get_feature_names())

    pred = clf.predict(words_df)
    print(pred)
    return words_df, pred


def get_shap_image(word_df):
    # Yoinked from
    # https://github.com/slundberg/shap/issues/153

    global clf, x_train
    explainer = shap.LinearExplainer(
        clf,
        x_train,
        feature_dependence="independent"
    )

    plt.close("all")

    plt.figure(figsize=(2, 6))

    shap_values = explainer.shap_values(word_df)
    img = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        word_df.iloc[[0]],
        show=False,
        matplotlib=True
    )

    return img


def image_to_b64(img):
    # Taken from
    # https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='png')
    my_stringIObytes.seek(0)
    my_base64_pngData = base64.b64encode(my_stringIObytes.read())
    return my_base64_pngData


def init():
    global clf
    df = load_data()
    clf = train_model(df)


def predict(text):
    p = None
    word_df, pred = get_prediction(text)
    img = get_shap_image(word_df)
    b64 = image_to_b64(img)

    if pred == 0:
        p = "Statement is not an opinion"
    else:
        p = "Statement is an opinion"

    resp = {
        'txt': p,
        'img': b64.decode()
    }

    return resp


if __name__ == '__main__':
    text = "I think that it is going to rain today"

    df = load_data()
    print("Data loaded")
    '''
    clf = train_model(df)
    print("Classifier trained")
    save_clf()
    print("Classifier Saved")
    '''
    clf = load_clf()
    word_df, pred = get_prediction(text)
    if pred == 0:
        print("Statement is not an opinion")
    else:
        print("Statement is an opinion")
    img = get_shap_image(word_df)
    print("Image made")
    b64 = image_to_b64(img)
    print(b64)
