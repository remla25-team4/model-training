#this is the mdoel training file


import joblib


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from model_training.data_processing import get_dataset

def train_model(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    print(acc)

    return classifier

def save_model(model):
    joblib.dump(model, 'models/naive_bayes.joblib')

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataset()
    
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model)
