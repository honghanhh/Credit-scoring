
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sklearn
from scipy import stats
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
from sklearn.decomposition import PCA
import itertools
import pickle as pkl

# %matplotlib inline

class Model():
    def __init__(self, df):
        self.df = df
        self.cols = ['label',  '1', '2',
       '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
       '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
       '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
       '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51',
       '52', '53', '54', '55', '56', '57', 'age', 'FIELD_3_RESIDUAL',
       'count_NaN', 'count_True', 'count_False', 'region', 'job_cluster']

    def build_model(self):
        self.df = self.df[self.cols]
        categorical_data = self.df.select_dtypes(exclude=[np.number])
        model = LabelEncoder()
        for i in categorical_data:
            self.df[i] = model.fit_transform(self.df[i].astype('str'))

        X = list(self.df.columns)
        X.remove('label')
        Y = self.df['label']

        self.df.fillna(value=self.df[X].mean(),inplace=True)
        # credit_data = pd.get_dummies(self.df, drop_first = True)
        X_train, X_test, y_train, y_test = train_test_split(self.df[X], Y, test_size = 0.15, random_state = 42 )
        
        sm = SMOTE(random_state=2)
        X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        
        scaler = StandardScaler()
        # Fit on training set only.
        scaler.fit(X_train_res)
        # Apply transform to both the training set and the test set.
        X_train = scaler.transform(X_train_res)
        X_test = scaler.transform(X_test)

        pca = PCA(.95) # choose min(#PCs) s.t 95% of the variance is retained.
        pca.fit_transform(X_train) # fitting PCA on the training set only.
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        lr = LogisticRegression(solver = 'lbfgs')
        lr.fit(X_train,y_train_res)
        # pkl.dump(model, open('../model/logreg_model.pkl', 'wb'))
        # return logreg
        y_preds = lr.predict(X_test)
        cnf_matrix_tra = confusion_matrix(y_test, y_preds)
        print("Confusion matrix.{}".format(cnf_matrix_tra))
        print("Recall metric in the test dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
        
    
        # If you measure Gini, check this function out!
    def Gini(y_true, y_pred):
        # check and get number of samples
        assert y_true.shape == y_pred.shape
        n_samples = y_true.shape[0]

        # sort rows on prediction column
        # (from largest to smallest)
        arr = np.array([y_true, y_pred]).transpose()
        true_order = arr[arr[:, 0].argsort()][::-1, 0]
        pred_order = arr[arr[:, 1].argsort()][::-1, 0]

        # get Lorenz curves
        L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
        L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
        L_ones = np.linspace(1 / n_samples, 1, n_samples)

        # get Gini coefficients (area between curves)
        G_true = np.sum(L_ones - L_true)
        G_pred = np.sum(L_ones - L_pred)

        # normalize to true Gini coefficient
        return G_pred * 1. / G_true

    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

if __name__ == "__main__":
    train_pd = pd.read_csv('../data/clean_train.csv')
    model = Model(train_pd)
    model.build_model()