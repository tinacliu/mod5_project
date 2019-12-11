import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score,precision_score,f1_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV



def log_reg(X_train_scale, X_test_scale, y_train, y_test,
            c=1, penalty = 'l1'):

  """
  Assumption: takes train and test X, y data to fit a logistic refression model
  :return: acc, roc_auc, fpr, tpr, conf_matrix, log_reg (i.e. the model itself)

  Also prints the AUC of the model using test dataset, and plots ROC curve.
  """
  logreg = LogisticRegression(C=c, penalty = penalty, solver='liblinear')

  model_log = logreg.fit(X_train_scale, y_train)

  # predict
  y_hat_train = logreg.predict(X_train_scale)

  preds = logreg.predict(X_test_scale)
  probas = logreg.predict_proba(X_test_scale)

  # Calculate accuracy
  acc = accuracy_score(y_test, preds)
  print('Accuracy is :{0}'.format(round(acc,4)))


  # Check the AUC for predictions
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:,1])
  roc_auc = auc(false_positive_rate,true_positive_rate)
  print('\nAUC is :{0}'.format(round(roc_auc, 3)))

  # Create and print a confusion matrix
  conf_matrix = pd.crosstab(y_test, preds, rownames=['True'], colnames=['Predicted'], margins=True)

  plt.scatter(false_positive_rate, true_positive_rate,
    marker='.', alpha=0.4, c='orange');

  return acc, roc_auc, false_positive_rate, true_positive_rate, conf_matrix, log_reg



def grid_search(X_train, y_train, score = 'roc_auc', cv=3):
  """
  Assumption: takes X, y train data to apply grid search to find best hyperparameter
  based on given score measure.
  :return: best_C, best_Penalty, best_clf.cv_results_

  """
  clf = LogisticRegression(solver='liblinear')

  param_grid = [
      {'penalty' : ['l1', 'l2'],
      'C' : np.logspace(-4, 4, 20)}
  ]

  gs_clf = GridSearchCV(clf, param_grid = param_grid, scoring = score, cv=cv);
  best_clf = gs_clf.fit(X_train, y_train)

  best_c = best_clf.best_params_['C']
  best_pen = best_clf.best_params_['penalty']

  return best_c, best_pen, best_clf.cv_results_



