import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt

data_files = {'original': '../data/original.csv',
              'augmented': '../dataaugmented.csv',
              'reduced': '../data/reduced.csv'}

def read_in_data(filetype='original'):
    print('Filetype to be used is : {}'.format(filetype))
    data = pd.read_csv(data_files[filetype])
    print(data.info())
    return data

def create_train_test_data(data):
    ''' Splits the dataset into data for training and testing ultimate testing
    '''
    test_size = 0.3
    print('\n\n\nUsing {}% of the data\n\n'.format(test_size*100))

    X_all = data[:, :-1]
    y_all = (data[:, -1] == 'Abnormal').astype(int)
    X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test = \
        train_test_split(X_all, y_all, test_size=test_size)
    return X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test

def find_best_svm_model(columns, type_of_dataset, scoring='accuracy', C_range=None, gamma_range=None, n_folds=20):
    # list of soft-margin classifiers to test
    if C_range is None:
        C_range = np.logspace(-3, 0, 10)

    if gamma_range is None:
        gamma_range = np.logspace(-6, -3, 10)

    data = read_in_data(type_of_dataset)
    data = data.iloc[:, columns].values
    print('Inside find_best_svm_model. Shape of dataset to be used: {}'.format(data.shape))
    X_crossval, X_test, y_crossval, y_test = create_train_test_data(data)

    param_grid = {'gamma': gamma_range, 'C': C_range}
    NUMBER_OF_FOLDS = 5

    # the default kernel is rbf hence not explicitly setting it
    # When set cv to an integer, it will use StratifiedKFold therefore shuffles once
    clf = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=NUMBER_OF_FOLDS)
    clf.fit(X_crossval, y_crossval)

    print('cv_results\n')
    print(clf.cv_results_.keys())

    print("The best parameters are %s with a score of %0.2f" % (clf.best_params_, clf.best_score_))

    # Testing on unseen data
    new_clf = svm.SVC(C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    new_clf.fit(X_crossval, y_crossval)
    predicted_y = new_clf.predict(X_test)


    score_result = accuracy_score(y_test, predicted_y)
    print('\n\n#########Testing best params C: {}, gamma: {}, on dataset: {} with dimensions: {}##########\nAccuracy score is: {}\n\n\n'.format(
        clf.best_params_['C'], clf.best_params_['gamma'], type_of_dataset, data.shape, score_result))
    return clf.cv_results_['param_C'], clf.cv_results_['param_gamma'], clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']



if __name__ == '__main__':
    '''
        At the command line need to enter python3 <python-script-name>
        If only enter that, then the program will assume want to just use the original dataset 
        , all 12 features. 

        Optionally, can enter the dataset types to be used and the columns to be used:
        python <script name> --dataset_type reduced,original,augmented --columns 1,2,3,7,8,9
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type',
                        help='There are 3 datasets: original, reduced, augmented\nOriginal dataset will be used is not specified')
    parser.add_argument('--columns',
                        help='Choose the columns to use from 0 - 11. Separate columns by comma. If not specify, default of all 12 will be used')

    args = parser.parse_args()

    datasets_to_use = ['original']
    best_accuracy_for_all_datasets = []

    # if have --dataset_type as an option
    if args.dataset_type:
        datasets_to_use = args.dataset_type.split(',')
    # if have --columns as an option
    if args.columns:
        # add target column to specified columns
        cols_to_use = list(map(int, args.columns.split(',')))
        cols_to_use.append(12)
    else:
        cols_to_use = list(range(0,13))
        print('Will use all 12 features')

    score = ['accuracy']

    for type_of_dataset in datasets_to_use:
        Cs, gammas, mean_test_score, mean_error = \
            find_best_svm_model(cols_to_use, type_of_dataset, scoring=score, C_range=None, gamma_range=None, n_folds=None)

        data = {'Cs': np.array(Cs), 'gammas': np.array(gammas), 'mean_score': np.array(mean_test_score),
                'mean_error':np.array(mean_error)}
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.title(type_of_dataset)
        ax.set_xlabel('gamma value')
        ax.set_ylabel('mean cv score')

        c_gamma_df = pd.DataFrame(data)

        c_vals = set(data['Cs'])

        for c in c_vals:
            subset_df = c_gamma_df.loc[c_gamma_df['Cs'] == c]
            g_vals = subset_df['gammas']

            scores = subset_df['mean_score']
            ax.plot(g_vals, subset_df.mean_score, label='C: ' + str(c))

        ax.legend(loc=4)
        plt.show()


def test_svm_model(dataset_to_use, best_c_param, X_train, y_train, X_test, y_test, score='accuracy'):
    clf = svm.SVC(kernel='rbf', C=best_c_param)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)
    score_result = None
    if score == 'accuracy':
        score_result = accuracy_score(y_test, predicted_y)

   #elif score == 'f1':
   #    score_result = f1_score(y_test, predicted_y)
   #elif score == 'precision':
   #    score_result = precision_score(y_test, predicted_y)
   #elif score == 'recall':
   #    score_result = recall_score(y_test, predicted_y)
   #elif score == 'roc_auc':
   #    score_result = roc_auc_score(y_test, predicted_y)

    print('\n\n**************** Testing on {} dataset****************\n\n'.format(dataset_to_use))
    print('Using evaluation method: {}'.format(score))
    print('Using unseen data, the overall score of this model is {}\n\n'.format(score_result))
    return score_result
