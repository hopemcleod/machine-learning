import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


data_files = {'original': '../data/original.csv',
              'augmented': '../dataaugmented.csv',
              'reduced': '../data/reduced.csv'}

# Percentage of data to be used for ultimate testing (i.e. data not seen by model during training)
test_size = 0.3

def read_in_data(filetype='original'):
    print('Filetype to be used is : {}'.format(filetype))
    data = pd.read_csv(data_files[filetype])
    print(data.info())
    return data

def create_train_test_data(data):
    ''' Splits the dataset into data for training and testing
    '''
    X_all = data[:, :-1]
    y_all = (data[:, -1] == 'Abnormal').astype(int)
    X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test = \
        train_test_split(X_all, y_all, test_size=test_size)
    return X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test

def find_best_svm_model(columns, type_of_dataset, scoring='accuracy', c_val_list=None, n_folds=20):
    data = read_in_data(type_of_dataset)
    data = data.iloc[:, columns].values
    print('\n\n#########Inside find_best_svm_model. Shape of dataset to be used: {}########\n\n'.format(data.shape))
    # list of soft-margin classifiers to test
    if c_val_list is None:
        c_val_list = np.logspace(-3, 0, 10)

    X_crossval, X_test, y_crossval, y_test = create_train_test_data(data)

    print("y_test = {}".format(y_test))
    all_scores = []

    for C in c_val_list:
        clf = svm.SVC(kernel='linear', C=C)
        scores = cross_val_score(clf, X_crossval, y_crossval, cv=n_folds, scoring=scoring)
        print('Scores from CV process: {}'.format(scores))
        all_scores.append(scores)

    all_scores = np.array(all_scores)
    print("all_scores = {}".format(all_scores))

    score_vals = np.mean(all_scores, axis=1)
    print('Score_vals: {}'.format(score_vals))
    score_errs = np.std(all_scores, axis=1) / np.sqrt(n_folds)
    print('Score_errs: {}'.format(score_errs))

    return clf, c_val_list, score_vals, score_errs, X_crossval, y_crossval, X_test, y_test

def get_max_score_pair(c_score_pair):

    tmp = [(0, 0)]
    for c_score in c_score_pair:
        if c_score[1] > tmp[0][1]:
            tmp = []
            tmp.append(c_score)

    print(tmp)
    return tmp[0]

def test_svm_model(dataset_to_use, best_c_param, X_train, y_train, X_test, y_test, score):

    clf = svm.SVC(kernel='linear', C=best_c_param)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)
    score_result = None
    if score == 'accuracy':
        score_result = accuracy_score(y_test, predicted_y)
    elif score == 'f1':
        score_result = f1_score(y_test, predicted_y)
    elif score == 'precision':
        score_result = precision_score(y_test, predicted_y)
    elif score == 'recall':
        score_result = recall_score(y_test, predicted_y)
    elif score == 'roc_auc':
        score_result = roc_auc_score(y_test, predicted_y)

    print('\n\n**************** Testing on {} dataset****************\n\n'.format(dataset_to_use))
    print('Using evaluation method: {}'.format(score))
    print('Using unseen data, the overall score of this model is {}\n\n'.format(score_result))
    return score_result

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

    parser.add_argument('--eval_method', help='Choose either accuracy, f1, precision, recall, roc_auc')
    args = parser.parse_args()

    datasets_to_use = ['original']
    #score = 'accuracy'
    score = 'f1'
    best_accuracy_for_all_datasets = []

    # if have --dataset_type as an option
    if args.dataset_type:
        datasets_to_use = args.dataset_type.split(',')
    # if have --columns as an option
    if args.columns:
        # add target column to columns specified
        cols_to_use = list(map(int, args.columns.split(',')))
        cols_to_use.append(12)
    else:
        cols_to_use = list(range(0,13))
        print('Will use all 12 features')

    if args.eval_method:
        score = args.eval_method
    # else use default which is 'accuracy'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Soft Margin Regulariser Value (C)')
    ax.set_ylabel('Average Cross Validation Score')
    colors = ['r', 'g', 'b', 'm', 'c']
    all_models = []
    best_score_for_all_datasets = []
    datasets_used = []
    for dataset in enumerate(datasets_to_use):
        print('\n\n\nAbout to analyse the {} dataset ...'.format(dataset))
        clf, c_vals, score_vals, score_errs, X_train, y_train, X_test, y_test = find_best_svm_model(cols_to_use, dataset[1], scoring=score, c_val_list=None, n_folds=20) # NB: dataset is enumerate type therefore need to get 2nd element of enumerate
        c_score_pair = list(zip(c_vals, score_vals))

        # For each dataset used, store the highest accuracy score
        max_score = get_max_score_pair(c_score_pair)
        best_score_for_all_datasets.append({dataset[1]: max_score}) # dataset[1] is the name of the dataset
        all_models.append({dataset[1]: (X_test, y_test, X_train, y_train)})
        ax.plot(c_vals, score_vals, color=colors[dataset[0]], label=dataset[1])
        upper = score_vals + score_errs
        lower = score_vals - score_errs
        ax.fill_between(c_vals, lower, upper, alpha=0.5, color=colors[dataset[0]])
        datasets_used.append(dataset[1])

    title_str = ''
    for d in datasets_used:
        title_str += '{}/ '.format(d)
        plt.title('Outcome for {} dataset using {} evaluator'.format(title_str, score))
    ax.set_xscale('log')
    ax.legend(loc=0)
    plt.show()

    tmp = [{'empty':(0,0)}]

    # check which dataset has the best score and use that dataset and parameter(s)
    for element in best_score_for_all_datasets:
        key = element.keys()
        if element.get((list(key)[0]))[1] > tmp[0].get(list(tmp[0].keys())[0])[1]:
            tmp = []
            tmp.append(element)

    dataset_to_use = list(tmp[0].keys())[0]
    # set the c_val to use
    c_val_to_use = tmp[0].get(list(tmp[0].keys())[0])[0]

    test_svm_model(dataset_to_use, c_val_to_use, X_train, y_train, X_test, y_test, score)
