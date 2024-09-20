import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data_files = {'original': '../data/original.csv',
              'augmented': '../dataaugmented.csv',
              'reduced': '../data/reduced.csv'}

def read_in_data(filetype='original'):
    print('Filetype to be used is : {}'.format(filetype))
    data = pd.read_csv(data_files[filetype])
    return data


def create_train_test_data(data):
    ''' Splits the dataset into data for training and ultimate testing
    '''
    X_all = data[:, :-1]
    y_all = (data[:, -1] == 'Abnormal').astype(int)
    X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test = \
        train_test_split(X_all, y_all, test_size=0.3)
    return X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test

def find_best_logreg_model(columns, type_of_dataset, scoring='accuracy', c_val_list=None, n_folds=20):

    data = read_in_data(type_of_dataset)
    data = data.iloc[:, columns].values
    print('Inside find_best_logreg_model. Shape of dataset to be used: {}'.format(data.shape))
    if c_val_list is None:
        c_val_list = np.logspace(-3, 0, 10)
        print(c_val_list)

    X_crossval, X_test, y_crossval, y_test = create_train_test_data(data)

    print("y_test = %r" % (y_test,)) # ???

    all_scores = []
    for C in c_val_list:
        clf = LogisticRegression(C=C, penalty='l1')
        scores = cross_val_score(clf, X_crossval, y_crossval, cv=n_folds, scoring=scoring)
        print('Scores from CV process: {}'.format(scores))
        all_scores.append(scores)
    all_scores = np.array(all_scores)

    print('\n\nThe logistic regression object: {}\n'.format(clf))
    print("all_scores = %r" % (all_scores,))

    accuracy_vals = np.mean(all_scores, axis=1)
    print('Accuracy_vals: {}'.format(accuracy_vals))
    accuracy_errs = np.std(all_scores, axis=1) / np.sqrt(n_folds)
    print('Accuracy_errs: {}'.format(accuracy_errs))

    return clf, c_val_list, accuracy_vals, accuracy_errs, X_crossval, y_crossval, X_test, y_test

def get_max_score_pair(c_accuracy_pair):

    tmp = [(0, 0)]
    for c_accuracy in c_accuracy_pair:
        if c_accuracy[1] > tmp[0][1]:
            tmp = []
            tmp.append(c_accuracy)

    print(tmp)
    return tmp[0]

def test_logreg_model(dataset_to_use, best_c_param, X_train, y_train, X_test, y_test):
    '''
    Need to build the model again using the training data and training target.
    '''

    clf = LogisticRegression(C=best_c_param)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)
    accuracy_result = accuracy_score(y_test, predicted_y)
    print('\n\n**************** Using unseen data, the overall accuracy of this model is {} ****************\nUsing {} dataset\n'.format(accuracy_result, dataset_to_use))
    return accuracy_result

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

    scoring = ['accuracy']
    #scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Soft Margin Regulariser Value (C)')
    ax.set_ylabel('Average Cross Validation Score')
    colors = ['r', 'g', 'b', 'm', 'c']
    all_models = []

    for dataset in enumerate(datasets_to_use):
        print('\n\n\nAbout to analyse the {} dataset ...'.format(dataset))
        for score in scoring:
            clf, c_vals, accuracy_vals, accuracy_errs, X_train, y_train, X_test, y_test = find_best_logreg_model(cols_to_use, dataset[1], scoring=score, c_val_list=None, n_folds=20) # NB: dataset is enumerate type therefore need to get 2nd element of enumerate

            c_accuracy_pair = list(zip(c_vals, accuracy_vals))

            # For each dataset used, store the highest accuracy score
            if score == 'accuracy':
                max_score = get_max_score_pair(c_accuracy_pair)
                best_accuracy_for_all_datasets.append({dataset[1]: max_score}) # dataset[1] is the name of the dataset
                all_models.append({dataset[1]: (X_test, y_test, X_train, y_train)})
            ax.plot(c_vals, accuracy_vals, color=colors[dataset[0]], label=dataset[1])
            upper = accuracy_vals + accuracy_errs
            lower = accuracy_vals - accuracy_errs
            ax.fill_between(c_vals, lower, upper, alpha=0.5, color=colors[dataset[0]])


    ax.set_xscale('log')
    ax.legend(loc=0)
    plt.show()

    tmp = [{'empty':(0,0)}]

    # check which dataset has the best score and use that parameter
    for element in best_accuracy_for_all_datasets:
        key = element.keys()
        if element.get((list(key)[0]))[1] > tmp[0].get(list(tmp[0].keys())[0])[1]:
            tmp = []
            tmp.append(element)

    dataset_to_use = list(tmp[0].keys())[0]
    # set the c_val to use
    c_val_to_use = tmp[0].get(list(tmp[0].keys())[0])[0]

    # using the reduced dataset - need to run the program again but with --dataset_type reduced
    test_logreg_model(dataset_to_use, c_val_to_use, X_train, y_train, X_test, y_test)


