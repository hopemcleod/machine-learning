import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
from cycler import cycler

monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '-.', '_']) * cycler('marker', ['^',',', '.']))

data_files = {'original': '~/osteo/machine-learning-models/spinal_12_attrib_2_class.csv',
              'augmented': '~/osteo/machine-learning-models/augmented_minority_class.csv',
              'reduced': '~/osteo/machine-learning-models/reduced_dominant_class.csv'}


def read_in_data(filetype='original'):
    ''' Default dataset is the original one'''

    print('Filetype to be used is : {}'.format(filetype))
    data = pd.read_csv(data_files[filetype])
    print(data.info())
    return data

def create_train_test_data(data):
    ''' Splits the dataset into data for training and testing ultimate testing
        The training data is split into folds for cross validation. One of the
        folds is used for testing whilst the model is being trained.
    '''
    test_size = 0.3
    print('\n\n\nUsing {}% of the data\n\n'.format(test_size*100))

    X_all = data[:, :-1]
    y_all = (data[:, -1] == 'Abnormal').astype(int)
    X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test = \
        train_test_split(X_all, y_all, test_size=test_size)
    return X_for_cross_val, X_ultimate_test, y_for_cross_val, y_ultimate_test

def find_best_svm_model(columns, type_of_dataset, scoring='accuracy', C_range=None, gamma_range=None, n_folds=20):
#def grid_search_svm(ifname, scoring=score, C_range=None, gamma_range=None, n_splits=5):
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

    # What's the advantage of using StratifiedShiffleSplit? My understanding is that this cross validation
    # object will shuffle the sets and create different folds each time. Whereas with KFold, each fold is
    # just shifted along to create the new training and test data. I'm going to use the same KFold as I did with
    # the linear svm kernel to keep it consistent.
    #kf = KFold(n_splits=NUMBER_OF_FOLDS)

    # Combine the training input with training target created with the train_test_split
    #data_for_cross_val = X_crossval.join(y_crossval)

    # the default kernel is rbf hence not explicitly setting it
    # When set cv to an integer, it will use StratifiedKFold therefore shuffles once
    clf = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=NUMBER_OF_FOLDS)
    clf.fit(X_crossval, y_crossval)
    # Later when testing the model parameters on unseen data, will call fit

    # Although stackoverflow and warnings given by interpreter says grid_scores is
    # deprecated, it seems to be the most useful gridsearchcv attribute. However, cannot use it
    # as interpreter crosses it out in editor
    print('cv_results\n')
    print(clf.cv_results_.keys()) # I want the param_C results, param_gamma, and mean_test_score. How get? I'm hoping the arrays
    # for all these are the same size

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
        #data = read_in_data(args.dataset_type)
    #else:
        #data = read_in_data()
    # if have --columns as an option
    if args.columns:
        # split the --columns parameter to make a list of individual columns and add the last column also as this is the target
        cols_to_use = list(map(int, args.columns.split(',')))
        cols_to_use.append(12)
    else:
        #cols_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        cols_to_use = list(range(0,13))
        print('Will use all 12 features')

    score = ['accuracy']

    for type_of_dataset in datasets_to_use:
        Cs, gammas, mean_test_score, mean_error = \
            find_best_svm_model(cols_to_use, type_of_dataset, scoring=score, C_range=None, gamma_range=None, n_folds=None)

        data = {'Cs': np.array(Cs), 'gammas': np.array(gammas), 'mean_score': np.array(mean_test_score),
                'mean_error':np.array(mean_error)}
        #type_of_dataset
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.title(type_of_dataset)
        ax.set_xlabel('gamma value')
        ax.set_ylabel('mean cv score')
        ax.set_prop_cycle(monochrome)

        c_gamma_df = pd.DataFrame(data)

        c_vals = set(data['Cs'])

        for c in c_vals:
            subset_df = c_gamma_df.loc[c_gamma_df['Cs'] == c]
            g_vals = subset_df['gammas']

            scores = subset_df['mean_score']
            ax.plot(g_vals, subset_df.mean_score, label='C: ' + str(c))

        ax.legend(loc=4)
        plt.show()


def test_svm_model(dataset_to_use, cols_to_use, best_c_param, X_train, y_train, X_test, y_test, score='accuracy'):
    '''
    Need to build the model again using the training data and training target. Once have a handle/object for that model
    can then can use test data to check the precision of the model
    '''
    # {dataset[1]: (X_test, y_test, X_train, y_train)})
    clf = svm.SVC(kernel='rbf', C=best_c_param)
    clf.fit(X_train, y_train)
    predicted_y = clf.predict(X_test)
    score_result = None
    if score == 'accuracy':
        score_result = accuracy_score(y_test, predicted_y)

   ##elif score == 'f1':
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

'''
from: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
scores = [x[1] for x in clf.grid_scores_]
scores = np.array(scores).reshape(len(Cs), len(Gammas))

for ind, i in enumerate(Cs):
    plt.plot(Gammas, scores[ind], label='C: ' + str(i))
'''