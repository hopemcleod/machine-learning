import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

data_files = {'original': '../data/original.csv',
              'augmented': '../dataaugmented.csv',
              'reduced': '../data/reduced.csv'}

def create_learning_curve(dataset_type='original'):
    '''
    :param dataset_type: 'original' is the default. To run on 'augmented', or 'reduced' datasets, uncomment
    code in main()
    :return:
    '''
    print('About to read in the {} dataset'.format(dataset_type))
    if dataset_type == 'reduced':
        data = pd.read_csv(data_files['reduced'])
        train_sizes = np.arange(10, 160, 10)

    elif dataset_type == 'augmented':
        data = pd.read_csv(data_files['augmented'])
        train_sizes = np.arange(10, 330, 10)
    else:
        data = pd.read_csv(data_files['original'])
        train_sizes = np.arange(10, 240, 10)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    # The data is shuffled because the learning curve method had some samples where there was only one class in it
    training_sizes, train_scores, validation_scores = learning_curve(estimator=LogisticRegression(), X=X, y=y,
                                                                  train_sizes=train_sizes, cv=5,
                                                                  scoring='accuracy', shuffle=True)
    print('Training scores: {}\n\n'.format(train_scores))
    print('\n', '-' * 70)
    print('Validation scores: {}\n\n'.format(validation_scores))

    mean_train_scores = train_scores.mean(axis=1)
    mean_validation_scores = validation_scores.mean(axis=1)
    print('Mean train scores: {}'.format(mean_train_scores))
    print('Mean validation scores: {}'.format(mean_validation_scores))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)


    ax.plot(training_sizes, mean_train_scores, color='r')
    ax.plot(training_sizes, mean_validation_scores, color='g')
    ax.set_title(dataset_type)
    ax.set_xlabel('size of training set')
    ax.set_ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    #create_learning_curve('reduced')
    #create_learning_curve('augmented')
    create_learning_curve('original')