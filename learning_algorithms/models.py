from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



def decicion_tree(criterion='gini', max_depth=10, class_weight=None):
    '''    
    Parameters
    ----------
    criterion : str
        selected from either 'gini' or 'entropy'.
    max_depth : int
        the maximum depth of the trees.
    class_weight : dict or list
        Assigining weights to  class labels e.g., (0:1, 1:2).

    Returns
    -------
    clf : class
        A compiled decision tree model.

    '''
    
    clf = DecisionTreeClassifier(criterion = criterion,
                                 max_depth = max_depth,
                                 class_weight = class_weight,
                                 random_state = None)
    
    return clf


def random_forest(n_estimators=10, criterion='gini', max_depth=5, class_weight=None):
    '''
    Parameters
    ----------
    n_estimators : int
        set the number of trees in the forest, default is 100
    criterion : string
        selected from either 'gini' or 'entropy'.
    max_depth : int
        the maximum depth of the trees..
    class_weight : dict or list
         Assigining weights to  class labels e.g., (0:1, 1:2).

    Returns
    -------
    clf : class
        A compiled random forest model

    '''

    clf = RandomForestClassifier(criterion = criterion,
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 class_weight = class_weight,
                                 n_jobs = -1,
                                 random_state = None)
    return clf





def svm_kernel(kernel='linear', poly_degree=3, c_val=1, class_weight=None):
    '''
    Parameters
    ----------
    kernel : str
        selected from 'linear', 'rbf', 'gaussian', or 'poly'.
    poly_degree : int
        specify the defree of polynomianl if poly kernel used.
    c_val : float
        regularization parameter.
    class_weight : dict or list
        Assigining weights to  class labels e.g., (0:1, 1:2).

    Returns
    -------
    clf : class
        A compiled kernel SVM model.

    '''
    
    clf = svm.SVC(kernel = kernel, degree = poly_degree,
                             gamma = 'scale', C = c_val, tol = 1e-1,
                             class_weight = class_weight,
                             probability = True,
                             random_state  = None,
                             max_iter= -1)
    
    return clf



def knn(neighbors=5, weights=None, n_jobs=-1):
    '''
    Parameters
    ----------
    neighbors : int
        number of neighbors for calculations.
    weights : str
        weighting function for prediction.
        selected from either 'distance' or 'uniform'
    n_jobs : int
        number of parallel jobs.

    Returns
    -------
    clf : class
        A compiled Knearest neighbor model.

    '''

    clf = KNeighborsClassifier(n_neighbors = neighbors,
                                   weights = 'distance',
                                   n_jobs = n_jobs)
    
    return clf

def adab_tree(max_depth=10, criterion='gini', class_weight=None, n_estimators=10):
    '''
    Parameters
    ----------
    max_depth : int
        the maximum depth of the trees..
    criterion : str
        selected from either 'gini' or 'entropy'.
    class_weight : dict or int.
        Assigining weights to  class labels e.g., (0:1, 1:2)..
    n_estimators : int
        the maximum number of estimators.

    Returns
    -------
    clf : class
        A compiled adaptive boosting metal model fitted on a decision tree model.

    '''
    
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = max_depth,
                                                    criterion = criterion,
                                                    class_weight = class_weight),
                                  n_estimators = n_estimators,
                                  random_state = None)    
    return clf


def lda():
    '''
    Returns
    -------
    clf : class
        A compiled LDA model with default settings.

    '''
    clf =  LinearDiscriminantAnalysis()
    
    return clf

def qda():
    '''
    Returns
    -------
    clf : class
        A compiled QDA model with default settings.

    '''
    clf =  QuadraticDiscriminantAnalysis()
    
    return clf

def naive():
    '''
    Returns
    -------
    clf : class
        A compiled naive bayesian model with default settings.

    '''

    clf = naive_bayes.GaussianNB()
    
    return clf


