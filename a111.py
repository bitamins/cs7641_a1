from sklearn import tree, neighbors, svm, ensemble, metrics, preprocessing, datasets
from sklearn.model_selection import train_test_split,learning_curve, validation_curve,GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import math

N_JOBS = 1
CV=5

def get_df(id='poker'):
    def get_data(path='poker_data/poker-hand-testing.data', sep=',',columns=None):
        df = pd.read_csv(path,sep=sep)
        if columns is not None:
            df.columns = columns
        return df
    
    if id == 'poker':
        path = 'data/poker_data/poker-hand-testing.data'
        columns = ['s1','c1','s2','c2','s3','c3','s4','c4','s5','c5','hand']
        sep = ','
        df = get_data(path,sep,columns)

    elif id == 'bank':
        #source: https://www.openml.org/t/14965
        path = 'data/bank_data/bank-full.csv'
        columns = None
        sep = ';'
        df = get_data(path,sep,columns)
        encode_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome','y']
        df = cols_to_encoding(df,encode_cols)

    elif id == 'adult':
        path = 'data/people_data/adult.data'
        columns = ['age','workclass','fnlwgt','education','education-num',
                    'marital-status','occupation','relationship','race','sex',
                    'capital-gain','capital-loss','hours-per-week','native-country',
                    'wage-class']
        sep = ','
        df = get_data(path,sep,columns)
        # path = 'data/people_data/adult.test'
        # df2 = get_data(path,sep,columns)
        # df2['wage-class'] = df2['wage-class'].apply(lambda x: x.strip('.'))
        # df = df.append(df2,ignore_index=True)
        
        # print(df.head())
        
        sample_1 = df[df['wage-class'] == df['wage-class'].unique()[0]].index
        sample_2 = df[df['wage-class'] == df['wage-class'].unique()[1]].index
        # print(sample_1,sample_2)
        count = 1000
        # choice_1 = np.random.choice(sample_1, 1000, replace=False)
        # choice_2 = np.random.choice(sample_2, 1000, replace=False)
        indicies = np.concatenate([sample_1[:count],sample_2[:count]],axis=None)
        df = df.loc[indicies]
        df.index = range(df.shape[0])
        # print(df.shape)
        

        # encode_cols = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','wage-class']
        # df = cols_to_encoding(df,encode_cols)
        
    elif id == 'adult2':
        path = 'data/people_data/adult.data'
        columns = ['age','workclass','fnlwgt','education','education-num',
                    'marital-status','occupation','relationship','race','sex',
                    'capital-gain','capital-loss','hours-per-week','native-country',
                    'wage-class']
        sep = ','
        df = get_data(path,sep,columns)
        sample_1 = df[df['sex'] == df['sex'].unique()[0]].index
        sample_2 = df[df['sex'] == df['sex'].unique()[1]].index
        
        df = df[['age','workclass','fnlwgt','education','education-num',
                    'marital-status','occupation','race','sex',
                    'capital-gain','capital-loss','hours-per-week','native-country',
                    'wage-class']].copy()
        
        count = 1000
        indicies = np.concatenate([sample_1[:count],sample_2[:count]],axis=None)
        df = df.loc[indicies]
        df.index = range(df.shape[0])
        # print(df.shape)
        
    elif id == 'balanced_wage':
        path = 'data/wage_balanced_data/balanced_wage.csv'
        df = get_data(path,',')
        return df
        
    elif id == 'iris':
        #source: https://archive.ics.uci.edu/ml/datasets/Iris
        path = 'data/iris_data/iris.data'
        columns = ['sepal_length','sepal_width','petal_length','petal_width','class']
        sep = ','
        df = get_data(path,sep,columns)
        # df = ohe_columns(df)
        
    elif id == 'housing':
        data = datasets.load_boston()
        values = data['data']
        columns = data['feature_names']
        df = pd.DataFrame(values,columns=columns)
        print(df.head())
        
    elif id == 'digits':
        data = datasets.load_digits()
        # print(data)
        values = data['data']
        df = pd.DataFrame(values)
        df['class'] = data['target']
        print(df.head())
        
    elif id == 'wine':
        data = datasets.load_wine()
        df = pd.DataFrame(data['data'])
        df['class'] = data['target']
        # print(df.head())
        
    elif id == 'car':
        path = 'data/car_data/car.data'
        columns = ['buy_cost','maint_cost','door_count','seat_count','trunk_size','safety','class']
        sep = ','
        df = get_data(path,sep,columns)
        # df = ohe_columns(df)

    else:
        df = pd.DataFrame()

    return df

def clean_df(df,label_columns):
    le = preprocessing.LabelEncoder()
    y = df[label_columns]
    df = pd.get_dummies(df.drop(label_columns,axis=1),prefix_sep='_',drop_first=False)
    df[label_columns] = le.fit_transform(y)
    return df

def get_xy(df,label_columns=[]):
    if len(label_columns) == 0:
        label_columns = df.columns[-1]
    X = df.drop(label_columns,axis=1)
    y = df[label_columns]
    return X,y

def plot_learning_curve(train_scores, test_scores, train_sizes,name_str, filename='l_curve.png'):
    #source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.clf()
    plt.figure()
    title_str = name_str
    plt.title('Learning Curve ({})'.format(title_str))
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('plots2/' + '_'.join([name_str,filename]))

def model_learning_curve(model,X,y,name_str,train_sizes=np.linspace(.1,1.0,5)):
    print('learning curve: {}'.format(name_str))

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring='accuracy', cv=CV, n_jobs=N_JOBS, shuffle=True)

    plot_learning_curve(train_scores,test_scores,train_sizes,name_str)

def plot_validation_curve(train_scores, test_scores, param, param_range, name_str,filename='v_curve.png'):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.clf()
    plt.title("Validation {}".format(name_str))
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    param_range = [str(x) for i,x in enumerate(param_range)]
    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('plots2/' + '_'.join([name_str,filename]))

def model_validation_curve(model,X,y,param,param_range,name_str):
    print('hyperparameter: {} - {}'.format(param,param_range))
    print(X.shape,y.shape)
    train_scores, test_scores = validation_curve(
                                    model, X, y, param_name=param, param_range=param_range,
                                    cv=CV, scoring="accuracy", n_jobs=N_JOBS)
    plot_validation_curve(train_scores,test_scores, param,param_range,name_str=name_str)


def run_model(model_dict,data_source):
    
    data_dict = model_dict[data_source]
    
    df = data_dict['data']
    label = data_dict['y']
    
    
    X,y = get_xy(df,label_columns=[x for x in df.columns if any(l in str(x) for l in label)])

    # validation curve param_1
    name_str = '_'.join([model_dict['name'],data_dict['name'],'_'.join(data_dict['y']),data_dict['param_1']])
    model_validation_curve(data_dict['model_initial'],X,y,data_dict['param_1'],data_dict['param_1_range'],name_str=name_str)

    # validation curve param_2
    name_str = '_'.join([model_dict['name'],data_dict['name'],'_'.join(data_dict['y']),data_dict['param_2']])
    model_validation_curve(data_dict['model_working'],X,y,data_dict['param_2'],data_dict['param_2_range'],name_str=name_str)

    # default learning curve
    name_str = '_'.join([model_dict['name'],data_dict['name'],'_'.join(data_dict['y']),'default'])
    model_learning_curve(data_dict['model_initial'],X,y,name_str=name_str)

    # optimized learning curve
    param_str = '{}={}_{}={}'.format(data_dict['param_1'],data_dict['param_1_optimal'],data_dict['param_2'],data_dict['param_2_optimal'])
    name_str = '_'.join([model_dict['name'],data_dict['name'],'_'.join(data_dict['y']),param_str])
    model_learning_curve(data_dict['model_final'],X,y,name_str=name_str)
    
def nn_epochs(model_dict,data_source):
    #source: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
    
    data_dict = model_dict[data_source]
    model = data_dict['model_final']
    df = data_dict['data']
    label = data_dict['y']
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(label,axis=1),df[label],test_size=0.33,random_state=42)
    
    classes = np.unique(y_train)
    batch_count = 10
    batch_size = int(X_train.shape[0]/batch_count)
    epochs = 1000
    scores_train = []
    scores_test = []
    for epoch in range(epochs):
        print('epoch:{}'.format(epoch))
        model.partial_fit(X_train,y_train,classes=classes)
        data_perm = np.random.permutation(X_train.shape[0])
        index = 0
        count = 0
        while index < X_train.shape[0]:
            print('batch:{}'.format(count))
            indicies = data_perm[index:index+batch_size]
            model.partial_fit(X_train.iloc[indicies],y_train.iloc[indicies],classes=classes)
            index += batch_size
            count += 1
            if index > X_train.shape[0]: break
        
        scores_train.append(model.score(X_train,y_train))
        scores_test.append(model.score(X_test,y_test))
        
    plt.clf()
    plt.plot(scores_train, color='orange', alpha=1.0, label='Train')
    plt.plot(scores_test, color='green', alpha=1.0, label='Test')
    plt.title('Accuracy vs. epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='upper left')
    plt.savefig('plots2/optimized_mlp_{}.png'.format(data_dict['name']))
        
        
        
        
        
    


def run():

    # df1 = None
    df1 = get_df('balanced_wage')
    # print(df1.head())
    df1 = clean_df(df1,'wage-class')
    
    df2 = get_df('iris')
    # print(df2.head())
    df2 = clean_df(df2,'class')
    
    df3 = get_df('wine')
    # print(df3.head())
    df3 = clean_df(df3,'class')
    
    models = {
        'knn':{
            'name':'KNN',
            'data_1':{
                'data':df1,
                'y':['wage-class'],
                'name':'adult',
                'model_initial': neighbors.KNeighborsClassifier(),
                'param_1':'n_neighbors',
                'param_1_range':[1,5,20,50,80,100,150,200],
                'param_1_optimal':100,
                'model_working': neighbors.KNeighborsClassifier(n_neighbors=100),
                # 'param_2':'metric',
                # 'param_2_range':['minkowski','chebyshev',lambda x,y: sum(abs(x[0]-y[0])**13)**(1/13) ],
                # 'param_2_optimal':'distance',
                # 'model_final': neighbors.KNeighborsClassifier(n_neighbors=100,metric='chebyshev'),
                'param_2':'weights',
                'param_2_range':['uniform','distance',lambda x: [(i-np.min(x))/(np.max(x)-np.min(x)) for i in x]],
                'param_2_optimal':'uniform',
                'model_final': neighbors.KNeighborsClassifier(n_neighbors=100,weights='uniform'),
                # 'param_2':'p',
                # 'param_2_range':[x for x in range(1,13,1)],
                # 'param_2_optimal':1,
                # 'model_final': neighbors.KNeighborsClassifier(n_neighbors=100,p=13),
            },
            'data_2':{
                'data':df2,
                'y':['class'],
                'name':'iris',
                'model_initial': neighbors.KNeighborsClassifier(),
                'param_1':'n_neighbors',
                'param_1_range':[x for x in range(1,20)],
                'param_1_optimal':10,
                'model_working': neighbors.KNeighborsClassifier(n_neighbors=10),
                'param_2':'p',
                'param_2_range':[x for x in range(1,20)],
                'param_2_optimal':4,
                'model_final': neighbors.KNeighborsClassifier(n_neighbors=10,p=4),
            },
            'data_3':{
                'data':df3,
                'y':['class'],
                'name':'wine',
                # 'model_initial': neighbors.KNeighborsClassifier(),
                # 'param_1':'n_neighbors',
                # 'param_1_range':[x for x in range(1,20)],
                # 'param_1_optimal':12,
                # 'model_working': neighbors.KNeighborsClassifier(n_neighbors=12),
                # 'param_2':'p',
                # 'param_2_range':[x for x in range(1,20)],
                # 'param_2_optimal':4,
                # 'model_final': neighbors.KNeighborsClassifier(n_neighbors=12,p=4),
                'model_initial': neighbors.KNeighborsClassifier(),
                'param_1':'n_neighbors',
                'param_1_range':[x for x in range(1,100)],
                'param_1_optimal':12,
                'model_working': neighbors.KNeighborsClassifier(n_neighbors=12),
                # 'param_2':'weights',
                # 'param_2_range':['uniform','distance',lambda x: [(i-np.min(x))/(np.max(x)-np.min(x)) for i in x]],
                # 'param_2_optimal':'uniform',
                # 'model_final': neighbors.KNeighborsClassifier(n_neighbors=12,weights='uniform'),
                'param_2':'p',
                'param_2_range':[x for x in range(1,100)],
                'param_2_optimal':1,
                'model_final': neighbors.KNeighborsClassifier(n_neighbors=12,p=1),
            },
        },
        'dt':{
            'name':'DT',
            'data_1':{
                'data':df1,
                'y':['wage-class'],
                'name':'adult',
                'model_initial': tree.DecisionTreeClassifier(),
                'param_1':'max_depth',
                'param_1_range':[1,2,3,4,5,10,20,50,100],
                'param_1_optimal':3,
                'model_working': tree.DecisionTreeClassifier(max_depth=3),
                'param_2':'min_samples_leaf',
                'param_2_range':[x for x in range(1,5000,50)],
                'param_2_optimal':20,
                'model_final': tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=20),
            },
            'data_2':{
                'data':df2,
                'y':['class'],
                'name':'iris',
                'model_initial': tree.DecisionTreeClassifier(),
                'param_1':'max_depth',
                'param_1_range':[x for x in range(1,10)],
                'param_1_optimal':3,
                'model_working': tree.DecisionTreeClassifier(max_depth=3),
                'param_2':'min_samples_split',
                'param_2_range':[x for x in range(2,10)],
                'param_2_optimal':6,
                'model_final': tree.DecisionTreeClassifier(max_depth=3,min_samples_split=6),
            },
            'data_3':{
                'data':df3,
                'y':['class'],
                'name':'wine',
                # 'model_initial': tree.DecisionTreeClassifier(),
                # 'param_1':'max_depth',
                # 'param_1_range':[x for x in range(1,10)],
                # 'param_1_optimal':4,
                # 'model_working': tree.DecisionTreeClassifier(max_depth=4),
                # 'param_2':'min_samples_split',
                # 'param_2_range':[x for x in range(2,10)],
                # 'param_2_optimal':7,
                # 'model_final': tree.DecisionTreeClassifier(max_depth=4,min_samples_split=7),
                'model_initial': tree.DecisionTreeClassifier(),
                'param_1':'max_depth',
                'param_1_range':[x for x in range(1,10)],
                'param_1_optimal':4,
                'model_working': tree.DecisionTreeClassifier(max_depth=4),
                'param_2':'min_samples_split',
                'param_2_range':[x for x in range(2,100,2)],
                'param_2_optimal':7,
                'model_final': tree.DecisionTreeClassifier(max_depth=4,min_samples_split=7),
            },
        },
        'boost':{
            'name':'BOOST',
            'data_1':{
                'data':df1,
                'y':['wage-class'],
                'name':'adult',
                'model_initial': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=20)),
                'param_1':'n_estimators',
                'param_1_range':[2,3,4,5,6,7,8,9,10,20,50,70,100],
                'param_1_optimal':6,
                'model_working': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=20),n_estimators=6),
                'param_2':'learning_rate',
                'param_2_range':[.1,.2,.5,.7,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],
                'param_2_optimal':.5,
                'model_final': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=20),n_estimators=6,learning_rate=.5),
            },
            'data_2':{
                'data':df2,
                'y':['class'],
                'name':'iris',
                'model_initial': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1,min_samples_split=6)),
                'param_1':'n_estimators',
                'param_1_range':[x for x in range(1,50)],
                'param_1_optimal':10,
                'model_working': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1,min_samples_split=6),n_estimators=10),
                'param_2':'learning_rate',
                'param_2_range':[x for x in range(1,10)],
                'param_2_optimal':1.,
                'model_final': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1,min_samples_split=6),n_estimators=10,learning_rate=1.),
            },
            'data_3':{
                'data':df3,
                'y':['class'],
                'name':'wine',
                'model_initial': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=7)),
                'param_1':'n_estimators',
                'param_1_range':[x for x in range(1,200,2)],
                'param_1_optimal':10,
                'model_working': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=7),n_estimators=10),
                'param_2':'learning_rate',
                'param_2_range':[x for x in range(1,100)],
                'param_2_optimal':1.,
                'model_final': ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,min_samples_split=7),n_estimators=10,learning_rate=1.),
            },
        },
        'nn':{
            'name':'NN',
            'data_1':{
                'data':df1,
                'y':['wage-class'],
                'name':'adult',
                'model_initial': MLPClassifier(),
                'param_1':'hidden_layer_sizes',
                'param_1_range':[x for x in range(1,1051,50)],
                'param_1_optimal':1000,
                'model_working': MLPClassifier(hidden_layer_sizes=1000),
                'param_2':'alpha',
                'param_2_range':[.00001,.0001,.001,.01,.1,1.,2.],
                'param_2_optimal':.0001,
                'model_final': MLPClassifier(hidden_layer_sizes=1000,alpha=.0001),
            },
            'data_2':{
                'data':df2,
                'y':['class'],
                'name':'iris',
                'model_initial': MLPClassifier(max_iter=1000),
                'param_1':'hidden_layer_sizes',
                'param_1_range':[x for x in range(1,200,10)],
                'param_1_optimal':100,
                'model_working': MLPClassifier(max_iter=1000,hidden_layer_sizes=100),
                'param_2':'alpha',
                'param_2_range':[.00001,.0001,.001,.01,.1],
                'param_2_optimal':.0001,
                'model_final': MLPClassifier(max_iter=1000,hidden_layer_sizes=100,alpha=.0001),
            },
            'data_3':{
                'data':df3,
                'y':['class'],
                'name':'wine',
                # 'model_initial': MLPClassifier(max_iter=1000),
                # 'param_1':'hidden_layer_sizes',
                # 'param_1_range':[x for x in range(100,1000,100)],
                # 'param_1_optimal':300,
                # 'model_working': MLPClassifier(max_iter=1000,hidden_layer_sizes=300),
                # 'param_2':'alpha',
                # 'param_2_range':[.00001,.0001,.001,.01,.1],
                # 'param_2_optimal':.1,
                # 'model_final': MLPClassifier(max_iter=1000,hidden_layer_sizes=300,alpha=.1),
                'model_initial': MLPClassifier(max_iter=200),
                'param_1':'hidden_layer_sizes',
                'param_1_range':[x for x in range(100,5000,100)],
                'param_1_optimal':800,
                'model_working': MLPClassifier(max_iter=200,hidden_layer_sizes=800),
                'param_2':'alpha',
                'param_2_range':[.00001,.0001,.001,.01,.1,1.,1.5,2.,5.],
                'param_2_optimal':.1,
                'model_final': MLPClassifier(max_iter=200,hidden_layer_sizes=800,alpha=.1),
            },
        },
        'svc':{
            'name':'SVC',
            'data_1':{
                'data':df1,
                'y':['wage-class'],
                'name':'adult',
                'model_initial': svm.SVC(),
                'param_1':'kernel',
                'param_1_range':['rbf','linear','sigmoid'],
                'param_1_optimal':'linear',
                'model_working': svm.SVC(kernel='linear'),
                'param_2':'C',
                'param_2_range':[0.5,1.,1.5],
                'param_2_optimal':1.,
                'model_final': svm.SVC(kernel='linear',C=1.),
            },
            'data_2':{
                'data':df2,
                'y':['class'],
                'name':'iris',
                'model_initial': svm.SVC(),
                'param_1':'kernel',
                'param_1_range':['rbf','linear','sigmoid'],
                'param_1_optimal':'linear',
                'model_working': svm.SVC(kernel='linear'),
                'param_2':'C',
                'param_2_range':[0.5,1.,1.5],
                'param_2_optimal':1.,
                'model_final': svm.SVC(kernel='linear',C=1.),
            },
            'data_3':{
                'data':df3,
                'y':['class'],
                'name':'wine',
                'model_initial': svm.SVC(),
                'param_1':'kernel',
                'param_1_range':['rbf','linear','sigmoid'],
                # 'param_1_optimal':'linear',
                # 'model_working': svm.SVC(kernel='linear'),
                # 'param_2':'C',
                # 'param_2_range':[0.1,0.5,1.,2.,5.,10.],
                # 'param_2_optimal':1.,
                # 'model_final': svm.SVC(kernel='linear',C=1.),
                'param_1_optimal':'rbf',
                'model_working': svm.SVC(kernel='rbf'),
                'param_2':'gamma',
                'param_2_range':[1/x for x in [178,250,300,600,1000,2000,10000,20000,40000]],
                'param_2_optimal':1.,
                'model_final': svm.SVC(kernel='rbf',gamma=1/10000),
            },
        },
        
    }
    
    
    # run_model(models['knn'],'data_1')
    # run_model(models['dt'],'data_1')
    # run_model(models['boost'],'data_1')
    # run_model(models['nn'],'data_1')
    # nn_epochs(models['nn'],'data_1')
    # nn_epochs(models['nn'],'data_3')
    # run_model(models['nn'],'data_3')
    
    
    # run_model(models['knn'],'data_3')
    # run_model(models['knn'],'data_2')
    
    
    # run_model(models['dt'],'data_3')
    # run_model(models['dt'],'data_2')
    
    
    # run_model(models['boost'],'data_3')
    # run_model(models['boost'],'data_2')
    
    
    # run_model(models['nn'],'data_3')
    # run_model(models['nn'],'data_2')
    
    # run_model(models['svc'],'data_1')
    # run_model(models['svc'],'data_3')
    # run_model(models['svc'],'data_2')
    
    
    
    #run these
    
    run_model(models['dt'],'data_1')
    run_model(models['dt'],'data_3')
    
    run_model(models['boost'],'data_1')
    run_model(models['boost'],'data_3')
    
    run_model(models['knn'],'data_1')
    run_model(models['knn'],'data_3')
    
    run_model(models['nn'],'data_1')
    run_model(models['nn'],'data_3')
    
    nn_epochs(models['nn'],'data_1')
    nn_epochs(models['nn'],'data_3')
    
    run_model(models['svc'],'data_1')
    run_model(models['svc'],'data_3')
    
    
    
    

if __name__ == "__main__":
    run()
