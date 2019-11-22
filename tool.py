import pandas as pd
import numpy as np

def make_wage_data():
    def balance_y(df,y,count_limit=5000):
        count = df.groupby(y).count().min()[0]
        print(df.groupby(y).count())
        
        if count > count_limit:
            count = count_limit
        
        indicies = []
        vals = df[y].unique()
        for v in vals:
            indicies.extend(df[df[y] == v].index[:count])
        
        new_df = df.loc[indicies]
        new_df.index = range(new_df.shape[0])
        
        new_df = new_df.sample(frac=1).reset_index(drop=True)
        
        
        return new_df
        
    
    
    def get_data(path, sep=',',columns=None):
        df = pd.read_csv(path,sep=sep)
        if columns is not None:
            df.columns = columns
        return df
    
    
    path = 'data/people_data/adult.data'
    columns = ['age','workclass','fnlwgt','education','education-num',
                'marital-status','occupation','relationship','race','sex',
                'capital-gain','capital-loss','hours-per-week','native-country',
                'wage-class']
    sep = ','
    df = get_data(path,sep,columns)
    
    df = balance_y(df=df,y='wage-class',count_limit=2500)
    
    # print(df.groupby('wage-class').count().min()[0])
        
    # sample_1 = df[df['wage-class'] == df['wage-class'].unique()[0]].index
    # sample_2 = df[df['wage-class'] == df['wage-class'].unique()[1]].index
    # # print(sample_1,sample_2)
    # count = 5000
    # # choice_1 = np.random.choice(sample_1, 1000, replace=False)
    # # choice_2 = np.random.choice(sample_2, 1000, replace=False)
    # indicies = np.concatenate([sample_1[:count],sample_2[:count]],axis=None)
    # df = df.loc[indicies]
    # df.index = range(df.shape[0])
    
    # df.to_csv('data/wage_balanced_data/balanced_wage.csv')
    
    return df
    
    


if __name__ == "__main__":
    df = make_wage_data()
    print(df.shape)
    print(df.head())
    pass