import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
from matplotlib import pyplot as plt
import configparser as cp
import os

def check_for_empty_values(df):
    print("Dropping rows with empty values \n")
    df.dropna(inplace = True)

def check_for_duplicates(df):
    print("Dropping duplicates \n")
    df.drop_duplicates(inplace = True)

def get_features_and_count(features, df):
    '''
    This function takes a dataframe and a features list, 
    creates a dictionary where keys are features and values 
    are the sum of the features' positive values.
    features: list of str objects
    df: DataFrame 
    returns OrderedDict.
    '''
    d = dict.fromkeys(features, 0)
    for g in features:
        d[g] = sum(df[g].tolist())
    d = OrderedDict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
    return d

def drop_genre_by_count(gc_dict, df, t_hold, inpl = False):
    '''
    Removes underrepresented genres from 
    the dataframe (less than the threshold).
    gc_dict: a dictionary of genre frequency, {str:int}
    df: DataFrame 
    t_hold: threshold for the frequency of a genre, int
    inpl: boolean for an attribute "inplace", bool
    returns DataFrame
    '''
    if inpl == True:
        df.drop([g for g in gc_dict if gc_dict[g] < t_hold], axis=1, inplace=True)
    else:
        return df.drop([g for g in gc_dict if gc_dict[g] < t_hold], axis=1, inplace=False)


def check_for_nonsensical_columns(df):
    features = list(set(df.columns) - {"Id", "Genre",})
    g_and_count = get_features_and_count(features,df)
    print("Dropping nonsensical columns \n")
    drop_genre_by_count(g_and_count, df, 10, True)

def check_for_ms_wt_genres(df):
    features = list(set(df.columns) - {"Id", "Genre",})
    null_genres_ix = [i for i in range(len(df)) if sum(np.array(df[features].iloc[[i]])[0]) == 0]
    print("Dropping rows without genres \n")
    df.drop(df.index[null_genres_ix], inplace = True)

def count_of_ms_by_genre(df):
    features = list(set(df.columns) - {"Id", "Genre",})
    g_and_count = get_features_and_count(features,df)
    print("Plotting \n")

    plt.figure(figsize=(22, 12))
    pd = sns.barplot(list(g_and_count.keys()), list(g_and_count.values()), 
            palette=("muted"))
    
    pd.set_title("Genre Distribution(Bar)", fontsize=18)
    plt.savefig('../visualisation/genre_distr_bar.png')

    plt.figure(figsize=(20, 12))
    plt.pie(list(g_and_count.values()), labels=list(g_and_count.keys()), autopct='%1.1f%%',
        shadow=True, startangle=90)

    plt.title('Genre Distribution(Pie)', fontsize=15)
    plt.savefig('../visualisation/genre_distr_pie.png')

def save_train(df):
    i = 0
    while os.path.exists("../data/preproc_res/prep_train%s.csv" % i):
        i += 1
    df.to_csv("../data/preproc_res/prep_train%s.csv" % i, sep='\t')
    print("Dataframe saved like: {}".format("../data/preproc_res/prep_train%s.csv" % i))

def start(df):
    cfg = cp.ConfigParser()  
    cfg.read("pipeline_prep.ini")
    prep_func = eval(cfg["P1"]["FUNC_LIST"]) 
    
    print('''Welcome to the data preprocessing section!  
        
        The columns of your dataframe:\n{}
        
        The data shape:\n{}'''.format(str(list(df.columns)), np.shape(df)))
    
    print("List of prep functions:\n") 
    for i in range(len(prep_func)):
        print("№: {} \t||\t func: {} \n".format(i,prep_func[i]))
    
    print("Prep begins...\n")
    try:    
        for i in range(len(prep_func)):
            prep_func[i](df)
    except Exception as e:
        print(e)
    
    print("Do you want to save the result? (y/N)")
    while True:
        try:
            ts = input("Answer: ")
            if ts == 'y':
                save_train(df)
                break
            elif ts == 'N':
                break
            else:
                raise ValueError
        except ValueError:
            print("Input error, try again")
    
    return df
