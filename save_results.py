
from datetime import datetime, timedelta
import sklearn
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
import sklearn.metrics
import os 
import pickle


boston = sklearn.datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target

X = df[df.columns[:-1]]
Y = df['MEDV']

result_dir = 'results'
cv = 10

######--------------- Zapis wszystkich wyników cross-validacji do dataframe ----
results = pd.DataFrame()
scorer = sklearn.metrics.make_scorer(sklearn.metrics.mean_squared_error)


for depth in range(3, 7):
    x=cross_validate(DecisionTreeRegressor(max_depth=depth), X,Y, scoring=scorer, 
                             cv=cv)
    tmp = pd.DataFrame({'max_depth':[depth]*cv,'mse':x['test_score']})
    results = results.append(tmp)
    

time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
name = 'DT_allvars_' + time_string
print(name)

results.to_csv("{0}/{1}.csv".format(result_dir,name),index=False,sep=';')


######--------------- Dla różnych kombinacji zmiennych zmiennych  ----

###---------- zapis do różnych folderów ----------

def check_dir(name,path):
    try:
        os.mkdir("{0}/{1}".format(path,name))
    except:
        print('dir exist')

var_numbers = [8,10,13]

for n in var_numbers:
    check_dir(n,result_dir)
    X = df[df.columns[:n]]
    results = pd.DataFrame()
    
    for depth in range(3, 7):
        x=cross_validate(DecisionTreeRegressor(max_depth=depth), X,Y, scoring=scorer, 
                                 cv=cv)
        tmp = pd.DataFrame({'max_depth':[depth]*cv,'mse':x['test_score']})
        results = results.append(tmp)
    
    time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
    name = 'DT_' + time_string
    print(name)
    
    results.to_csv("{0}/{1}/{2}.csv".format(result_dir,n,name),index=False,sep=';')
    
    
###---------- zapis listy zmiennych jako string ----------

var_numbers = [8,10,13]
results = pd.DataFrame()
    

for n in var_numbers:
    X = df[df.columns[:n]]
    tmp2 = pd.DataFrame()
    
    for depth in range(3, 7):
        x=cross_validate(DecisionTreeRegressor(max_depth=depth), X,Y, scoring=scorer, 
                                 cv=cv)
        tmp = pd.DataFrame({'max_depth':[depth]*cv,'mse':x['test_score']})
        tmp2 = tmp2.append(tmp)
    
    var = ','.join(list(df.columns[:n]))
    tmp2['variables'] = [var for i in range(len(tmp2))]
    results = results.append(tmp2)
    
results.groupby(['variables']).mean()['mse']
    
time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
name = 'DT_{0}'.format(time_string)
print(name)

results.to_csv("{0}/{1}.csv".format(result_dir,name),index=False,sep=';')

###------------ zapis w bardziej skomplikowane struktury i picklowanie ---


var_numbers = [8,10,13]
result_list = []

for n in var_numbers:
    check_dir(n,result_dir)
    X = df[df.columns[:n]]
    results_dict = {}
    results = pd.DataFrame()
    
    for depth in range(3, 7):
        x=cross_validate(DecisionTreeRegressor(max_depth=depth), X,Y, scoring=scorer, 
                                 cv=cv)
        tmp = pd.DataFrame({'max_depth':[depth]*cv,'mse':x['test_score']})
        results = results.append(tmp)
    
    results_dict['variables'] = list(df.columns[:n])
    results_dict['varnum'] = len(df.columns[:n])
    results_dict['results_tab'] = results
    result_list.append(results_dict)



time_string = datetime.now().strftime('%Y%m%d_%H%M%S')
name = 'DT_{0}'.format(time_string)
print(name)
    
###------- zapis do pikla 
path = '{0}/{1}.pickle'.format(result_dir,name)
with open(path, 'wb') as file:
    pickle.dump(result_list, file)


###------- odczyt z pikla 

with open(path, 'rb') as file:
    loaded_pickle = pickle.load(file)



