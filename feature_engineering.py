
# coding: utf-8

# # Feature Engineering

import pandas as pd


dtypes = {
    "ncodpers": int,
    "conyuemp": str,
    "indrel" : str,
    "ind_actividad_cliente" : str
}

products = (
#     "ind_ahor_fin_ult1",
#     "ind_aval_fin_ult1",
    "ind_cco_fin_ult1" ,
#     "ind_cder_fin_ult1",
    "ind_cno_fin_ult1" ,
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
#     "ind_deco_fin_ult1",
#     "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
#     "ind_hip_fin_ult1" ,
    "ind_plan_fin_ult1",
#     "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1" ,
    "ind_nomina_ult1"  ,
    "ind_nom_pens_ult1",
    "ind_recibo_ult1"  ,
)

drop_feature = ['ult_fec_cli_1t', 'renta', 'conyuemp', 'tipodom', 'cod_prov', 'fecha_alta', 
               'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
               'ind_hip_fin_ult1', 'ind_pres_fin_ult1', 'ind_cder_fin_ult1']


date_feature = ['fecha_dato', 'fecha_alta']
num_feature = ['age', 'antiguedad']
obj_feature = ['ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'indrel', 
               'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'canal_entrada',
              'indfall', 'segmento', 'ind_actividad_cliente', 'nomprov']
label_feature = ['pais_residencia', 'canal_entrada', 'nomprov']
one_hot_feature = [x for x in obj_feature if x not in label_feature]


train = pd.read_csv('data/train_clean2.csv', dtype=dtypes, parse_dates=True)


# ## Lag function

def make_lag(n_in, df, feature):
    tmp = pd.DataFrame()
    for i in range(n_in+1, 0, -1):
        name = feature+"-"+str(i-1)
        tmp[name] = df.shift(i-1)
    tmp.dropna(axis=0, inplace=True)
    return tmp.iloc[:,:-1], tmp.iloc[:,-1]

def make_lag_test(n_in, df, feature):
    tmp = pd.DataFrame()
    for i in range(n_in, 0, -1):
        name = feature+"-"+str(i-1)
        tmp[name] = df.shift(i-1)
    tmp.dropna(axis=0, inplace=True)
    return tmp


# ## Products

# flag if the product was added, dropped or unchanged from previous month (switching from 0 to 1, 1 to 0, 0 to 0, 1 to 1)
def prod_change_flag(row): 
    if row[-1] == row[-2]: return 0
    elif row[-2] == 1 and row[-1] == 0: return -1
    elif row[-2] == 0 and row[-1] == 1: return 1
    
# ratio of months that customer changed the product X
def prod_change_ratio(row):
    return round(1 - float(np.where(np.max(row) == row)[0].tolist()[-1]+1)/float(len(row)), 2)

# consecutive months that customer had product X
def prod_own_len(row):
    index = np.where(np.max(row) == row)[0].tolist()
    return index[-1] - index[0] + 1

# consecutive months that customer not had product X
def prod_not_own_len(row):
    index = np.where(np.min(row) == row)[0].tolist()
    return index[-1] - index[0] + 1

# ratio of months that customer had product X 
def prod_own_ratio(row):
    return np.mean(row)


# ## Customers

# these features have more than one value per unique user over time, can be used for lagged information
cust_feature = ['indrel', 'indrel_1mes', 'tiprel_1mes', 'segmento', 'ind_actividad_cliente']


# flag change of customer features from previous month such as 
def cust_change_flag(row): 
    if row[-1] == row[-2]: return 0
    elif row[-2] != row[-1]: return 1


# ## Date

# get month
from datetime import datetime
train['month'] = train['fecha_dato'].apply(lambda x: x.month)


# ## Get Sample

sample = train.iloc[:40005, :]
sample_id = sample['ncodpers'].unique().tolist()
print len(sample_id)
sample.tail(10)


# ## Building Train X and Train Y

train_X = pd.DataFrame()
train_Y = pd.DataFrame()

train_id = train['ncodpers'].unique()
for i, id in enumerate(train_id):
    if i % 200 == 0: print round(float(i)/float(len(train_id)) * 100, 2)
    train_df = pd.DataFrame()
    Y = pd.DataFrame()
    for prod in products:
        
        # build lagged features
        X1, Y1 = make_lag(lag, train.loc[train['ncodpers'] == id, prod], prod)
        train_df = pd.concat([train_df, X1], axis=1)
        Y = pd.concat([Y, Y1], axis=1)
        
        # feature engineering with lagged data
        list1 = []; list2 = []; list3 = []; list4 = []; list5 = []; code_list = []
        for row in X1.values:
            list1.append(prod_change_flag(row))
            list2.append(prod_change_ratio(row))
            list3.append(prod_own_len(row))
            list4.append(prod_not_own_len(row))
            list5.append(prod_own_ratio(row))
            code_list.append(id)
            
        train_df[str(prod+'_'+'prod_change_flag')] = list1
        train_df[str(prod+'_'+'prod_change_ratio')] = list2
        train_df[str(prod+'_'+'prod_own_len')] = list3
        train_df[str(prod+'_'+'prod_not_own_len')] = list4
        train_df[str(prod+'_'+'prod_own_ratio')] = list5
        train_df['ncodpers'] = code_list
        
    for cust in cust_feature:
        X1, __ = make_lag(lag, train.loc[train['ncodpers'] == id, cust], cust)
        list1 = []
        for row in X1.values:
            list1.append(cust_change_flag(row))
        train_df[str(cust+'_'+'cust_change_flag')] = list1
        
    month, __ = make_lag(lag, train.loc[train['ncodpers'] == id, 'month'], 'month')
    train_df = pd.concat([train_df, month], axis=1)
    train_X = pd.concat([train_X, train_df], axis=0)
    train_Y = pd.concat([train_Y, Y], axis=0)
    
train_X.to_csv('data/train_X.csv', index=False)
train_Y.to_csv('data/train_Y.csv', index=False)


train_X.shape


len(train_Y)


# test dataset function
# string cacate
# label encoding
# concat all the dataset
# modeling
# submission function
# interpretation
# report


# ## Build Test X

test = pd.read_csv('data/test_ver2.csv', dtype=dtypes, parse_dates=True)


test_id = test['ncodpers'].unique()


len(test_id)


len(train['ncodpers'].unique())

