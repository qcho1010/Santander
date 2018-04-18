
# coding: utf-8

import pandas as pd
import numpy as np


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

# these features have more than one value per unique user over time, can be used for lagged information
cust_feature = ['indrel', 'indrel_1mes', 'tiprel_1mes', 'segmento', 'ind_actividad_cliente']


dtypes = {
    "ncodpers": int,
    "conyuemp": str,
    "indrel" : str,
    "ind_actividad_cliente" : str
}

train = pd.read_csv('data/train_clean2.csv', dtype=dtypes, parse_dates=True)
train_X = pd.read_csv('data/train_X.csv')
train_Y = pd.read_csv('data/train_Y.csv')


def make_lag_test(n_in, df, feature):
    tmp = pd.DataFrame()
    for i in range(n_in, 0, -1):
        name = feature+"-"+str(i-1)
        tmp[name] = df.shift(i-1)
    tmp.dropna(axis=0, inplace=True)
    return tmp

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

# flag change of customer features from previous month such as 
def cust_change_flag(row): 
    if row[-1] == row[-2]: return 0
    elif row[-2] != row[-1]: return 1


test = pd.DataFrame()

group = train.groupby(['ncodpers'])
for name, g in group:
    test = pd.concat([test, g.iloc[-5:,:]], axis=0)


# ## Building Test X

lag = 5

test_X = pd.DataFrame()
test_Y = pd.DataFrame()

test_id = test['ncodpers'].unique()
for i, id in enumerate(test_id):
    if i % 200 == 0: print round(float(i)/float(len(test_id)) * 100, 2)
    test_df = pd.DataFrame()
    Y = pd.DataFrame()
    for prod in products:
        
        # build lagged features
        X1 = make_lag_test(lag, test.loc[test['ncodpers'] == id, prod], prod)
        test_df = pd.concat([test_df, X1], axis=1)
        
        # feature engineering with lagged data
        list1 = []; list2 = []; list3 = []; list4 = []; list5 = []; code_list = []
        for row in X1.values:
            list1.append(prod_change_flag(row))
            list2.append(prod_change_ratio(row))
            list3.append(prod_own_len(row))
            list4.append(prod_not_own_len(row))
            list5.append(prod_own_ratio(row))
            code_list.append(id)
            
        test_df[str(prod+'_'+'prod_change_flag')] = list1
        test_df[str(prod+'_'+'prod_change_ratio')] = list2
        test_df[str(prod+'_'+'prod_own_len')] = list3
        test_df[str(prod+'_'+'prod_not_own_len')] = list4
        test_df[str(prod+'_'+'prod_own_ratio')] = list5
        test_df['ncodpers'] = code_list
        
    for cust in cust_feature:
        X1 = make_lag_test(lag, test.loc[test['ncodpers'] == id, cust], cust)
        list1 = []
        for row in X1.values:
            list1.append(cust_change_flag(row))
        test_df[str(cust+'_'+'cust_change_flag')] = list1
        
    month = make_lag_test(lag, test.loc[test['ncodpers'] == id, 'month'], 'month')
    test_df = pd.concat([test_df, month], axis=1)
    test_X = pd.concat([test_X, test_df], axis=0)
    
test_X.to_csv('data/test_X.csv', index=False)


test_X.shape


# # Modeling

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(train_X, train_Y);
print("Predicting..")
preds = np.array(model.predict_proba(test_X))[:,:,1].T
del test_X

print("Getting last instance dict..")
last_instance_df = last_instance_df.fillna(0).astype('int')
cust_dict = {}
products = np.array(products)
for ind, row in last_instance_df.iterrows():
    cust = row['ncodpers']
    used_products = set(products[np.array(row[1:])==1])
    cust_dict[cust] = used_products
del last_instance_df

print("Creating submission..")
preds = np.argsort(preds, axis=1)
preds = np.fliplr(preds)
test_id = np.array(pd.read_csv(test_file, usecols=['ncodpers'])['ncodpers'])
final_preds = []
for ind, pred in enumerate(preds):
    cust = test_id[ind]
    top_products = target_cols[pred]
    used_products = cust_dict.get(cust,[])
    new_top_products = []
    for product in top_products:
        if product not in used_products:
            new_top_products.append(product)
        if len(new_top_products) == 7:
            break
    final_preds.append(" ".join(new_top_products))
out_df = pd.DataFrame({'ncodpers':test_id, 'added_products':final_preds})
out_df.to_csv('sub_rf.csv', index=False)

