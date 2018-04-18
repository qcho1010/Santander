
# coding: utf-8

# # Preprocessing

import pandas as pd
import numpy as np


data = pd.read_csv('data/train_ver2.csv')


# ## Cleaning

from qlearn.util import *
float_feature, int_feature, obj_feature, etc_feature = dftypes(data, show=True)
obj_col, obj_col2 = get_freq_table(data, level=15)


def clean_data(fi, fo, header, suffix):
    head = fi.readline().strip("\n").split(",")
    head = [h.strip('"') for h in head]
    for i, h in enumerate(head):
        if h == "nomprov":
            ip = i
    print(ip)
    n = len(head)
    if header:
        fo.write("%s\n" % ",".join(head))

    print(n)
    for line in fi:
        fields = line.strip("\n").split(",")
        if len(fields) > n:
            prov = fields[ip] + fields[ip+1]
            del fields[ip]
            fields[ip] = prov
        assert len(fields) == n
        fields = [field.strip() for field in fields]
        fo.write("%s%s\n" % (",".join(fields), suffix))

with open("data/train_clean.csv", "w") as f:
    clean_data(open("data/train_ver2.csv"), f, True, "")


products = (
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1" ,
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1" ,
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1" ,
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1" ,
    "ind_nomina_ult1"  ,
    "ind_nom_pens_ult1",
    "ind_recibo_ult1"  ,
)

dtypes = {
    "ncodpers": int,
    "conyuemp": str,
}


train = pd.read_csv('data/train_clean.csv', dtype=dtypes, parse_dates=True)


from qlearn.util import *
float_feature, int_feature, obj_feature, etc_feature = dftypes(train, show=True)
obj_col, obj_col2 = get_freq_table(train, level=15)


drop_feature = ['ult_fec_cli_1t', 'renta', 'conyuemp', 'tipodom', 'cod_prov', 'fecha_alta', 
               'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1',
               'ind_hip_fin_ult1', 'ind_pres_fin_ult1', 'ind_cder_fin_ult1']
train.drop(drop_feature, axis=1, inplace=True)

date_feature = ['fecha_dato', 'fecha_alta']
num_feature = ['age', 'antiguedad']
obj_feature = ['ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'indrel', 
               'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'canal_entrada',
              'indfall', 'segmento', 'ind_actividad_cliente', 'nomprov', 'cod_prov']
label_feature = ['pais_residencia', 'canal_entrada', 'nomprov', 'cod_prov']
one_hot_feature = [x for x in obj_feature if x not in label_feature]


# ## Exploration

size = 1000 # number of unique customer id
sample_id = list(data['ncodpers'].unique()[:size])


# checking number of minimum account profile in the database per unique ID
minimum = 20
for i, code in enumerate(sample_id):
    incident_count = len(data.loc[data['ncodpers'] == code])
    if incident_count < minimum:
        minimum = incident_count
        print minimum


# checking the variability of the categorical feature over time per unique ID
for feat in obj_feature:
    print feat
    maximum = 0
    for i, code in enumerate(sample_id):
        incident_count = len(train.loc[train['ncodpers'] == code, feat].value_counts())
        if incident_count > maximum:
            maximum = incident_count
            print maximum


# these features have more than one value per unique user over time, can be used for lagged information
# 'indrel', 'indrel_1mes', 'tiprel_1mes', 'segmento', 'ind_actividad_cliente', 'nomprov', 'cod_prov'


# ## Cleaning

# cleaning indrel_1mes
train["indrel_1mes"] = train["indrel_1mes"].map(lambda x: 5.0 if x == "P" else x).astype(float).fillna(0.0).astype(np.int8)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (10, 6)


# cleaning age
train.loc[train.age < 18,"age"]  = train.loc[(train.age >= 18) & (train.age <= 30),"age"].median(skipna=True)
train.loc[train.age > 100,"age"] = train.loc[(train.age >= 30) & (train.age <= 100),"age"].median(skipna=True)
train["age"] = train["age"].fillna(train["age"].median()).astype(np.int16)


with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(train["age"].dropna(),
                 bins=80,
                 kde=False,
                 color="tomato")
    plt.title("Age Distribution")
    plt.ylabel("Count")


# cleaning antiguedad
train['antiguedad'] = train['antiguedad'].replace([np.nan, -999999.0], 0, regex=True)


# replacing NaN with string
for feat in one_hot_feature:
    train[feat] = train[feat].astype('str')
    train[feat] = train[feat].replace([np.nan, 'nan'], 'undefined')
    print train[feat].value_counts(dropna=True)
    print "="*20


# cleaning segmento
seg_map = {'02 - PARTICULARES' : 'PARTICULARES',
           '03 - UNIVERSITARIO' : 'UNIVERSITARIO',
           '01 - TOP' : 'TOP'}

train.replace({'segmento' : seg_map}, inplace=True)

train['segmento'].value_counts()


# replacing NaN with string
for feat in label_feature:
    train[feat] = train[feat].astype('str')
    train[feat] = train[feat].replace([np.nan, 'nan'], 'undefined')
    print train[feat].value_counts(dropna=True)
    print "="*20


train['ind_nomina_ult1'].value_counts(dropna=False)


train["ind_nomina_ult1"] = train["ind_nomina_ult1"].fillna(0.0).astype(np.int16)
train["ind_nom_pens_ult1"] = train["ind_nom_pens_ult1"].fillna(0.0).astype(np.int16)


# from scipy.stats.mstats import mode
# gg = train.groupby('ncodpers')['fecha_alta'].transform(lambda x: mode(x).mode[0] if x is None else x)


train.isnull().any()


train = train.sort_values(by=['ncodpers', 'fecha_dato'])
train.to_csv('data/train_clean2.csv', index=False)


for feat in products:
    print train[feat].value_counts()
    print "="*20

