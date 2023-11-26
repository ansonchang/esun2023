import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import *

print("Start data preprocessing...")

training_df=pd.read_csv('../data/dataset_1st/training.csv')
eval_df=pd.read_csv('../data/dataset_2nd/public.csv')
test_df=pd.read_csv('../data/dataset_2nd/private_1_processed.csv')
submit_sample_df=pd.read_csv('../data/31_範例繳交檔案.csv')

print("============== eval_df,test_df,training_df shape:")
print(eval_df.shape,test_df.shape,training_df.shape)

all_df=pd.concat([training_df,eval_df,test_df],axis=0)
all_df2=all_df.copy()

print("============== calculate cano_agg:")
cano_agg = all_df.groupby(['chid','cano']).agg({
    'conam': ['max','mean','std'],
}).reset_index()
cano_agg.columns = ['_'.join([str(y) for y in x if y]) for x in cano_agg.columns]
all_df = pd.merge(all_df,cano_agg,how='left')

print("============== calculate contp_agg:")
contp_agg = all_df.groupby(['chid','cano','contp']).agg({
    'conam': ['max','mean','std'],
}).reset_index()
contp_agg.columns = ['_'.join([str(y) for y in x if y]) for x in contp_agg.columns]
contp_agg.columns=['chid','cano','contp','contp_max','contp_mean','contp_std']
all_df = pd.merge(all_df,contp_agg,how='left')

print("============== calculate mchno_agg:")
mchno_agg = all_df.groupby(['chid','cano','mchno']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
mchno_agg.columns = ['_'.join([str(y) for y in x if y]) for x in mchno_agg.columns]
mchno_agg.columns=['chid','cano','mchno','mchno_conam_max','mchno_conam_mean','mchno_conam_stb','mchno_conam_cnt']
all_df = pd.merge(all_df,mchno_agg,how='left')

print("============== calculate is_taiwan/is_ntd:")
all_df['is_taiwan']=np.where(all_df.stocn==0,1,0) 
all_df['is_ntd']=np.where(all_df.csmcu==70,1,0) 

print("============== calculate twn_agg:")
twn_agg = all_df.groupby(['chid','cano','is_taiwan']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
twn_agg.columns = ['_'.join([str(y) for y in x if y]) for x in twn_agg.columns]
twn_agg.columns=['chid','cano','is_taiwan','twn_conam_max','twn_conam_mean','twn_conam_std','twn_conam_count']
all_df = pd.merge(all_df,twn_agg,how='left')

print("============== calculate ntd_agg:")
ntd_agg = all_df.groupby(['chid','cano','is_ntd']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
ntd_agg.columns = ['_'.join([str(y) for y in x if y]) for x in ntd_agg.columns]
ntd_agg.columns=['chid','cano','is_ntd','ntd_conam_max','ntd_conam_mean','ntd_conam_std','ntd_conam_count']
all_df = pd.merge(all_df,ntd_agg,how='left')

print("============== calculate hcefg_agg:")
hcefg_agg = all_df.groupby(['chid','cano','hcefg']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
hcefg_agg.columns = ['_'.join([str(y) for y in x if y]) for x in hcefg_agg.columns]
ntd_agg.columns=['chid','cano','hcefg','hcefg_conam_max','hcefg_conam_mean','hcefg_conam_std','hcefg_conam_count']
all_df = pd.merge(all_df,hcefg_agg,how='left')

print("============== calculate cano_dt_agg:")
cano_dt_agg = all_df.groupby(['chid','cano']).agg({
    'locdt': ['max','min','std','count'],
}).reset_index()
cano_dt_agg.columns = ['_'.join([str(y) for y in x if y]) for x in cano_dt_agg.columns]
cano_dt_agg.columns=['chid','cano','cano_dt_max','cano_dt_min','cano_dt_std','cano_dt_cnt']
all_df = pd.merge(all_df,cano_dt_agg,how='left')

print("============== calculate dif_day_maxdate/dif_day_mindate:")
all_df['dif_day_maxdate']=all_df['cano_dt_max']-all_df['locdt']
all_df['dif_day_mindate']=all_df['locdt']-all_df['cano_dt_min']

print("============== calculate date features:")
all_df['locmonth']=all_df.locdt%30
all_df['locweekday']=all_df.locdt%7
all_df['locymonth']=all_df.locdt%12
all_df['locquarter']=all_df.locmonth%4
all_df['lochr']=all_df.loctm/10000
all_df['lochr']=all_df['lochr'].astype(int)

print("============== save csv:")
all_df.to_csv('../csv/all_df.csv', index=False)

print("Finish data preprocessing...")
print("Generate datasets in csv direcotory")

