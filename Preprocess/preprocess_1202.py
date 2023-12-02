import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import *

training_df=pd.read_csv('../data/dataset_1st/training.csv')
# 56,57,58,59
eval1_df=pd.read_csv('../data/dataset_2nd/public.csv')
# 60,61,62,,63,64
eval2_df=pd.read_csv('../data/dataset_2nd/private_1.csv')
# 65,66,67,68,69
test_df=pd.read_csv('../data/dataset_3rd/private_2_processed.csv')

submit_sample_df=pd.read_csv('../data/dataset_3rd/private_2_template_v2.csv')


all_df=pd.concat([training_df,eval1_df,eval2_df,test_df],axis=0)

cano_agg = all_df.groupby(['chid','cano']).agg({
    'conam': ['max','mean','std'],
}).reset_index()
cano_agg.columns = ['_'.join([str(y) for y in x if y]) for x in cano_agg.columns]
all_df = pd.merge(all_df,cano_agg,how='left')

contp_agg = all_df.groupby(['chid','cano','contp']).agg({
    'conam': ['max','mean','std'],
}).reset_index()
contp_agg.columns = ['_'.join([str(y) for y in x if y]) for x in contp_agg.columns]
contp_agg.columns=['chid','cano','contp','contp_max','contp_mean','contp_std']
all_df = pd.merge(all_df,contp_agg,how='left')

mchno_agg = all_df.groupby(['chid','cano','mchno']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
mchno_agg.columns = ['_'.join([str(y) for y in x if y]) for x in mchno_agg.columns]
mchno_agg.columns=['chid','cano','mchno','mchno_conam_max','mchno_conam_mean','mchno_conam_stb','mchno_conam_cnt']
all_df = pd.merge(all_df,mchno_agg,how='left')

all_df['is_taiwan']=np.where(all_df.stocn==0,1,0) 
all_df['is_ntd']=np.where(all_df.csmcu==70,1,0) 

twn_agg = all_df.groupby(['chid','cano','is_taiwan']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
twn_agg.columns = ['_'.join([str(y) for y in x if y]) for x in twn_agg.columns]
twn_agg.columns=['chid','cano','is_taiwan','twn_conam_max','twn_conam_mean','twn_conam_std','twn_conam_count']
all_df = pd.merge(all_df,twn_agg,how='left')

ntd_agg = all_df.groupby(['chid','cano','is_ntd']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
ntd_agg.columns = ['_'.join([str(y) for y in x if y]) for x in ntd_agg.columns]
ntd_agg.columns=['chid','cano','is_ntd','ntd_conam_max','ntd_conam_mean',
                'ntd_conam_std','ntd_conam_count']
all_df = pd.merge(all_df,ntd_agg,how='left')

hcefg_agg = all_df.groupby(['chid','cano','hcefg']).agg({
    'conam': ['max','mean','std','count'],
}).reset_index()
hcefg_agg.columns = ['_'.join([str(y) for y in x if y]) for x in hcefg_agg.columns]
ntd_agg.columns=['chid','cano','hcefg','hcefg_conam_max','hcefg_conam_mean',
                'hcefg_conam_std','hcefg_conam_count']
all_df = pd.merge(all_df,hcefg_agg,how='left')

cano_dt_agg = all_df.groupby(['chid','cano']).agg({
    'locdt': ['max','min','std','count'],
}).reset_index()
cano_dt_agg.columns = ['_'.join([str(y) for y in x if y]) for x in cano_dt_agg.columns]
cano_dt_agg.columns=['chid','cano','cano_dt_max','cano_dt_min','cano_dt_std','cano_dt_cnt']
all_df = pd.merge(all_df,cano_dt_agg,how='left')

all_df['dif_day_maxdate']=all_df['cano_dt_max']-all_df['locdt']
all_df['dif_day_mindate']=all_df['locdt']-all_df['cano_dt_min']

all_df['locmonth']=all_df.locdt%30
all_df['locweekday']=all_df.locdt%7
all_df['locymonth']=all_df.locdt%12
all_df['locquarter']=all_df.locmonth%4
all_df['lochr']=all_df.loctm/10000
all_df['lochr']=all_df['lochr'].astype(int)

print("============== save csv:")
all_df.to_csv('../csv/all_df_1202.csv', index=False)
print("============== end")

