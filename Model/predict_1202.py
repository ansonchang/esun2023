import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import *
from sklearn.metrics import f1_score

all_df=pd.read_csv('../csv/all_df_1202.csv')
train_df=all_df[all_df.label.notnull()][all_df.locdt<63]
eval_df=all_df[all_df.label.notnull()][all_df.locdt>=63]
test2_df=all_df[all_df.label.isnull()]

s = setup(data = train_df, target = 'label', session_id=721, train_size=0.8, #preprocess=False,
          ignore_features=['txkey','chid','cano','mchno','acqic','locmonth','loctm','locdt'] ,
          data_split_stratify=['label'] )

X_train = get_config('X_train')
X_test = get_config('X_test')
Y_train = get_config('y_train')
Y_test = get_config('y_test')

xgb= create_model('xgboost', fold=2)
lgb= create_model('lightgbm', fold=2)
rf= create_model('rf',fold=2)
et= create_model('et',fold=2)

final_rf = finalize_model(rf)
final_xgb = finalize_model(xgb)
final_et = finalize_model(et)
final_lgb = finalize_model(lgb) 

save_model(final_rf, 'final_rf_from1202') 
save_model(final_xgb, 'final_xgb_from1202') 
save_model(final_et, 'final_et_from1202') 
save_model(final_lgb, 'final_lgb_from1202')  

submit_sample_df=pd.read_csv('../data/dataset_3rd/private_2_template_v2.csv')
test3_df=test2_df.drop(['label'], axis=1)

test_rf2=predict_model(final_rf,data=test3_df)
test_xgb2=predict_model(final_xgb,data=test3_df)

test_rf2['prediction_label_rf']=test_rf2['prediction_label']
test_xgb2['prediction_label_xgb']=test_xgb2['prediction_label']

test_xgb2['prob_xgb']=np.where(test_xgb2.prediction_label==1,test_xgb2.prediction_score,1-test_xgb2.prediction_score)
test_rf2['prob_rf']=np.where(test_rf2.prediction_label==1,test_rf2.prediction_score,1-test_rf2.prediction_score)
test_all2 = pd.merge(test_rf2,test_xgb2[['txkey','prediction_label_xgb','prob_xgb']],on=['txkey'],how='left')

test_all2['prediction_label_rf2']=np.where(test_all2.prob_rf>0.4,1,0)
test_all2['prediction_label_xgb2']=np.where(test_all2.prob_xgb>=0.5,1,0)
test_all2['prediction_label']=test_all2['prediction_label_rf2']+test_all2['prediction_label_xgb2']

test_all2['prediction_label_1']=np.where(test_all2['prediction_label']>=1,1,0) 

submit_df = submit_sample_df.merge(test_all2[['txkey','prediction_label_1']], on='txkey', how='left')
submit_df['pred']= submit_df['prediction_label_1']
submit_df['pred']=submit_df['pred'].astype(int)
submit_df[['txkey','pred']].to_csv('../csv/TEAM_4058.csv', index=False)




