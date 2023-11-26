import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pycaret.classification import *
import time

start = time.time()
all_df=pd.read_csv('../csv/all_df.csv')

train_df=all_df[all_df.label.notnull()][all_df.locdt<=58]
eval_df=all_df[all_df.label.notnull()][all_df.locdt>58]
test2_df=all_df[all_df.label.isnull()]

print("============== model init:")
s = setup(data = train_df, target = 'label', session_id=42, train_size=0.8, #preprocess=False,
          ignore_features=['txkey','chid','cano','mchno','acqic','locmonth','loctm','locdt'] ,
          data_split_stratify=['label'])

print("============== random forest model create:")
rf= create_model('rf',fold=2)

print("============== random forest model evaluate:")
final_rf = finalize_model(rf)
predict_model(final_rf,data=eval_df)

print("============== random forest model save:")
save_model(final_rf, 'final_rf_from000') 

print("============== random forest model predict:")
submit_sample_df=pd.read_csv('../data/31_範例繳交檔案.csv')

train_df['label2']=train_df['label']
test3_df=test2_df.drop(['label'], axis=1)

pred_holdout_rf = predict_model(final_rf,data=test3_df)
pred_eval_rf=predict_model(final_rf,data=eval_df)
pred_eval_rf['eval_label']=pred_eval_rf['prediction_label']

print("============== calculate preidct output:")
submit_df = submit_sample_df.merge(pred_holdout_rf[['txkey','prediction_label']], on='txkey', how='left')
submit_df = submit_df.merge(pred_eval_rf[['txkey','eval_label']], on='txkey', how='left')
submit_df = submit_df.merge(train_df[['txkey','label2']], on='txkey', how='left')

submit_df['pred']= np.where(submit_df.prediction_label.notnull(),submit_df.prediction_label,submit_df.pred)
submit_df['pred']= np.where(submit_df.eval_label.notnull(),submit_df.eval_label,submit_df.pred)
submit_df['pred']= np.where(submit_df.label2.notnull(),submit_df.label2,submit_df.pred)
submit_df['pred']=submit_df['pred'].astype(int)

print("============== save preidct output:")
submit_df[['txkey','pred']].to_csv('../output/rf_from000all_1124.csv', index=False)

print("執行時間：%f 秒" % ( time.time() - start))

