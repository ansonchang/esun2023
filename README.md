## 環境
- 系統平台：Ubuntu 18.04.6
- 程式語言：python 3.8.13

## 每個資料夾/檔案的用途
```
├ Preprocess
│ └ preprocess.py             (前處理與特徵工程, 在 csv 目錄產生 dataset)
│ └ preprocess_1202.py        (複賽前處理與特徵工程, 在 csv 目錄產生 dataset)
├ Model
│ └ predict.py                (Model 的預測結果, 在 output 目錄產生最終的結果)
│ └ predict_1202.py           (複賽Model 的預測結果, 在 output 目錄產生最終的結果)
├ csv                         (放置 data prepocess 後的資料集目錄)
├ output                      (放置 model prediction 後的資料集目錄)
├ requirements.txt
└ README.md
```

## 可復現步驟

0. 安裝套件
```
$ pip install -r requirements.txt
```

1. 執行資料預處理步驟 : (原始 dataset 需放置在上一層 data 目錄)
```
$ cd Proprocess
$ python preprocess.py
```

2. 執行模型訓練與預測 : 
```
$ cd Model
$ python predict.py
```
