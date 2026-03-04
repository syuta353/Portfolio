# Failure Analysis Dashboard & Parts Prediction

故障履歴データを活用して  
**故障分析・異常検知・交換部品予測**を行う  
Streamlitベースのデータ分析ダッシュボードです。

Excel形式の故障データを読み込み、  
データ整形・分析・機械学習予測までを  
一つのUIで実行できるツールとして構築しています。

---

# 📊 ダッシュボード画面

### 全体概要
![ダッシュボード全体](https://github.com/syuta353/Portfolio/blob/main/upload_files/images/dashboard_overview.png)

### 異常検知結果
![異常検知](https://github.com/syuta353/Portfolio/blob/main/upload_files/images/anomaly_analysis.png)

### 交換部品予測
![予測結果](https://github.com/syuta353/Portfolio/blob/main/upload_files/images/prediction_result.png)

---

# 🧠 プロジェクト概要

本ツールは故障履歴データを基に
- 故障件数の可視化
- 異常検知
- 交換部品予測

などを行うデータ分析ダッシュボードです。

故障情報のExcelデータを整形し、  
分析・予測までを一貫して行えるツールとして設計しています。

---

# ⚙ 技術スタック

|分類|技術|
|---|---|
Language | Python  
UI | Streamlit  
Data Processing | Pandas / NumPy  
Machine Learning | Scikit-learn  
Models | RandomForest / LogisticRegression / KNN / MLP  
Ensemble | VotingClassifier  
Visualization | Plotly  
Data Format | Excel / JSON  
Model Storage | Joblib  

交換部品予測モデルは以下のアルゴリズムを組み合わせた  
**VotingClassifierアンサンブルモデル**を使用しています。

- RandomForestClassifier
- LogisticRegression
- KNeighborsClassifier
- MLPClassifier

---

# 📊 主な機能

### データ整形
- Excel故障データ読み込み
- JSONデータ生成
- 部品名の表記ゆれ補正
- ホスト名から装置種別・ビル名を解決

### データ分析
- 年間 / 月次故障件数の可視化
- 故障件数統計量算出
- しきい値ベース異常検知

### 機械学習
- エラーコードから交換部品を予測
- 装置種別ごとのモデル生成
- 学習データ量に基づく信頼度表示

---

# 🏗 システム構成
```
upload_files/
├── failure_dashboard/
│   ├── dashboard.py
│   ├── data.py
│   ├── services.py
│   ├── .streamlit/
│   │   └── config.toml
│   ├── excel/
│   │   └── original_db.xlsx
│   ├── data_log/
│   │   └── missing_list.txt
│   └── json/
│       ├── failed_db.json
│       ├── hostname.json
│       ├── bill.json
│       └── d_type.json
│
└── create_model/
    ├── save_extended_model.py
    ├── model_check.py
    └── failed_db.json
```

---

# 🚀 実行方法

### 1. 必要パッケージのインストール
```pip install -r requirements.txt```

### 2. ダッシュボードの起動
```cd upload_files/failure_dashboard```
```streamlit run dashboard.py```

---

# 📈 モデル評価

|指標|値|
|---|---|
Accuracy | 0.76  
Macro Avg | 0.54 / 0.58 / 0.54  
Weighted Avg | 0.77 / 0.76 / 0.75  
サンプル数 | 558  

---

# 📌 ポートフォリオ目的

本プロジェクトは以下スキルを示すためのポートフォリオです。

- データ分析
- 機械学習モデル構築
- Streamlitダッシュボード開発
- データ処理パイプライン構築
