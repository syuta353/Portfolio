import streamlit as st
import pandas as pd
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from janome.tokenizer import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import label_binarize

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

st.title("🔍 モデル評価ツール")
matplotlib.rcParams['font.family'] = 'MS Gothic'

# データ読み込みと前処理
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").replace("\t", " ").strip()

def build_error_mapping(df):
    records = []
    for _, row in df.iterrows():
        error_text = str(row.get("障害エラー名抽出", ""))
        parts_text = str(row.get("交換部品", ""))
        device_type = str(row.get("装置種別", ""))
        error_codes = set(re.findall(r"\b\d{7}\b", error_text))
        for code in error_codes:
            records.append({
                "エラーコード": code,
                "交換部品": parts_text,
                "装置種別": device_type
            })
    return pd.DataFrame(records)

# データ読み込み
json_data = load_json("failed_db.json")
df = pd.DataFrame(json_data)
df["不具合内容_cleaned"] = df["不具合内容"].apply(clean_text)

# エラー名抽出（簡易版：7桁数字のみ）
df["障害エラー名抽出"] = df["不具合内容_cleaned"].apply(lambda x: " ".join(re.findall(r"\b\d{7}\b", x)))

# error_df構築
error_df = build_error_mapping(df)

# ラベルエンコード
le_code = LabelEncoder()
le_device = LabelEncoder()
le_parts = LabelEncoder()

error_df["error_code_encoded"] = le_code.fit_transform(error_df["エラーコード"])
error_df["device_type_encoded"] = le_device.fit_transform(error_df["装置種別"])
error_df["parts_encoded"] = le_parts.fit_transform(error_df["交換部品"])

# 特徴量と目的変数
X = error_df[["error_code_encoded", "device_type_encoded"]]
y = error_df["parts_encoded"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データ分割
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# モデル構築（max_iter 増加）
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
nb = MultinomialNB()
knn = KNeighborsClassifier(n_neighbors=5)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.001)

extended_voting = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('knn', knn), ('mlp', mlp)],
    voting='soft'
)

# 学習
extended_voting.fit(X_train, y_train)


def show_evaluation(name, model, X_test, y_test, le_parts):
    y_pred = model.predict(X_test)
    labels = np.unique(y_test)
    st.markdown(f"### 📊 {name} モデル評価")
    st.text(classification_report(y_test, y_pred, labels=labels, target_names=le_parts.inverse_transform(labels)))
    return y_pred
    
def plot_confusion_matrix(y_true, y_pred, le_parts, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le_parts.inverse_transform(np.unique(y_true)),
                yticklabels=le_parts.inverse_transform(np.unique(y_true)))
    ax.set_title(title)
    ax.set_xlabel("予測された部品")
    ax.set_ylabel("実際の部品")
    st.pyplot(fig)
    return cm

def show_top_misclassified(cm, le_parts):
    cm_df = pd.DataFrame(cm,
                         index=le_parts.inverse_transform(np.unique(y_test)),
                         columns=le_parts.inverse_transform(np.unique(y_test)))
    misclassified = cm_df.copy()
    np.fill_diagonal(misclassified.values, 0)
    misclassified_sum = misclassified.sum(axis=1).sort_values(ascending=False)
    st.markdown("### ❌ 誤分類の多い部品トップ10")
    st.dataframe(misclassified_sum.head(10))
    
def plot_pr_curve(model, X_test, y_test, le_parts,title):
    y_test_bin = label_binarize(y_test, classes=model.classes_)
    y_score = model.predict_proba(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(y_score.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        label = le_parts.inverse_transform([model.classes_[i]])[0]
        ax.plot(recall, precision, label=f"Class {label}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title + "　" + "Precision-Recall Curve")
    ax.legend()
    st.pyplot(fig)
   

# ==============================
# 📈 統計表示
# ==============================
st.markdown("### 📈 データ統計情報")
st.write("ユニークなエラーコード:", error_df["エラーコード"].nunique())
st.write("ユニークな装置種別:", error_df["装置種別"].nunique())
st.write("ユニークな交換部品:", error_df["交換部品"].nunique())

# 評価表示
y_pred_extended = extended_voting.predict(X_train)
st.markdown("### 📊 Extended VotingClassifier モデル評価")
st.text(classification_report(y_train, y_pred_extended, target_names=le_parts.inverse_transform(np.unique(y_train))))

# 混同行列と誤分類分析
st.markdown("### 🔍 各モデルの混同行列")
stack_cm = plot_confusion_matrix(y_train, y_pred_extended, le_parts, "Extended VotingClassifier の混同行列")

# Precision-Recall 曲線
st.markdown("### 📉 Precision-Recall 曲線")
plot_pr_curve(extended_voting, X_train, y_train, le_parts, "Extended VotingClassifier")


import joblib

st.markdown("### 🔍 装置種別ごとのモデルによる交換部品予測")

device_type_str = st.text_input("装置種別を入力")
error_code = st.number_input("エラーコード（7桁）を入力", min_value=1000000, max_value=9999999, step=1)
error_code_str = str(error_code)  # 数値を文字列に変換

if st.button("予測する"):
    safe_device = re.sub(r"[^\w\-]", "_", device_type_str)
    try:
        model_path = f"model/device_models/model_{safe_device}.pkl"
        le_code_path = f"model/device_models/le_code_{safe_device}.pkl"
        le_parts_path = f"model/device_models/le_parts_{safe_device}.pkl"
        scaler_path = f"model/device_models/scaler_{safe_device}.pkl"

        model = joblib.load(model_path)
        le_code = joblib.load(le_code_path)
        le_parts = joblib.load(le_parts_path)
        scaler = joblib.load(scaler_path)

        if error_code_str not in le_code.classes_:
            st.error("指定されたエラーコードはこの装置種別に存在しません。")
        else:
            code_encoded = le_code.transform([error_code_str])
            X_input = scaler.transform([[code_encoded[0]]])
            proba = model.predict_proba(X_input)[0]
            part_indices = proba.argsort()[::-1]

            st.markdown("#### 🔧 予測される交換部品とその確率")
            for idx in part_indices:
                percent = round(proba[idx] * 100, 2)
                if percent > 0.0:
                    part_name = le_parts.inverse_transform([idx])[0]
                    st.write(f"- {part_name}: {percent}%")
    except Exception as e:
        st.error(f"モデル読み込みまたは予測に失敗しました: {e}")