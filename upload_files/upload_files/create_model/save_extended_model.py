import os
import json
import re
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

class DeviceModelTrainer:
    def __init__(self, json_path, model_dir="device_models"):
        # JSONファイルのパスとモデル保存先ディレクトリを初期化
        self.json_path = json_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_json(self):
        # JSONファイルを読み込む
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def clean_text(self, text):
        # 改行やタブをスペースに置換し、前後の空白を除去
        if pd.isna(text):
            return ""
        return str(text).replace("\n", " ").replace("\t", " ").strip()

    def extract_error_codes(self, text):
        # 7桁の数字（エラーコード）を抽出
        return " ".join(re.findall(r"\b\d{7}\b", text))

    def build_error_mapping(self, df):
        # エラーコード・交換部品・装置種別の対応表を構築
        records = []
        for _, row in df.iterrows():
            error_text = str(row.get("障害エラー名抽出", ""))
            parts_text = str(row.get("交換部品", ""))
            device_type = str(row.get("装置種別", ""))
            error_codes = set(re.findall(r"\b\d{7}\b", error_text))
            split_parts = str(parts_text).splitlines()
            split_parts = [p.strip() for p in split_parts if p.strip()]
            for code in error_codes:
                for part in split_parts:
                    records.append({
                        "エラーコード": code,
                        "交換部品": part,
                        "装置種別": device_type
                    })
        return pd.DataFrame(records)

    def train_and_save_models(self):
        # モデル構築と保存のメイン処理
        json_data = self.load_json()
        df = pd.DataFrame(json_data)

        # 不具合内容の前処理とエラーコード抽出
        df["不具合内容_cleaned"] = df["不具合内容"].apply(self.clean_text)
        df["障害エラー名抽出"] = df["不具合内容_cleaned"].apply(self.extract_error_codes)

        # エラーコード・装置種別・交換部品の対応表を作成
        error_df = self.build_error_mapping(df)

        for device in error_df["装置種別"].unique():
            subset = error_df[error_df["装置種別"] == device].copy()
            sample_counts = subset["交換部品"].value_counts().to_dict()

            if len(subset["交換部品"].unique()) < 2:
                continue

            # ラベルエンコード
            le_code = LabelEncoder()
            le_parts = LabelEncoder()
            subset["error_code_encoded"] = le_code.fit_transform(subset["エラーコード"])
            subset["parts_encoded"] = le_parts.fit_transform(subset["交換部品"])

            # 特徴量と目的変数
            X = subset[["error_code_encoded"]]
            y = subset["parts_encoded"]

            # スケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 学習データと検証データに分割
            X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            knn_n = min(5, len(X_train))

            # クラスが1つしかない場合はスキップ
            if len(set(y_train)) < 2:
                continue

            # モデル構築
            model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
                    ('knn', KNeighborsClassifier(n_neighbors=knn_n)),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.001))
                ],
                voting='soft'
            )
            model.fit(X_train, y_train)

            # 保存
            safe_device = re.sub(r"[\\/:*?\"<>|]", "_", device)
            joblib.dump(model, os.path.join(self.model_dir, f"model_{safe_device}.pkl"))
            joblib.dump(le_code, os.path.join(self.model_dir, f"le_code_{safe_device}.pkl"))
            joblib.dump(le_parts, os.path.join(self.model_dir, f"le_parts_{safe_device}.pkl"))
            joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_{safe_device}.pkl"))
            joblib.dump(sample_counts, os.path.join(self.model_dir, f"sample_counts_{safe_device}.pkl"))

            # 保存内容の確認出力
            print(f"\n📦 装置種別: {device}")
            print("モデル構成:", model)
            print("エンコード対象エラーコード:", list(le_code.classes_))
            print("エンコード対象交換部品:", list(le_parts.classes_))
            print("スケーラー平均:", scaler.mean_)
            
        print("✅ Fin")

# 実行例
if __name__ == "__main__":
    trainer = DeviceModelTrainer(json_path="failed_db.json")
    trainer.train_and_save_models()