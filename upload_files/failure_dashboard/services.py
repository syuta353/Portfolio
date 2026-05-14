import io
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from data import DataManager

#インスタンス化
data_manager = DataManager()


"""
0. DeviceModelTrainer（装置種別ごとの拡張モデル生成）
責務：

・JSONデータからエラーコードと交換部品の対応表を生成
・装置種別ごとに VotingClassifier を学習し、関連するエンコーダやスケーラーと一緒に保存

主な関数：

・train_and_save_models()

"""
class DeviceModelTrainer:
    def __init__(self):
        self.json_path = data_manager.get_path("output_json")
        self.model_dir = data_manager.get_model_dir()
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_json(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _clean_text(self, text):
        if pd.isna(text):
            return ""
        return str(text).replace("\n", " ").replace("\t", " ").strip()

    def _extract_error_codes(self, text):
        return " ".join(re.findall(r"\b\d{7}\b", text))

    def _build_error_mapping(self, df):
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
        with st.spinner("🤖 拡張モデルを再学習しています..."):
            json_data = self._load_json()
            df = pd.DataFrame(json_data)

            df["不具合内容_cleaned"] = df["不具合内容"].apply(self._clean_text)
            df["障害エラー名抽出"] = df["不具合内容_cleaned"].apply(self._extract_error_codes)

            error_df = self._build_error_mapping(df)

            for device in error_df["装置種別"].unique():
                subset = error_df[error_df["装置種別"] == device].copy()
                sample_counts = subset["交換部品"].value_counts().to_dict()

                if len(subset["交換部品"].unique()) < 2:
                    continue

                le_code = LabelEncoder()
                le_parts = LabelEncoder()
                subset["error_code_encoded"] = le_code.fit_transform(subset["エラーコード"])
                subset["parts_encoded"] = le_parts.fit_transform(subset["交換部品"])

                X = subset[["error_code_encoded"]]
                y = subset["parts_encoded"]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
                knn_n = min(5, len(X_train))

                if len(set(y_train)) < 2:
                    continue

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

                safe_device = re.sub(r"[\\/:*?\"<>|]", "_", device)
                joblib.dump(model, os.path.join(self.model_dir, f"model_{safe_device}.pkl"))
                joblib.dump(le_code, os.path.join(self.model_dir, f"le_code_{safe_device}.pkl"))
                joblib.dump(le_parts, os.path.join(self.model_dir, f"le_parts_{safe_device}.pkl"))
                joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_{safe_device}.pkl"))
                joblib.dump(sample_counts, os.path.join(self.model_dir, f"sample_counts_{safe_device}.pkl"))

        st.success("✅ 拡張モデルの再学習と保存が完了しました。")

"""
1. DeviceModelManager（モデルの読み込み・予測）
責務：

・モデル・エンコーダ・スケーラーの読み込み
・交換部品の予測、選択された装置種別の信頼度の表示
・ジャンプスタート警告の判定

主な関数：

・load_model(device_type: str)
・predict_parts(error_code: str)
・check_jumpstart_warning(error_code: str)

"""
class DeviceModelManager:
    def __init__(self):
        self.model = None
        self.le_code = None
        self.le_parts = None
        self.scaler = None
        self.sample_counts = None
        self.all_device_totals = None  # 全装置種別の総件数リスト

    def load_model(self, device_type):
        model_dir = data_manager.get_model_dir()
        try:
            # モデルと関連ファイル読み込み
            self.model = joblib.load(os.path.join(model_dir, f"model_{device_type}.pkl"))
            self.le_code = joblib.load(os.path.join(model_dir, f"le_code_{device_type}.pkl"))
            self.le_parts = joblib.load(os.path.join(model_dir, f"le_parts_{device_type}.pkl"))
            self.scaler = joblib.load(os.path.join(model_dir, f"scaler_{device_type}.pkl"))
            self.sample_counts = joblib.load(os.path.join(model_dir, f"sample_counts_{device_type}.pkl"))

            # 全装置種別の総件数を取得
            self.all_device_totals = []
            for file in os.listdir(model_dir):
                if file.startswith("sample_counts_") and file.endswith(".pkl"):
                    counts = joblib.load(os.path.join(model_dir, file))
                    self.all_device_totals.extend(counts.values())
        except Exception as e:
            st.error(f"モデルまたは関連ファイルの読み込みに失敗しました: {e}")

    def predict_parts(self, error_code):
        if error_code not in self.le_code.classes_:
            st.error("指定されたエラーコードはこの装置種別に存在しません。")
            return

        # エラーコードをエンコードして予測
        code_encoded = self.le_code.transform([error_code])
        X_input = self.scaler.transform(pd.DataFrame({"error_code_encoded": [code_encoded[0]]}))
        proba = self.model.predict_proba(X_input)[0]
        part_indices = proba.argsort()[::-1]
        
        
        # 装置種別の総学習件数
        device_total = sum(self.sample_counts.values())
        
        # 全装置種別の総件数の統計量を取得
        series = pd.Series(self.all_device_totals)
        stats = series.describe()
        
        # 全装置種別の分布からしきい値を決定
        #高い数値の外れ値に影響を受けているので50%以下を小にする
        q50 = stats["50%"]
        q75 = stats["75%"]

        # 装置種別の信頼度ラベル
        if device_total <= q50:
            device_confidence = "小"
        elif device_total >= q75:
            device_confidence = "大"
        else:
            device_confidence = "中"

        st.markdown("### 🏭 予測される交換部品の信頼度")
        st.write(f"装置種別の総学習件数: **{device_total} 件**")
        st.write(f"信頼度: **{device_confidence}**")
        
        # 予測結果の表示
        st.markdown("### 🔍 予測される交換部品とその確率")
        predicted_parts = []
        probabilities = []

        for idx in part_indices:
            percent = round(proba[idx] * 100, 2)
            if percent > 0.0:
                part_name = self.le_parts.inverse_transform([idx])[0]
                st.write(f"- {part_name}: {percent}%")
                
        st.markdown("### ℹ️ 信頼度についての説明")
        st.markdown("""
        信頼度は、**この装置種別の学習データの量に基づいて、予測結果の確かさを示す指標**です。
        学習データが多いほど、モデルはより多くの事例を学んでいるため、予測の精度が高くなる傾向があります。
        逆に、学習データが少ない場合は、予測の確かさが低くなる可能性があります。

        - **大**：学習データが多く、予測の信頼性が高い
        - **中**：学習データが標準的で、予測の信頼性は普通
        - **小**：学習データが少なく、予測の信頼性が低い可能性がある
        """)
        
    # 警告表示処理
    def check_jumpstart_warning(self, error_code, device_type):
        """
        指定されたエラーコードがジャンプスタート対象かどうかと
        指定された装置種別をジャンプスタート対象の装置種別かどうかを判定し、警告を表示します。

        引数:
        - error_code (str): エラーコード。
        - device_type (str): 装置種別。

        戻り値:
        - なし（Streamlit上に警告表示）
        """
        jump_errors = data_manager.get_device_jump_errors()
        
        if device_type in jump_errors:
            if error_code in jump_errors[device_type]:
                st.warning(data_manager.jump_warning)
        
        if device_type in ["device_type1", "device_type2", "device_type3"]:
            st.warning(data_manager.route_warning)

"""
2. FailureDataProcessor（Excel→JSON変換、部品名正規化）
責務：

・Excelシートの読み込みと整形
・部品名の正規化
・JSON保存
・# Excel出力

主な関数：

・process_sheet(sheet_name, file_path)
・convert_excel_to_json(file_path)
・clean_part_name(part, counter_text)
・normalize_parts(part)
・save_to_json(df, path)
・export_excel

"""
class FailureDataProcessor:
    def __init__(self):
        self.rear_word, self.front_word = data_manager.get_keywords()
        self.host_df, self.type_df, self.bill_df = data_manager.get_dataframes()
        self.missing_list = data_manager.get_missing_list()

    def process_sheet(self, sheet_name, file_path):
        """
        指定されたExcelシートを読み込み、構造化された故障データを抽出します。

        引数:
        - sheet_name (str): 処理対象のシート名。
        - file_path (str): Excelファイルのパス。

        戻り値:
        - list: 故障データの辞書リスト。
        """
        output_data = []
        try:
            df = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=1,
                usecols=["発生日", "発生局", "装置種別", "装置名(host)", "不具合内容", "現地対応部門での対策（処置）", "オンサイト交換部品"],
                engine='openpyxl'
            )
            df = df.rename(columns={
                '発生局': 'ビル名',
                '装置名(host)': '装置名',
                '現地対応部門での対策（処置）': '対策内容',
                'オンサイト交換部品': '交換部品'
            })
            df['発生日'] = pd.to_datetime(df['発生日'], errors='coerce')
            df['交換部品'] = df['交換部品'].apply(self.normalize_parts)
            df = df.replace("-", pd.NA).replace("不明", pd.NA).replace("", pd.NA)
            df = df.where(pd.notna(df), None)
            df = df.dropna(subset=['発生日', 'ビル名', '装置種別', '装置名', '交換部品'])

            resolver = HostInfoResolver()

            for _, row in df.iterrows():
                for dname in str(row['装置名']).splitlines():
                    dname = dname.strip()
                    result = resolver.get_info(dname)
                    if result is None:
                        continue
                    bill, type_ = result
                    original_part = str(row['交換部品'])
                    counter_text = row['対策内容']
                    part_type = self.clean_part_name(original_part, counter_text)
                    if part_type in ["target_type1", "target_type2", "target_type3"] and part_type != type_:
                        continue
                    output_data.append({
                        "発生日": row['発生日'].strftime('%Y-%m-%d'),
                        "ビル名": bill,
                        "装置種別": type_,
                        "装置名": dname,
                        "不具合内容": row['不具合内容'],
                        "対策内容": row['対策内容'],
                        "交換部品": part_type,
                        "年度": row['発生日'].strftime('%Y')
                    })
        except Exception as e:
            print(f"{sheet_name} の読み込み中にエラーが発生しました: {e}")
        return output_data

    def convert_excel_to_json(self, file_path):
        """
        Excelファイル内の複数シートを読み込み、1つのJSONファイルに変換して保存します。

        引数:
        - file_path (str): Excelファイルのパス。

        戻り値:
        - なし
        """
        with st.spinner("🔄 JSON変換中です。しばらくお待ちください..."):
            all_output_data = []
            for sheet in data_manager.get_target_sheets():
                all_output_data.extend(self.process_sheet(sheet, file_path))
            with open(data_manager.get_path("output_json"), 'w', encoding='utf-8') as f:
                json.dump(all_output_data, f, ensure_ascii=False, indent=2)
            with open(data_manager.get_path("missing_list"), 'w', encoding='utf-8') as f:
                for item in self.missing_list:
                    f.write(item + '\n')
                self.missing_list.clear()

    def clean_part_name(self, part, counter_text):
        """
        交換部品名をルールに従って正規化する関数。

        引数:
        - part (str): 元の交換部品名
        - countermeasure_text (str): 対策内容
        - rear_word (list): リア判定用キーワードリスト
        - front_word (list): フロント判定用キーワードリスト

        戻り値:
        - str: 正規化された交換部品名
        """
        if not isinstance(part, str):
            return part
        part_lower = part.lower()
        if "target_part" in part_lower:
            text = str(counter_text).lower()
            is_rear = any(word.lower() in text for word in self.rear_word)
            is_front = any(word.lower() in text for word in self.front_word)
            if is_rear and is_front:
                part = "replace_part"
            elif is_rear:
                part = "replace_part"
            elif is_front:
                part = "replace_part"
            else:
                part = "replace_part"
        if "target_part" in part_lower or "target_part" in part:
            part = "replace_part"
        elif "target_part" in part:
            part = "replace_part"
        elif "target_part" in part_lower:
            part = "replace_part"
        elif "target_part" in part_lower:
            part = "replace_part"
        part = re.sub(r"\(.*?\)", "", part)
        part = re.sub(r"※.*", "", part)
        return part.strip()

    def normalize_parts(self, part):
        """
        部品名の表記ゆれを正規化します。

        引数:
        - part (str): 元の部品名。

        戻り値:
        - str: 正規化された部品名。
        """
        if pd.isna(part):
            return part
        part = str(part)
        part = part.replace("-", "経過観察/HW異常なし")
        part = part.replace("target_part1", "不明")
        part = re.sub(r"target_part2", "不明", part)
        part = re.sub(r"target_part3", "不明", part)
        return part

    def save_to_json(self, df, path):
        """
        DataFrameを整形されたJSONファイルとして保存します。

        引数:
        - df (pd.DataFrame): 保存対象のデータ。
        - json_path (str): 保存先のファイルパス。

        戻り値:
        - なし
        """
        df = df.fillna("")
        df["id"] = df["id"].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
        json_data = df.to_json(orient='records', force_ascii=False)
        json_obj = json.loads(json_data)
        json_str = json.dumps(json_obj, indent=4, ensure_ascii=False)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
    def export_excel(self, df, sheet_name, file_name, button_label="📥 Excelファイルをダウンロード"):
        """
        DataFrameをExcelファイルとして出力し、Streamlit上にダウンロードボタンを表示します。

        引数:
        - df (pd.DataFrame): 出力対象のデータ。
        - sheet_name (str): Excelシート名。
        - file_name (str): ダウンロード時のファイル名。
        - button_label (str): ダウンロードボタンのラベル。

        戻り値:
        - なし
        """
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        st.download_button(
            label=button_label,
            data=buffer.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
"""
3. AnomalyDetector（異常検知）
責務：

・故障件数の集計
・異常判定
・統計情報の算出

主な関数：

・detect_anomalies(df, type_counts, threshold, unit, year)
・count_anomalies_by_type(type_counts, threshold)
・calculate_failure_statistics(df, selected_years)

"""
class AnomalyDetector:
    def detect_anomalies(self, df, type_counts, threshold_percent, aggregation_unit, selected_year=None):
        """
        故障件数に基づいて異常を検知し、異常・正常のラベルを付与します。

        引数:
        - df (DataFrame): 故障データ。
        - type_counts (DataFrame): 装置種別ごとの母数。
        - threshold_percent (float): 異常判定のしきい値（%）。
        - aggregation_unit (str): 集計単位（年度単位または月間単位）。
        - selected_year (int, optional): 指定年度（必要に応じて）。

        戻り値:
        - DataFrame: 異常判定結果を含むデータ。
        """
        threshold = threshold_percent / 100
        anomaly_results = []
        df['発生日'] = pd.to_datetime(df['発生日'], errors='coerce').dt.strftime('%Y-%m-%d')

        for dtype in df['装置種別'].dropna().unique():
            sub_df = df[df['装置種別'] == dtype]

            if aggregation_unit == "指定年度の月間単位":
                if selected_year is None:
                    continue
                sub_df = sub_df[sub_df['年度'] == selected_year]
                grouped = sub_df.groupby(['年度', '月']).size().reset_index(name='件数')
                if len(grouped) < 1:
                    continue
                grouped['発生日'] = grouped.apply(
                    lambda row: f"{int(row['年度']):04d}-{int(row['月']):02d}-01",
                    axis=1
                )
            else:
                grouped = sub_df.groupby(['年度']).size().reset_index(name='件数')
                if len(grouped) < 1:
                    continue
                grouped['発生日'] = grouped['年度'].astype(int).astype(str) + '-01-01'

            mother_count = type_counts.query("d_type_name == @dtype")['count'].sum()
            if mother_count == 0:
                continue

            threshold_count = int(np.ceil(mother_count * threshold))
            grouped['装置種別'] = dtype
            grouped['異常'] = grouped['件数'] >= threshold_count
            grouped['異常'] = grouped['異常'].apply(lambda x: '異常' if x else '正常')

            anomaly_results.append(grouped)

        if anomaly_results:
            result_df = pd.concat(anomaly_results, ignore_index=True)
            result_df = result_df.sort_values(by='発生日', ascending=True).reset_index(drop=True)
            return result_df
        else:
            return pd.DataFrame()

    def count_anomalies_by_type(self, type_counts, threshold_percent):
        """
        装置種別ごとのしきい値件数を計算する関数。
        Parameters:
            type_counts (pd.DataFrame): 装置種別ごとの全体数を含む DataFrame（列: 'd_type_name', 'count'）
            threshold_percent (float): 閾値（％）
        Returns:
            pd.DataFrame: ['装置種別', '全体数', 'しきい値件数'] を含む DataFrame
        """
        threshold_ratio = threshold_percent / 100.0
        result_df = type_counts.copy()
        result_df = result_df.rename(columns={'d_type_name': '装置種別', 'count': '全体数'})
        result_df['しきい値件数'] = np.ceil(result_df['全体数'] * threshold_ratio).astype(int)
        return result_df

    def calculate_failure_statistics(self, df, selected_years):
        """
        故障件数の統計量（平均・中央値・最小・最大・標準偏差）と傾向説明を算出します。

        引数:
        - df (DataFrame): 故障データ。
        - selected_years (list): 対象年度のリスト。

        戻り値:
        - dict: 統計量と傾向説明を含む辞書。
        """
        if not selected_years:
            return {
                "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0,
                "trend_description": "年度が選択されていません。",
                "label": "（年度未選択）"
            }

        if len(selected_years) == 1:
            year = selected_years[0]
            values = df[df['年度'] == year]['月'].value_counts().sort_index().values
            label = f"{year}年（月別）"
            mode = "月"
            thresholds = [5, 15]
        else:
            values = df[df['年度'].isin(selected_years)]['年度'].value_counts().sort_index().values
            min_year, max_year = min(selected_years), max(selected_years)
            label = f"{min_year}〜{max_year}年" if min_year != max_year else f"{min_year}年"
            mode = "年度"
            thresholds = [10, 30]

        if len(values) == 0:
            stats = {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
        else:
            stats = {
                "mean": int(np.nanmean(values)),
                "median": int(np.nanmedian(values)),
                "min": int(np.nanmin(values)),
                "max": int(np.nanmax(values)),
                "std": int(np.nanstd(values))
            }

        std_val = stats["std"]
        if std_val == 0:
            trend_description = f"{mode}ごとの故障件数はほぼ一定で、ばらつきがありません。"
        elif std_val < thresholds[0]:
            trend_description = f"{mode}ごとの故障件数は比較的安定しており、大きな変動は見られません。"
        elif std_val < thresholds[1]:
            trend_description = f"{mode}ごとの故障件数には中程度のばらつきがあり、{mode}によって多少の違いがあります。"
        else:
            trend_description = f"{mode}ごとの故障件数には大きなばらつきがあり、{mode}によって故障件数に大きな違いがあります。"

        return {
            **stats,
            "trend_description": trend_description,
            "label": label
        }

"""
4. HostInfoResolver（ホスト名→装置種別・ビル名）
責務：

・ホスト名から関連情報を取得
・欠損リストの管理

主な関数：

・get_info(hostname)
・aggregate_counts()

"""
class HostInfoResolver:
    def __init__(self):
        self.host_df, self.type_df, self.bill_df = data_manager.get_dataframes()
        self.missing_list = data_manager.get_missing_list()

    def get_info(self, hostname):
        
        """
        指定されたホスト名に対応するビル名と装置種別を取得します。
        欠損がある場合は missing_list に日本語メッセージを追加します。

        引数:
        - hostName (str): 対象のホスト名。
        - host_df (pd.DataFrame): ホスト情報。
        - bill_df (pd.DataFrame): ビル情報。
        - type_df (pd.DataFrame): 装置種別情報。
        - missing_list (list): 欠損メッセージを記録するリスト。

        戻り値:
        - tuple: (ビル名, 装置種別)。該当なしの場合は None。
        """
        data = self.host_df.loc[self.host_df['hostname'] == hostname]
        if data.empty:
            self.missing_list.append(f"ホスト名 '{hostname}' に該当するデータが見つかりませんでした。")
            return None

        data = data.iloc[0]

        bill_match = self.bill_df.loc[self.bill_df['key'] == data['bill']]
        bill_name = bill_match['name'].values[0] if not bill_match.empty else "不明"
        if bill_match.empty:
            self.missing_list.append(f"ビル名 '{data['bill']}' に該当するデータが見つかりませんでした。")

        if data['d_type'] not in self.type_df:
            d_type = "不明"
            self.missing_list.append(f"装置種別 '{data['d_type']}' に該当するデータが見つかりませんでした。")
        else:
            d_type = self.type_df[data['d_type']][0]

        return bill_name, d_type

    def aggregate_counts(self):
        """
        ホスト情報からビル名と装置種別の出現回数を集計します。

        引数:
        - host_df (pd.DataFrame): ホスト情報。
        - bill_df (pd.DataFrame): ビル情報。
        - type_df (pd.DataFrame): 装置種別情報。

        戻り値:
        - tuple: (ビル別件数のDataFrame, 装置種別別件数のDataFrame)
        """
        results = []
        for hostname in self.host_df['hostname']:
            result = self.get_info(hostname)
            if result is not None:
                bill_name, d_type_name = result
                results.append({'bill_name': bill_name, 'd_type_name': d_type_name})

        results_df = pd.DataFrame(results)
        bill_counts = results_df.groupby('bill_name').size().reset_index(name='count')
        type_counts = results_df.groupby('d_type_name').size().reset_index(name='count')
        return bill_counts, type_counts
        

"""
5. ThemeManager（UIテーマ適用）
責務：

・StreamlitのCSSテーマ適用

主な関数：

・apply_theme(theme_name: str)

"""
class ThemeManager:
    def apply_theme(self, theme_name: str):
        """
        選択されたテーマに応じてStreamlitアプリのCSSスタイルを適用します。

        引数:
        - theme_option (str): テーマ名（ライト、ダークなど）。

        戻り値:
        - なし
        """
        themes = {
            "ライト": {
                "bg_color": "#ffffff", "text_color": "#000000", "app_bg": "#f8f8f8",
                "metric_bg": "#e6f2ff", "button_bg": "#e0e0e0", "tab_bg": "#e6f2ff",
                "tab_selected_bg": "#cce0ff", "expander_bg": "#e6f2ff", "expander_border": "#cce0ff",
                "sidebar_bg": "#f0f0f0", "dataframe_bg": "#ffffff"
            },
            "ダーク": {
                "bg_color": "#1e1e1e", "text_color": "#ffffff", "app_bg": "#2e2e2e",
                "metric_bg": "#333333", "button_bg": "#444444", "tab_bg": "#444444",
                "tab_selected_bg": "#666666", "expander_bg": "#333333", "expander_border": "#666666",
                "sidebar_bg": "#2e2e2e", "dataframe_bg": "#333333"
            },
            "ブルー": {
                "bg_color": "#e6f0ff", "text_color": "#003366", "app_bg": "#d9e6f2",
                "metric_bg": "#cce0ff", "button_bg": "#99ccff", "tab_bg": "#cce0ff",
                "tab_selected_bg": "#99ccff", "expander_bg": "#cce0ff", "expander_border": "#99ccff",
                "sidebar_bg": "#b3d1ff", "dataframe_bg": "#ffffff"
            },
            "グリーン": {
                "bg_color": "#e6ffe6", "text_color": "#004d00", "app_bg": "#ccffcc",
                "metric_bg": "#b3ffb3", "button_bg": "#80ff80", "tab_bg": "#b3ffb3",
                "tab_selected_bg": "#80ff80", "expander_bg": "#b3ffb3", "expander_border": "#80ff80",
                "sidebar_bg": "#99ff99", "dataframe_bg": "#ffffff"
            }
        }

        if theme_name not in themes:
            return

        t = themes[theme_name]
        st.markdown(f"""
        <style>
        body {{ background-color: {t['bg_color']}; color: {t['text_color']}; }}
        .stApp {{ background-color: {t['app_bg']}; }}
        h1, h2, h3, h4, h5, h6, p, .stMarkdown {{ color: {t['text_color']} !important; }}
        .stMetric {{ background-color: {t['metric_bg']}; padding: 10px; border-radius: 5px; }}
        .stMetric label {{ color: {t['text_color']} !important; font-weight: bold; }}
        .stMetric div {{ color: {t['text_color']} !important; }}
        .stButton>button, .stDownloadButton>button {{
            background-color: {t['button_bg']};
            color: {t['text_color']};
            font-weight: bold;
            border-radius: 5px;
            padding: 6px 12px;
        }}
        label[data-testid="stSelectboxLabel"] {{
            color: {t['text_color']} !important;
            font-weight: bold;
        }}
        .stDataFrame {{ background-color: {t['dataframe_bg']}; color: {t['text_color']}; }}
        .stSidebar {{ background-color: {t['sidebar_bg']}; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {t['tab_bg']};
            color: {t['text_color']};
            border-radius: 5px;
            padding: 6px;
            margin-right: 4px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {t['tab_selected_bg']};
            font-weight: bold;
        }}
        .stExpander {{
            background-color: {t['expander_bg']} !important;
            color: {t['text_color']} !important;
            border: 1px solid {t['expander_border']};
            border-radius: 5px;
        }}
        .stExpanderHeader {{
            color: {t['text_color']} !important;
            font-weight: bold;
        }}
        </style>
        """, unsafe_allow_html=True)