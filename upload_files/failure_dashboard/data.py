import os
import pandas as pd

class DataManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # ファイルパス定義
        self.paths = {
            "host_json": "app/static/json/hostname.json",
            "type_json": "app/static/json/d_type.json",
            "bill_json": "app/static/json/bill.json",
            "output_json": "app/static/json/failed_db.json",
            "missing_list": os.path.join(self.base_dir, "data_log/missing_list.txt"),
            "original_db": os.path.join(self.base_dir, "excel/original_db.xlsx"),
            "model_dir": os.path.join(self.base_dir, "device_models")
        }

        # 対象年度のシート名一覧（2009〜2025）
        self.target_sheets = [f"{year}年度" for year in range(2009, 2026)]

        # 交換部品の判定用キーワード
        #元ファイルでは様々な形式で記載されているため
        self.rear_keywords = ["word1", "word2", "word3"]
        self.front_keywords = ["word1", "word2", "word3"]

        # データフレームの読み込み
        self.host_df = pd.read_json(self.paths["host_json"])
        self.type_df = pd.read_json(self.paths["type_json"])
        self.bill_df = pd.read_json(self.paths["bill_json"])

        # 欠損リストの初期化
        self.missing_list = []
        
        # ジャンプスタート対象エラーコードの定義
        self.device_jump_errors = {
            "device_type1": ['num', 'num', 'num', 'num', 'num'],
            "device_type2": ['num', 'num', 'num', 'num', 'num'],
            "device_type3": ['num', 'num', 'num']
        }

        # 警告文
        self.jump_warning = (
            "ジャンプスタート対象の可能性があります。\n"
            "○○様より即時交換の要望がある場合は現地配備品を利用して交換する事が可能です。"
        )
        self.route_warning = (
            "コマンド実行結果が'no route to host'の場合はジャンプスタート対象です。\n"
            "○○様より即時交換の要望がある場合は現地配備品を利用して交換する事が可能です。"
        )

    def get_path(self, key):
        return self.paths.get(key)

    def get_target_sheets(self):
        return self.target_sheets

    def get_keywords(self):
        return self.rear_keywords, self.front_keywords

    def get_dataframes(self):
        return self.host_df, self.type_df, self.bill_df

    def get_missing_list(self):
        return self.missing_list

    def get_model_dir(self):
        return self.paths["model_dir"]
        
    def get_device_jump_errors(self):
        return self.device_jump_errors
        
    def get_jump_warning(self):
        return self.jump_warning
        
    def get_route_warning(self):
        return self.route_warning
