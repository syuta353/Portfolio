# Portfolio

## プロジェクト概要
- **Failure Count Dashboard**: Streamlit 製の可視化・データ整備ツール。故障データの Excel→JSON 変換、ビル名/装置種別集計、Plotly による年間・月次の故障件数可視化、異常検知、装置種別ごとの交換部品予測などを提供します。【F:upload_files/failure_dashboard/dashboard.py】
- **Create Model**: 故障内容から交換部品を推定する VotingClassifier を学習・保存するためのスクリプト群です。【F:upload_files/create_model/save_extended_model.py】

## 成果物/フォルダ構成
- `upload_files/failure_dashboard/`: ダッシュボード本体。`dashboard.py`（Streamlit UI）、`data.py`・`services.py`（データ読み込み・加工・モデル呼び出しを行うサービス層）を中心に、設定ファイル `.streamlit/config.toml` や入力用データ (`excel/original_db.xlsx`)、補助ログ (`data_log/missing_list.txt`) を格納します。【F:upload_files/failure_dashboard/directory_structure.txt】
- `upload_files/failure_dashboard/json/`: ダッシュボードが扱う JSON データ群（`failed_db.json`, `hostname.json`, `bill.json`, `d_type.json`）。
- `upload_files/create_model/`: モデル学習に使用するスクリプト (`save_extended_model.py`, `model_check.py`) とデータ (`failed_db.json`)、実行補助ファイルを格納しています。【F:upload_files/create_model/save_extended_model.py】

## モデル評価
Extended VotingClassifier を用いたモデルの評価結果（精度など）は次の通りです。【F:upload_files/create_model/model_result.txt】

| 指標 | 値 |
| --- | --- |
| Accuracy | 0.76 |
| Macro Avg (precision/recall/f1) | 0.54 / 0.58 / 0.54 |
| Weighted Avg (precision/recall/f1) | 0.77 / 0.76 / 0.75 |
| サポート件数 | 558 |

## 利用方法のヒント
- ダッシュボードは `upload_files/failure_dashboard/dashboard.py` を Streamlit で起動すると、サイドバーから JSON 変換や装置種別別の部品予測を実行できます。【F:upload_files/failure_dashboard/dashboard.py】
- モデルの再学習や更新は、`upload_files/create_model/save_extended_model.py` の `DeviceModelTrainer` で JSON データを読み込み、装置種別ごとのモデル・エンコーダを自動保存するフローを用います。【F:upload_files/create_model/save_extended_model.py】
