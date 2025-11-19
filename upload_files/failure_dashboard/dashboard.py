# ----------------------------------------
# 📦 必要なモジュールとクラスのインポート
# ----------------------------------------
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# クラスベースのサービス群をインポート
from services import (
    DeviceModelManager,
    FailureDataProcessor,
    AnomalyDetector,
    HostInfoResolver,
    ThemeManager
)
from data import DataManager

# ----------------------------------------
# 🧠 各クラスのインスタンス化
# ----------------------------------------
data_manager = DataManager()
model_manager = DeviceModelManager()
data_processor = FailureDataProcessor()
anomaly_detector = AnomalyDetector()
host_resolver = HostInfoResolver()
theme_manager = ThemeManager()

# ----------------------------------------
# 🖼️ ページ設定とテーマ選択
# ----------------------------------------
st.set_page_config(page_title="Failure Count Dashboard", layout="wide")

# CSS設定：expanderの最小高さを指定して、セクションを閉じたときのスクロール操作性を改善する
st.markdown("""
<style>
.stExpander {
    min-height: 100px !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 🎨 サイドバーUI（テーマ選択・Excel→JSON変換・部品予測）
# ----------------------------------------

with st.sidebar:
    # ----------------------------------------
    # 🎨 テーマ選択と適用
    # ----------------------------------------
    theme_option = st.selectbox("🎨 表示スタイルを選択", ["デフォルト", "ライト", "ダーク", "ブルー", "グリーン"])
    theme_manager.apply_theme(theme_option)

    st.markdown("---")
    st.markdown("### 📄 Excel_TO_JSON")

    # ----------------------------------------
    # 📄 JSON変換対象の選択
    # ----------------------------------------
    conversion_target = st.selectbox("変換対象を選択", ["failure_data", "hostname", "bill", "device_type"])

    if conversion_target == "failure_data":
        # Excelファイルアップロード
        uploaded_file = st.file_uploader("Excelファイルをアップロードしてください（.xlsxのみ）", type=["xlsx"], key="excel_json_upload")

        if uploaded_file:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ファイル名チェック
            if "NTT-NGN西日本" not in uploaded_file.name:
                st.error("❌ ファイル名に 'NTT-NGN西日本' が含まれていないため、処理を実行できません。")
            else:
                data_processor.convert_excel_to_json(temp_path)
                st.success("✅ 故障データのJSON変換が完了しました。")
    else:
        # hostname, bill, device_type の編集処理
        sheet_name = conversion_target
        try:
            df = pd.read_excel(data_manager.get_path("original_db"), sheet_name=sheet_name, engine='openpyxl')
            st.markdown(f"#### ✏️ {sheet_name} シートの編集")
            edited_df = st.data_editor(df, num_rows="dynamic")

            if st.button("JSON保存"):
                json_path = data_manager.get_path(f"{sheet_name}_json")
                data_processor.save_to_json(edited_df, json_path)

                with pd.ExcelWriter(data_manager.get_path("original_db"), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    edited_df.to_excel(writer, sheet_name=sheet_name, index=False)

                st.success(f"✅ {sheet_name} のJSON保存とExcel上書きが完了しました。")
        except Exception as e:
            st.error(f"❌ シートの読み込み中にエラーが発生しました: {e}")

    # ----------------------------------------
    # 🔧 装置種別ごとの交換部品予測セクション
    # ----------------------------------------
    st.markdown("---")
    st.markdown("### 🔧 装置種別ごとの交換部品予測セクション")

    # モデルディレクトリ内の "model_<type>.pkl" ファイル名から装置種別を安全に取得します。
    # ディレクトリが存在しない、または該当ファイルがない場合は空リストを返します。
    model_dir = Path(data_manager.get_model_dir())
    device_types = []
    try:
        if model_dir.exists() and model_dir.is_dir():
            device_types = sorted([
                fname[len("model_"):-len(".pkl")]
                for fname in os.listdir(model_dir)
                if fname.startswith("model_") and fname.endswith(".pkl")
            ])
    except Exception:
        # 失敗しても UI が壊れないように空リストを使う
        device_types = []

    device_type_str = st.selectbox("装置種別を選択", device_types)
    error_code = st.text_input("エラーコード（7桁）を入力", max_chars=7)

    if st.button("交換部品を予測する"):
        model_manager.load_model(device_type_str)
        model_manager.check_jumpstart_warning(error_code, device_type_str)
        model_manager.predict_parts(error_code)
        
# ----------------------------------------
# 🧯 ダッシュボードタイトル
# ----------------------------------------
st.title("🧯 Failure Count Dashboard 🧯")

# ----------------------------------------
# 📄 故障データの読み込み（JSON形式）
# ----------------------------------------
try:
    df = pd.read_json(data_manager.get_path("output_json"))
    df['発生日'] = pd.to_datetime(df['発生日'], errors='coerce')
    df['月'] = df['発生日'].dt.month
except Exception as e:
    st.error(f"❌ データ読み込みエラー: {e}")
    st.stop()

# ----------------------------------------
# 🏢 ホスト情報からビル名・装置種別の件数を集計
# ----------------------------------------
bill_counts, type_counts = host_resolver.aggregate_counts()

# ----------------------------------------
# 🕒 タイムスタンプ（ファイル名用）
# ----------------------------------------
timestamp = datetime.now().strftime("%Y%m%d")


# ----------------------------------------
# 📊 可視化セクション（折りたたみ可能）
# ----------------------------------------
with st.expander("📉 可視化セクション", expanded=True):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "年間故障件数",
        "年間故障件数×装置種別",
        "月別傾向",
        "月別傾向×装置種別",
        "異常検知",
        "故障件数ランキング",
    ])

    # ----------------------------------------
    # 📅 タブ1：年間故障件数
    # ----------------------------------------
    with tab1:
        st.markdown("### 📅 年間故障件数")
        yearly_counts = df['年度'].value_counts().sort_index()
        year_table = pd.DataFrame({
            '年度': yearly_counts.index,
            '故障件数': yearly_counts.values
        })
        fig = px.bar(year_table, x='年度', y='故障件数', color='故障件数', text='故障件数', title='年間故障件数')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(year_table)
        data_processor.export_excel(year_table, sheet_name='年間故障件数', file_name=f'yearly_output_plotly_{timestamp}.xlsx')

    # ----------------------------------------
    # 🏷️ タブ2：年間故障件数 × 装置種別
    # ----------------------------------------
    with tab2:
        st.markdown("### 🏷️ 年間故障件数 × 装置種別")
        grouped_data = df.groupby(['年度', '装置種別']).size().reset_index(name='故障件数')
        fig = px.bar(grouped_data, x='年度', y='故障件数', color='装置種別', title='年間故障件数×装置種別')
        st.plotly_chart(fig, use_container_width=True)
        pivot_data = grouped_data.pivot(index='年度', columns='装置種別', values='故障件数').fillna(0).astype(int)
        st.dataframe(pivot_data)
        data_processor.export_excel(pivot_data.reset_index(), sheet_name='年間故障件数×装置種別', file_name=f'devices_output_plotly_{timestamp}.xlsx')

    # ----------------------------------------
    # 📈 タブ3：月別傾向（平均・移動平均・トレンド）
    # ----------------------------------------
    with tab3:
        st.markdown("### 📈 月別故障件数の傾向")
        monthly_by_year = df.groupby(['年度', '月']).size().reset_index(name='故障件数')
        monthly_average = monthly_by_year.groupby('月')['故障件数'].mean()
        months = monthly_average.index
        values = monthly_average.values

        # 移動平均とトレンドライン
        moving_avg = pd.Series(values).rolling(window=3, center=True).mean()
        z = np.polyfit(months, values, 1)
        trend = np.poly1d(z)

        # グラフ描画
        fig = go.Figure()
        fig.add_trace(go.Bar(x=months, y=values, name='月別平均故障件数', marker_color='green'))
        fig.add_trace(go.Scatter(x=months, y=moving_avg, mode='lines+markers', name='移動平均（3ヶ月）', line=dict(dash='dash', color='blue')))
        fig.add_trace(go.Scatter(x=months, y=trend(months), mode='lines', name='トレンドライン', line=dict(dash='dot', color='red')))
        fig.update_layout(title='月別平均故障件数の傾向', xaxis_title='月', yaxis_title='故障件数', xaxis=dict(tickmode='linear', tick0=1, dtick=1), yaxis=dict(range=[0, max(values)*1.2]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        #### 📘 補足説明
        - **年度平均ベース**：各月の故障件数を年度ごとに集計し、その平均値を算出しています。
        - **移動平均（3ヶ月）**：前後の月の平均故障件数を滑らかにし、短期的な変動を視覚化します。
        - **トレンドライン**：月別平均故障件数の全体的な傾向を示す直線です。
        """)

        table_data = monthly_average.astype('int')
        st.dataframe(table_data)
        data_processor.export_excel(table_data, sheet_name='月別平均故障件数', file_name=f'monthly_avg_Count_{timestamp}.xlsx')

    # ----------------------------------------
    # 📊 タブ4：月別傾向 × 装置種別
    # ----------------------------------------
    with tab4:
        st.markdown("### 📊 月別傾向 × 装置種別の平均件数")
        monthly_by_type = df.groupby(['月', '装置種別']).size().reset_index(name='件数')
        avg_by_type = monthly_by_type.groupby(['月', '装置種別'])['件数'].mean().reset_index()
        fig = px.line(avg_by_type, x='月', y='件数', color='装置種別', markers=True, title="月別平均故障件数（装置種別）")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(avg_by_type)
        data_processor.export_excel(avg_by_type, sheet_name='月別傾向_装置種別', file_name=f'monthly_type_trend_{timestamp}.xlsx')

    # ----------------------------------------
    # 🚨 タブ5：異常検知（AnomalyDetectorを使用）
    # ----------------------------------------
    with tab5:
        st.markdown("### 🚨 故障件数の異常検知")
        threshold_percent = st.slider("異常と判定する故障率のしきい値（%）", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        aggregation_unit = st.radio("異常判定の集計単位を選択してください", ["年度単位", "指定年度の月間単位"])
        selected_year = None
        if aggregation_unit == "指定年度の月間単位":
            selected_year = st.selectbox("📅 表示する年度を選択", sorted(df['年度'].dropna().unique()))

        result_df = anomaly_detector.detect_anomalies(df, type_counts, threshold_percent, aggregation_unit, selected_year)

        if not result_df.empty:
            filter_option = st.selectbox("表示するデータの種類", ["すべて", "異常のみ", "正常のみ"])
            if filter_option == "異常のみ":
                filtered_df = result_df[result_df["異常"] == "異常"]
            elif filter_option == "正常のみ":
                filtered_df = result_df[result_df["異常"] == "正常"]
            else:
                filtered_df = result_df

            fig = px.scatter(
                filtered_df, x='発生日', y='件数', size='件数', color='異常',
                title=f"異常検知（しきい値: {threshold_percent:.0f}%、単位: {aggregation_unit}）",
                hover_data={'発生日': True, '装置種別': True, '件数': True}
            )
            st.plotly_chart(fig, use_container_width=True)

            anomaly_counts = anomaly_detector.count_anomalies_by_type(type_counts, threshold_percent)
            tab_a, tab_b = st.tabs(["🧮 装置種別ごとの全体数と異常件数", "📋 故障件数と異常判定結果"])
            with tab_a:
                st.dataframe(anomaly_counts)
                data_processor.export_excel(anomaly_counts, sheet_name='異常件数集計', file_name=f'anomaly_summary_{timestamp}.xlsx')
            with tab_b:
                st.dataframe(result_df)
                data_processor.export_excel(result_df, sheet_name='異常検知', file_name=f'anomaly_detection_{timestamp}.xlsx')
        else:
            st.warning("装置種別ごとの異常検知に十分なデータがありません。")

    # ----------------------------------------
    # 🏆 タブ6：装置種別ごとの故障件数ランキング
    # ----------------------------------------
    with tab6:
        st.markdown("### 🏆 装置種別ごとの故障件数ランキング")
        device_ranking = df['装置種別'].value_counts().reset_index()
        device_ranking.columns = ['装置種別', '故障件数']
        fig = px.bar(device_ranking, x='装置種別', y='故障件数', color='故障件数', text='故障件数', title='装置種別ごとの故障件数ランキング')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(device_ranking)
        data_processor.export_excel(device_ranking, sheet_name='装置種別_合計故障件数', file_name=f'devices_ranking_output_plotly_{timestamp}.xlsx')
        
# ----------------------------------------
# 📈 故障率分析セクション
# ----------------------------------------
with st.expander("📈 故障率分析セクション", expanded=True):
    st.markdown("""
    #### 📘 故障率の計算式について ★
    **故障率 = 故障件数 ÷ 全体数**

    - 「全体数」は装置種別またはビル名ごとの母数です。original_db.xlsxを元に母数を求めています。
    - 故障率は 0〜1 の範囲に正規化されており、1 に近いほど故障頻度が高いことを示します。
    - 故障率が高い装置やビルは、保守・交換の優先度が高いと判断できます。
    """)

    tab1, tab2 = st.tabs(["装置種別ごとの故障率推移", "ビルごとの故障率推移"])

    # ----------------------------------------
    # 🛠️ タブ1：装置種別ごとの故障率推移
    # ----------------------------------------
    with tab1:
        type_rate_data = []
        for year in sorted(df['年度'].unique()):
            year_data = df[df['年度'] == year]
            type_group = year_data.groupby('装置種別').size().reset_index(name='故障件数')
            for _, row in type_group.iterrows():
                dtype = row['装置種別']
                failures = row['故障件数']
                total = type_counts[type_counts['d_type_name'] == dtype]['count'].sum()
                rate = failures / total if total > 0 else 0
                rate = min(max(rate, 0), 1)
                type_rate_data.append({
                    '年度': year,
                    '装置種別': dtype,
                    '故障件数': failures,
                    '故障率': rate
                })

        type_rate_df = pd.DataFrame(type_rate_data)

        selected_types = st.multiselect(
            "表示する装置種別を選択（未選択の場合は全表示）",
            options=sorted(type_rate_df['装置種別'].unique()),
            default=sorted(type_rate_df['装置種別'].unique())
        )

        filtered_type_df = type_rate_df[type_rate_df['装置種別'].isin(selected_types)]

        fig_type = px.line(
            filtered_type_df, x='年度', y='故障率', color='装置種別',
            markers=True, title="装置種別ごとの故障率推移"
        )
        fig_type.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("#### 🔍 装置種別ごとの故障率")
        type_tab1, type_tab2 = st.tabs(["年間", "年間平均"])
        with type_tab1:
            st.dataframe(type_rate_df)
            data_processor.export_excel(type_rate_df, sheet_name='年間_故障率', file_name=f'dfType_rate_output_{timestamp}.xlsx')
        with type_tab2:
            type_avg_rate = type_rate_df.groupby('装置種別')[['故障件数', '故障率']].mean().reset_index()
            type_avg_rate = type_avg_rate.sort_values(by='故障率', ascending=False).reset_index(drop=True)
            st.dataframe(type_avg_rate)
            data_processor.export_excel(type_avg_rate, sheet_name='年間平均', file_name=f'dfType_avg_rate_output_{timestamp}.xlsx')

    # ----------------------------------------
    # 🏢 タブ2：ビルごとの故障率推移
    # ----------------------------------------
    with tab2:
        bill_rate_data = []
        for year in sorted(df['年度'].unique()):
            year_data = df[df['年度'] == year]
            bill_group = year_data.groupby('ビル名').size().reset_index(name='故障件数')
            for _, row in bill_group.iterrows():
                bname = row['ビル名']
                failures = row['故障件数']
                total = bill_counts[bill_counts['bill_name'] == bname]['count'].sum()
                rate = failures / total if total > 0 else 0
                rate = min(max(rate, 0), 1)
                bill_rate_data.append({
                    '年度': year,
                    'ビル名': bname,
                    '故障件数': failures,
                    '故障率': rate
                })

        bill_rate_df = pd.DataFrame(bill_rate_data)

        selected_bills = st.multiselect(
            "表示するビル名を選択（未選択の場合は全表示）",
            options=sorted(bill_rate_df['ビル名'].unique()),
            default=sorted(bill_rate_df['ビル名'].unique())
        )

        filtered_bill_df = bill_rate_df[bill_rate_df['ビル名'].isin(selected_bills)]

        fig_bill = px.line(
            filtered_bill_df, x='年度', y='故障率', color='ビル名',
            markers=True, title="ビルごとの故障率推移"
        )
        fig_bill.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_bill, use_container_width=True)

        st.markdown("#### 🏢 ビルごとの故障率")
        bill_tab1, bill_tab2 = st.tabs(["年間", "年間平均"])
        with bill_tab1:
            st.dataframe(bill_rate_df)
            data_processor.export_excel(bill_rate_df, sheet_name='年間_故障率', file_name=f'bill_rate_output_{timestamp}.xlsx')
        with bill_tab2:
            bill_avg_rate = bill_rate_df.groupby('ビル名')[['故障件数', '故障率']].mean().reset_index()
            bill_avg_rate = bill_avg_rate.sort_values(by='故障率', ascending=False).reset_index(drop=True)
            st.dataframe(bill_avg_rate)
            data_processor.export_excel(bill_avg_rate, sheet_name='年間平均', file_name=f'bill_avg_rate_output_{timestamp}.xlsx')
            
# ----------------------------------------
# 🔍 条件検索セクション（フィルター付きテーブル）
# ----------------------------------------
with st.expander("📋 条件検索セクション（フィルター付きテーブル）", expanded=True):
    st.markdown("### 🔍 条件で絞り込み")

    # 年度フィルター（常に表示）
    years = st.multiselect(
        "年度を選択",
        sorted(df['年度'].unique()),
        default=list(range(2009, 2026))
    )
    filtered_df = df[df['年度'].isin(years)]

    # その他のフィルター項目（ドロップダウンで選択）
    filter_options = {
        "ビル名": sorted(df['ビル名'].dropna().unique()),
        "装置種別": sorted(df['装置種別'].dropna().unique()),
        "装置名": sorted(df['装置名'].dropna().unique()),
        "月": sorted(df['月'].dropna().unique()),
        "不具合内容": None  # テキスト検索
    }

    selected_filter = st.selectbox("追加のフィルター項目を選択", options=["なし"] + list(filter_options.keys()))

    if selected_filter != "なし":
        if selected_filter == "不具合内容":
            keyword = st.text_input("不具合内容に含まれるキーワードを入力")
            if keyword:
                filtered_df = filtered_df[filtered_df['不具合内容'].str.contains(keyword, na=False)]
        else:
            selected_values = st.multiselect(
                f"{selected_filter}を選択",
                options=filter_options[selected_filter]
            )
            if selected_values:
                filtered_df = filtered_df[filtered_df[selected_filter].isin(selected_values)]

    # ----------------------------------------
    # 📊 集計と表示
    # ----------------------------------------
    grouped = filtered_df.groupby(
        ['年度', '発生日', 'ビル名', '装置種別', '装置名', '不具合内容']
    ).size().reset_index(name='故障件数')

    # 発生日を文字列に変換（表示用）
    grouped['発生日'] = pd.to_datetime(grouped['発生日'], errors='coerce').dt.strftime('%Y-%m-%d')

    st.dataframe(grouped)

    # 合計件数の表示
    total_failures = grouped['故障件数'].sum()
    st.markdown(f"#### 🧮 検索結果の合計故障件数: {total_failures} 件")

    # Excel出力
    data_processor.export_excel(grouped, sheet_name='故障件数', file_name=f'failedCount_output_{timestamp}.xlsx')
    
# ----------------------------------------
# 📊 統計セクション
# ----------------------------------------
with st.expander("📊 統計セクション", expanded=True):
    # 年度選択フィルター（複数選択可）
    selected_years = st.multiselect(
        "統計対象とする年度を選択（複数選択可）",
        options=sorted(df['年度'].dropna().unique()),
        default=list(range(2009, 2026))
    )

    # 統計量の取得（AnomalyDetector クラスのメソッドを使用）
    stats = anomaly_detector.calculate_failure_statistics(df, selected_years)

    # タイトルと傾向説明の表示
    st.markdown(f"### 🐙 故障件数の統計量（{stats['label']}）")
    st.markdown(f"###### ※ {stats['trend_description']}")

    # メトリクス表示（平均・中央値・最小・最大・標準偏差）
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("平均", f"{stats['mean']} 件")
    col2.metric("中央値", f"{stats['median']} 件")
    col3.metric("最小", f"{stats['min']} 件")
    col4.metric("最大", f"{stats['max']} 件")
    col5.metric("標準偏差", f"{stats['std']}")
    
# ----------------------------------------
# 📅 前年度比較セクション
# ----------------------------------------
with st.expander("📅 前年度比較セクション", expanded=True):
    # 最新年度とその前年度を取得
    current_year = df['年度'].max()
    previous_year = current_year - 1

    # デフォルト月（現在月の前月、1月の場合は12月）
    default_month = datetime.now().month - 1 if datetime.now().month > 1 else 12

    # 月選択（1〜12）
    selected_month = st.selectbox("比較する月を選択", options=list(range(1, 13)), index=default_month - 1)

    # データ抽出
    current_month_data = df[(df['年度'] == current_year) & (df['月'] == selected_month)]
    previous_month_data = df[(df['年度'] == previous_year) & (df['月'] == selected_month)]

    # 故障件数のカウント
    current_month_count = len(current_month_data)
    previous_month_count = len(previous_month_data)

    # 差分と前年比率の表示
    if previous_month_count != 0:
        diff_count = current_month_count - previous_month_count
        percent_change = (diff_count / previous_month_count) * 100
        delta_color = "normal" if diff_count < 0 else "inverse"
        trend_symbol = "📈" if diff_count > 0 else "📉" if diff_count < 0 else "➖"

        st.markdown(f"#### 📅 {selected_month}月の故障件数比較（{previous_year}年 → {current_year}年）")
        st.metric(
            label=f"{current_year}年 {selected_month}月の故障件数",
            value=f"{current_month_count} 件",
            delta=f"{trend_symbol} {diff_count} 件（{percent_change:.1f}%）",
            delta_color=delta_color
        )
    else:
        st.markdown(f"#### ⚠️ {previous_year}年 {selected_month}月のデータが存在しないため、前年比率は計算できません。")