import logging

import pandas as pd
import pandas_ta as ta
import yfinance as yf

# yfinanceの404等のエラーログを抑制（上場廃止銘柄などのノイズ対策）
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def get_all_jpx_tickers():
    """JPX(日本取引所グループ)公式サイトから全上場銘柄のティッカーリスト(.T付)を取得する[cite: 1, 2]"""
    print("📡 JPXから全上場銘柄リストを取得中...")
    jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"

    try:
        df_jpx = pd.read_excel(jpx_url)
        code_column = "コード"
        tickers = [
            f"{str(code).strip().zfill(4)}.T"
            for code in df_jpx[code_column]
            if len(str(code).strip()) == 4
        ]
        print(f"✅ 全 {len(tickers)} 銘柄のリストを取得しました。")
        return tickers
    except Exception as e:
        print(f"❌ 銘柄リスト取得エラー: {e}")
        return ["7203.T", "6758.T", "9983.T", "6861.T", "8306.T"]


def is_analyst_buy_recommendation(ticker):
    """Yahoo Financeからアナリスト評価を取得し、Buy以上（buy / strong_buy）かチェックする"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # recommendationKey（例: 'strong_buy', 'buy', 'hold', 'sell' など）
        rec_key = info.get("recommendationKey", "").lower()

        # recommendationMean (1.0 = Strong Buy, 2.0 = Buy, 3.0 = Hold...)
        rec_mean = info.get("recommendationMean", None)

        # 'strong_buy' または 'buy'、あるいは平均スコアが 2.2 以下（Buy寄り）であるか判定
        is_buy_key = rec_key in ["strong_buy", "buy"]
        is_buy_mean = (rec_mean is not None) and (rec_mean <= 2.2)

        if is_buy_key or is_buy_mean:
            return True, rec_key if rec_key else f"Mean: {rec_mean}"
        else:
            return False, rec_key if rec_key else "評価なし/Hold以下"
    except Exception:
        return False, "取得不可"


def screen_breakout_stocks(ticker_list, max_stocks=None, chunk_size=100):
    """テクニカル条件（EMAパーフェクトオーダー＋10日高値更新）に加え、

    アナリスト判断が「Buy以上」の銘柄のみを抽出する[cite: 1]
    """
    matched_stocks = []

    target_tickers = (
        ticker_list[:max_stocks] if max_stocks else ticker_list
    )
    total = len(target_tickers)

    print(f"\n=== 全銘柄スクリーニング開始 (対象: {total}銘柄) ===")

    for chunk_start in range(0, total, chunk_size):
        chunk = target_tickers[chunk_start:chunk_start + chunk_size]

        try:
            # チャンク単位で一括ダウンロード（yfinance内部で並列化）
            data = yf.download(
                chunk,
                period="100d",
                interval="1d",
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            continue

        for offset, ticker in enumerate(chunk):
            try:
                df = data[ticker].copy()
            except Exception:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=["Close"])
            if len(df) < 50:
                continue

            df["EMA10"] = ta.ema(df["Close"], length=10)
            df["EMA20"] = ta.ema(df["Close"], length=20)
            df["EMA50"] = ta.ema(df["Close"], length=50)
            df["Highest10"] = df["High"].shift(1).rolling(window=10).max()

            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # 1. テクニカル条件の判定
            # 条件A: EMA10 > EMA20 > EMA50 ＆ 終値 > EMA10[cite: 1]
            is_trend_up = (
                (latest["EMA10"] > latest["EMA20"])
                and (latest["EMA20"] > latest["EMA50"])
                and (latest["Close"] > latest["EMA10"])
            )

            # 条件B: 10日高値更新ブレイクアウト[cite: 1]
            is_breakout = (prev["Close"] <= latest["Highest10"]) and (
                latest["Close"] > latest["Highest10"]
            )

            # 2. テクニカル条件クリア時のみ、アナリスト評価の判定を実行（API通信回数を最小化）
            if is_trend_up and is_breakout:
                is_buy, rec_detail = is_analyst_buy_recommendation(ticker)

                if is_buy:
                    matched_stocks.append({
                        "ticker": ticker.replace(".T", ""),
                        "close": round(float(latest["Close"]), 1),
                        "recommendation": rec_detail,
                        "highest10": round(float(latest["Highest10"]), 1),
                    })
                    print(
                        f"  👉 【BUY判定＆アナリスト高評価】 銘柄: {ticker.replace('.T', '')} | 評価: {rec_detail} | 終値: {latest['Close']:.1f}円"
                    )

        done = min(chunk_start + chunk_size, total)
        print(f"進捗: {done}/{total} 銘柄処理中... (検出数: {len(matched_stocks)})")

    print("\n=== スクリーニング完了 ===")
    return pd.DataFrame(matched_stocks)


# ---------------------------------------------------------
# 実行部
# ---------------------------------------------------------
if __name__ == "__main__":
    ticker_list = get_all_jpx_tickers()
    result_df = screen_breakout_stocks(ticker_list, max_stocks=None)

    if not result_df.empty:
        print("\n【アナリスト評価Buy以上 ＆ テクニカル条件適合銘柄】")
        print(result_df.to_string(index=False))
        result_df.to_csv("analyst_buy_breakout_signals.csv", index=False)
        print("\n💾 抽出結果を 'analyst_buy_breakout_signals.csv' に保存しました。")
    else:
        print("\n本日、全条件（テクニカル＋アナリストBuy以上）を満たす銘柄はありませんでした。")