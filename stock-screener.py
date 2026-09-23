import logging
import time
import pandas as pd
import pandas_ta as ta
import yfinance as yf

# yfinance の「No data found / Failed download」ログを抑止
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def get_all_jpx_tickers():
    """JPX(日本取引所グループ)公式サイトから全上場銘柄のティッカーリスト(.T付)を取得する[cite: 1, 2]"""
    print("📡 JPXから全上場銘柄リストを取得中...")
    jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"

    try:
        code_column = "コード"
        df_jpx = pd.read_excel(jpx_url, dtype={code_column: str})
        tickers = [
            f"{code.strip()}.T"
            for code in df_jpx[code_column].dropna()
            if len(code.strip()) == 4
        ]
        print(f"✅ 全 {len(tickers)} 銘柄のリストを取得しました。")
        return tickers
    except Exception as e:
        print(f"❌ 銘柄リスト取得エラー: {e}")
        # フォールバック用の主要銘柄リスト
        return [
            "7203.T",
            "6758.T",
            "9983.T",
            "6861.T",
            "8306.T",
            "7970.T",
            "6501.T",
        ]


def get_analyst_rating(ticker):
    """Yahoo Finance のアナリストレコメンデーションを取得する (例: 'strong_buy', 'buy', 'hold')"""
    try:
        return yf.Ticker(ticker).info.get("recommendationKey")
    except Exception:
        return None


def screen_pullback_stocks(ticker_list, max_stocks=None):
    """押し目買い（EMA20反発）の条件を満たす銘柄を抽出"""
    matched_stocks = []
    total = (
        len(ticker_list)
        if max_stocks is None
        else min(len(ticker_list), max_stocks)
    )

    print(f"\n=== 押し目買いスクリーニング開始 (対象: {total}銘柄) ===")

    target_tickers = (
        ticker_list[:max_stocks] if max_stocks else ticker_list
    )

    for i, ticker in enumerate(target_tickers, 1):
        if i % 100 == 0 or i == total:
            print(f"進捗: {i}/{total} 銘柄処理中... (検出数: {len(matched_stocks)})")

        try:
            # 日足データの取得 (EMA50計算のため最低50日以上が必要)
            df = yf.download(
                ticker, period="100d", interval="1d", progress=False
            )

            if len(df) < 50:
                continue

            # MultiIndexカラム対策
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # EMA（指数平滑移動平均）の計算
            df["EMA10"] = ta.ema(df["Close"], length=10)
            df["EMA20"] = ta.ema(df["Close"], length=20)
            df["EMA50"] = ta.ema(df["Close"], length=50)

            latest = df.iloc[-1]

            # --------------------------------------------------
            # 1. テクニカル条件判定（押し目買い）
            # --------------------------------------------------
            # 条件A: 上昇パーフェクトオーダー (EMA10 > EMA20 > EMA50)
            is_trend_up = (latest["EMA10"] > latest["EMA20"]) and (
                latest["EMA20"] > latest["EMA50"]
            )

            # 条件B: 安値がEMA20にタッチ/割込みし、終値ではEMA20の上で反発
            is_pullback_bounce = (latest["Low"] <= latest["EMA20"]) and (
                latest["Close"] > latest["EMA20"]
            )

            if is_trend_up and is_pullback_bounce:
                # アナリスト評価の確認（BUY / STRONG BUY のみ抽出）
                rating = get_analyst_rating(ticker)
                if rating not in ("buy", "strong_buy"):
                    print(
                        f"  ⏭️  {ticker.replace('.T', '')} をスキップ: アナリスト評価がBUY/STRONG BUY以外 (評価: {rating or 'なし'})"
                    )
                    time.sleep(0.05)
                    continue

                # 前日比変化率の計算
                prev_close = df.iloc[-2]["Close"]
                change_pct = (
                    (latest["Close"] - prev_close) / prev_close
                ) * 100

                matched_stocks.append({
                    "ticker": ticker.replace(".T", ""),
                    "close": round(float(latest["Close"]), 1),
                    "change_pct": round(float(change_pct), 2),
                    "ema10": round(float(latest["EMA10"]), 1),
                    "ema20": round(float(latest["EMA20"]), 1),
                    "ema50": round(float(latest["EMA50"]), 1),
                    "volume": int(latest["Volume"]),
                    "analyst_rating": rating.upper(),
                })

                print(
                    f"  👉 【押し目買いシグナル検出】 銘柄: {ticker.replace('.T', '')} | 終値: {latest['Close']:.1f}円 (前日比: {change_pct:+.2f}%) | アナリスト評価: {rating.upper()}"
                )

        except Exception:
            continue

        # サーバー負荷軽減用のウェイト
        time.sleep(0.05)

    print("\n=== スクリーニング完了 ===")
    return pd.DataFrame(matched_stocks)


# ---------------------------------------------------------
# 実行部
# ---------------------------------------------------------
if __name__ == "__main__":
    # 全銘柄リストの取得
    ticker_list = get_all_jpx_tickers()

    # テスト実行の場合は max_stocks=100 などで動作確認してください
    result_df = screen_pullback_stocks(ticker_list, max_stocks=None)

    if not result_df.empty:
        print("\n【押し目買い（EMA20反発）シグナル検出銘柄】")
        print(result_df.to_string(index=False))

        # CSVに保存
        result_df.to_csv("pullback_buy_signals.csv", index=False)
        print("\n💾 抽出結果を 'pullback_buy_signals.csv' に保存しました。")
    else:
        print("\n本日、押し目買い条件を満たす銘柄はありませんでした。")