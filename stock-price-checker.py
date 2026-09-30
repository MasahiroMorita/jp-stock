"""
NotionのポートフォリオDBを参照し、保有銘柄の株価情報を更新するスクリプト。

処理内容:
1. 購入日が本日と同じ & 購入株価が空欄
   → 購入株価 = 本日の始値, 購入数 = 1 を書き込む
2. 購入日が記入済み & 売却日が空欄
   → 現在日 = 本日, 現在株価 = 本日の終値,
     現在損益 = (現在株価 - 購入株価) × 購入数 を書き込む
3. 更新したページに株価チャート画像（EMA10/20/50・購入日/購入価格のマーカー・
   購入価格+5.5%(TP)/+12%(TP2)/-3.5%(LC)のライン付き）を貼り付ける。
   ページ内に既存の画像ブロックがあれば削除して置き換える。

株価は yfinance から取得する。休日などで本日の日足が無い場合は、
最新の取引日の値を代用して警告を表示する。

Usage:
  python stock-price-checker.py            # 実際にNotionへ書き込む
  python stock-price-checker.py --dry-run  # 書き込まずに内容だけ表示（チャートはローカルに生成）
"""

import math
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from dotenv import load_dotenv
from notion_client import Client
from curl_cffi import requests as cffi_requests

# .envの読み込み
load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

JST = ZoneInfo("Asia/Tokyo")

# Notion DB のカラム名
COL_TICKER = "銘柄コード"
COL_BUY_DATE = "購入日"
COL_BUY_PRICE = "購入株価"
COL_QTY = "購入数"
COL_SELL_DATE = "売却日"
COL_CURRENT_DATE = "現在日"
COL_CURRENT_PRICE = "現在株価"
COL_CURRENT_PL = "現在損益"


# --- Step 1: 株価取得（yfinance） ---
_YF_SESSION = None


def _get_yf_session():
    """Yahoo Finance 用のブラウザ偽装セッションを返す（stock-analyze.py と同様のBot対策）。"""
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = cffi_requests.Session(impersonate="chrome")
    return _YF_SESSION


def fetch_history(ticker_code: str, buy_date: date | None = None) -> pd.DataFrame | None:
    """
    銘柄の日足履歴を取得する。buy_date があれば購入日の45日前から、
    なければ直近1年分を取得する（チャートに購入マーカーを表示できるようにするため）。
    取得できない場合は None。レートリミットには指数バックオフでリトライする。
    """
    delay = 20
    if buy_date is not None:
        kwargs = {"start": (buy_date - timedelta(days=45)).isoformat()}
    else:
        kwargs = {"period": "1y"}
    for attempt in range(1, 5):
        try:
            hist = yf.Ticker(f"{ticker_code}.T", session=_get_yf_session()).history(**kwargs)
            if hist.empty:
                print(f"⚠️ {ticker_code}: 日足データがありません。")
                return None
            return hist
        except YFRateLimitError:
            if attempt == 4:
                print(f"⚠️ {ticker_code}: レートリミットが続いたため株価取得を中断します。")
                return None
            print(f"⏳ {ticker_code}: Yahoo Financeからレートリミットされました。{delay}秒待ってリトライします（{attempt}/3回目）...")
            time.sleep(delay)
            delay = min(delay * 2, 120)
        except Exception as e:
            print(f"⚠️ {ticker_code}: 株価取得に失敗しました: {e}")
            return None
    return None


def latest_bar_from_history(hist: pd.DataFrame, ticker_code: str) -> dict | None:
    """履歴DataFrameから最新の日足1本を取り出し {"date": 取引日, "open": 始値, "close": 終値} を返す。"""
    last = hist.iloc[-1]
    open_price = float(last["Open"])
    close_price = float(last["Close"])
    if math.isnan(open_price) or math.isnan(close_price):
        print(f"⚠️ {ticker_code}: 始値・終値が取得できませんでした。")
        return None
    return {
        "date": hist.index[-1].date(),
        "open": open_price,
        "close": close_price,
    }


# --- Step 2: Notion ページのプロパティ読み取り ---
def get_date_value(props: dict, key: str) -> date | None:
    """date型プロパティの開始日を返す。空欄の場合は None。"""
    prop = props.get(key)
    if not prop or not prop.get("date"):
        return None
    start = prop["date"].get("start")
    if not start:
        return None
    return date.fromisoformat(start[:10])


def get_number_value(props: dict, key: str) -> float | None:
    """number型プロパティの値を返す。空欄の場合は None。"""
    prop = props.get(key)
    return prop.get("number") if prop else None


def get_rich_text_value(props: dict, key: str) -> str:
    """rich_text型プロパティのテキストを返す。空欄の場合は空文字。"""
    prop = props.get(key)
    if prop and prop.get("rich_text"):
        return prop["rich_text"][0].get("plain_text", "").strip()
    return ""


def fetch_all_pages(notion: Client, ds_id: str) -> list[dict]:
    """データソースから全ページを取得する（ページネーション対応）。"""
    pages: list[dict] = []
    cursor = None
    while True:
        kwargs = {"page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        res = notion.data_sources.query(ds_id, **kwargs)
        pages.extend(res["results"])
        if not res.get("has_more"):
            break
        cursor = res["next_cursor"]
    return pages


# --- Step 3: 更新内容の組み立て ---
def needs_update(props: dict, today: date) -> bool:
    """
    このページが更新対象かどうかを判定する（株価取得前に呼び出して無駄なAPI呼び出しを避ける）。

    ケース1: 購入日が本日 & 購入株価が空欄
    ケース2: 購入日が記入済み & 売却日が空欄
    """
    buy_date = get_date_value(props, COL_BUY_DATE)
    sell_date = get_date_value(props, COL_SELL_DATE)
    if buy_date is None:
        return False
    if buy_date == today and get_number_value(props, COL_BUY_PRICE) is None:
        return True
    return sell_date is None


def build_updates(props: dict, bar: dict | None, today: date) -> dict | None:
    """
    1ページ分の更新内容を組み立てる。更新が不要な場合は None を返す。

    ケース1: 購入日が本日 & 購入株価が空欄 → 購入株価 = 本日の始値,
             購入数は空欄の場合のみ 1 を記録
    ケース2: 購入日記入済み & 売却日が空欄 → 現在日 = 本日, 現在株価 = 本日の終値,
             現在損益 = (現在株価 - 購入株価) × 購入数
    """
    ticker = get_rich_text_value(props, COL_TICKER)
    buy_date = get_date_value(props, COL_BUY_DATE)
    buy_price = get_number_value(props, COL_BUY_PRICE)
    qty = get_number_value(props, COL_QTY)
    sell_date = get_date_value(props, COL_SELL_DATE)

    if not ticker:
        print("⚠️ 銘柄コードが空のページをスキップします。")
        return None

    updates: dict[str, dict] = {}

    # ケース1: 購入日が本日 & 購入株価が空欄
    if buy_date == today and buy_price is None:
        if bar is None:
            print(f"⚠️ {ticker}: 株価を取得できないため購入株価の記録をスキップします。")
        else:
            _warn_if_not_today(ticker, bar, today)
            updates[COL_BUY_PRICE] = {"number": bar["open"]}
            # ケース2の現在損益計算用に値を反映
            buy_price = bar["open"]
            if qty is None:
                # 購入数が空欄の場合のみ 1 を記録する
                qty = 1.0
                updates[COL_QTY] = {"number": 1}
                print(f"📝 {ticker}: 購入株価={bar['open']}（本日の始値）・購入数=1 を記録します。")
            else:
                print(f"📝 {ticker}: 購入株価={bar['open']}（本日の始値）を記録します。購入数は既存の値({qty})のまま。")

    # ケース2: 購入日記入済み & 売却日が空欄
    if buy_date is not None and sell_date is None:
        if bar is None:
            print(f"⚠️ {ticker}: 株価を取得できないため現在株価の更新をスキップします。")
        else:
            _warn_if_not_today(ticker, bar, today)
            updates[COL_CURRENT_DATE] = {"date": {"start": today.isoformat()}}
            updates[COL_CURRENT_PRICE] = {"number": bar["close"]}
            if buy_price is not None and qty is not None:
                current_pl = round((bar["close"] - buy_price) * qty, 2)
                updates[COL_CURRENT_PL] = {"number": current_pl}
                print(f"📈 {ticker}: 現在株価={bar['close']}・現在損益={current_pl} を更新します。")
            else:
                print(f"⚠️ {ticker}: 購入株価または購入数が空のため、現在株価のみ更新します。")

    return updates or None


def _warn_if_not_today(ticker: str, bar: dict, today: date) -> None:
    """取得した日足が本日のものでない場合は警告を出す（市場休みの可能性）。"""
    if bar["date"] != today:
        print(f"⚠️ {ticker}: 本日({today})の日足がありません（市場休みの可能性）。最新の取引日 {bar['date']} の値を使用します。")


# --- Step 4: チャート生成 & ページ画像の置き換え ---
# EMAのカテゴリ色は datavizスキルのライトパレット slot1-3、TP/LCはステータス色を使う
COLOR_EMA10 = "#2a78d6"      # blue
COLOR_EMA20 = "#eb6834"      # orange
COLOR_EMA50 = "#1baf7a"      # aqua
COLOR_TP = "#0ca30c"         # good（利益確定 +5.5%）
COLOR_TP2 = "#006300"        # 成功テキスト緑（利益確定2 +12%）
COLOR_LC = "#d03b3b"         # critical（損切り -3.5%）

TP_PCT = 5.5                 # 利益確定ライン: 購入価格 +5.5%（%表示が基準）
TP2_PCT = 12.0               # 利益確定ライン2: 購入価格 +12%（%表示が基準）
LC_PCT = 3.5                 # 損切りライン: 購入価格 -3.5%（%表示が基準）
TP_RATE = 1 + TP_PCT / 100   # 購入価格からの倍率
TP2_RATE = 1 + TP2_PCT / 100
LC_RATE = 1 - LC_PCT / 100   # 購入価格からの倍率


def generate_portfolio_chart(
    ticker_code: str,
    hist: pd.DataFrame,
    buy_date: date | None,
    buy_price: float | None,
    out_path: str,
) -> bool:
    """
    EMA10/20/50と購入マーカー、TP(+5.5%)/TP2(+12%)/LC(-3.5%)ライン付きの日足チャートをPNGで保存する。
    購入情報が無い場合はマーカー・ラインなしのチャートになる。
    """
    import matplotlib
    matplotlib.use("Agg")  # ディスプレイ非依存のバックエンドで描画する
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import numpy as np

    close = hist["Close"]
    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    addplots = [
        mpf.make_addplot(ema10, color=COLOR_EMA10, width=1.0, label="EMA10"),
        mpf.make_addplot(ema20, color=COLOR_EMA20, width=1.0, label="EMA20"),
        mpf.make_addplot(ema50, color=COLOR_EMA50, width=1.2, label="EMA50"),
    ]

    has_buy_info = buy_date is not None and buy_price is not None
    marker_pos = None
    marker_x = None
    if has_buy_info:
        # 購入日のバーに下向き三角のマーカーを打つ（色はインク色で、形とラベルで識別する）
        marker = pd.Series(np.nan, index=hist.index)
        mask = pd.Index(hist.index.date) == buy_date
        if mask.any():
            marker_pos = hist.index[mask][0]
            # mplfinanceのx軸は日付ではなく整数座標(0..N-1)なので、注釈用に位置を変換しておく
            marker_x = list(hist.index).index(marker_pos)
            marker.loc[marker_pos] = buy_price
            addplots.append(
                mpf.make_addplot(marker, type="scatter", marker="v", markersize=100, color="#0b0b0b")
            )
        else:
            print(f"⚠️ {ticker_code}: 購入日({buy_date})がチャート範囲外のためマーカーは表示されません。")

    title = f"{ticker_code} Daily (EMA10/20/50)"
    if has_buy_info:
        tp = buy_price * TP_RATE
        tp2 = buy_price * TP2_RATE
        lc = buy_price * LC_RATE
        title += (f" | Buy {buy_price:.1f}  TP {tp:.1f} (+{TP_PCT:.1f}%)"
                  f"  TP2 {tp2:.1f} (+{TP2_PCT:.1f}%)  LC {lc:.1f} (-{LC_PCT:.1f}%)")

    fig, axes = mpf.plot(
        hist,
        type="candle",
        style="yahoo",
        title=title,
        ylabel="Price (JPY)",
        volume=True,
        addplot=addplots,
        returnfig=True,
    )

    # TP/LCラインとラベルは axes に直接描画する（凡例の重複を避けるため）
    if has_buy_info:
        ax = axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for price, label, color in [
            (tp, f" TP +{TP_PCT:.1f}% ({tp:.1f})", COLOR_TP),
            (tp2, f" TP2 +{TP2_PCT:.1f}% ({tp2:.1f})", COLOR_TP2),
            (lc, f" LC -{LC_PCT:.1f}% ({lc:.1f})", COLOR_LC),
        ]:
            if y0 <= price <= y1:
                ax.hlines(price, x0, x1, linestyle="--", linewidths=1.2, color=color, alpha=0.9)
                ax.text(x1, price, label, fontsize=8, va="bottom", ha="right", color=color)
            else:
                print(f"ℹ️ {ticker_code}: {label.strip()}のライン({price:.1f})はチャート範囲外のため表示されません。")
        if marker_pos is not None and y0 <= buy_price <= y1:
            ax.text(marker_x, buy_price, f" Buy {buy_price:.1f}", fontsize=8, va="top", color="#0b0b0b")

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def replace_page_chart_image(notion: Client, page_id: str, png_path: str, filename: str) -> None:
    """
    チャートPNGをNotionにアップロードし、新しい画像ブロックをページ末尾に追加した上で、
    ページ内の既存の image ブロックをすべて削除する（古いチャート画像の置き換え）。
    """
    # 1. アップロード（single_part アップロードは send 完了で即利用可能になる）
    upload = notion.file_uploads.create(filename=filename, content_type="image/png")
    with open(png_path, "rb") as f:
        notion.file_uploads.send(upload["id"], file=f)
    file_upload_id = upload["id"]

    # 2. 既存の image ブロックを収集する（ページネーション対応。先に集めてから削除する）
    image_block_ids: list[str] = []
    cursor = None
    while True:
        kwargs = {"page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        res = notion.blocks.children.list(page_id, **kwargs)
        for block in res.get("results", []):
            if block.get("type") == "image":
                image_block_ids.append(block["id"])
        if not res.get("has_more"):
            break
        cursor = res["next_cursor"]

    # 3. 新しい画像ブロックを追加する
    notion.blocks.children.append(
        page_id,
        children=[
            {
                "type": "image",
                "image": {"type": "file_upload", "file_upload": {"id": file_upload_id}},
            }
        ],
    )

    # 4. 古い画像ブロックを削除する
    for block_id in image_block_ids:
        notion.blocks.delete(block_id)


def _effective_buy_price(props: dict, updates: dict) -> float | None:
    """チャート用の購入価格を返す。ケース1で本日の始値を記録した場合はその値を優先する。"""
    buy_price = get_number_value(props, COL_BUY_PRICE)
    if buy_price is None and COL_BUY_PRICE in updates:
        buy_price = updates[COL_BUY_PRICE]["number"]
    return buy_price


def _update_page_chart(notion: Client, page_id: str, ticker: str, hist: pd.DataFrame,
                       props: dict, updates: dict, today: date) -> None:
    """チャート画像を生成してページ内の既存画像と置き換える。失敗しても価格更新の結果には影響させない。"""
    buy_date = get_date_value(props, COL_BUY_DATE)
    buy_price = _effective_buy_price(props, updates)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    try:
        if not generate_portfolio_chart(ticker, hist, buy_date, buy_price, tmp.name):
            return
        replace_page_chart_image(notion, page_id, tmp.name, f"{ticker}_portfolio_{today}.png")
        print(f"🖼️ {ticker}: チャート画像をページに貼り付けました（既存画像は削除済み）。")
    except Exception as e:
        print(f"⚠️ {ticker}: チャート画像の更新に失敗しました: {e}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def _preview_chart(ticker: str, hist: pd.DataFrame, props: dict, updates: dict) -> None:
    """dry-run時: チャートをローカルに生成して貼り付け内容を確認できるようにする。"""
    buy_date = get_date_value(props, COL_BUY_DATE)
    buy_price = _effective_buy_price(props, updates)
    out_path = f"chart_preview_{ticker}.png"
    try:
        if generate_portfolio_chart(ticker, hist, buy_date, buy_price, out_path):
            print(f"  🖼️ {ticker}: チャートを {out_path} に生成しました（本番では既存画像を削除して貼り付けます）。")
    except Exception as e:
        print(f"⚠️ {ticker}: チャート生成に失敗しました: {e}")


# --- メイン実行処理 ---
def main():
    if not NOTION_API_KEY or not NOTION_DATABASE_ID:
        print("❌ .env に NOTION_API_KEY / NOTION_DATABASE_ID を設定してください。")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv

    notion = Client(auth=NOTION_API_KEY)

    # 1. データベースのデータソースIDを取得（新しいNotion APIではクエリにデータソースIDを使う）
    db = notion.databases.retrieve(NOTION_DATABASE_ID)
    data_sources = db.get("data_sources") or []
    if not data_sources:
        print("❌ データベースにデータソースが見つかりません。")
        sys.exit(1)
    ds_id = data_sources[0]["id"]

    today = datetime.now(JST).date()
    mode = "（dry-run: 書き込みません）" if dry_run else ""
    print(f"🔍 ポートフォリオDBを確認しています{mode}（本日: {today}）...")

    # 2. 全ページを取得
    pages = fetch_all_pages(notion, ds_id)
    print(f"📄 {len(pages)}ページを確認します。")

    # 3. ページごとに更新（株価取得は更新対象のページのみ行う）
    # チャートに購入マーカーを必ず表示できるよう、更新対象ページの銘柄ごとの最古の購入日を先に集計する
    earliest_buy_dates: dict[str, date | None] = {}
    for page in pages:
        props = page["properties"]
        if not needs_update(props, today):
            continue
        ticker = get_rich_text_value(props, COL_TICKER)
        if not ticker:
            continue
        buy_date = get_date_value(props, COL_BUY_DATE)
        prev = earliest_buy_dates.get(ticker)
        if prev is None or (buy_date is not None and buy_date < prev):
            earliest_buy_dates[ticker] = buy_date

    hist_cache: dict[str, pd.DataFrame | None] = {}
    updated_count = 0
    for page in pages:
        props = page["properties"]
        if not needs_update(props, today):
            continue

        ticker = get_rich_text_value(props, COL_TICKER)
        if ticker and ticker not in hist_cache:
            hist_cache[ticker] = fetch_history(ticker, earliest_buy_dates.get(ticker))
        hist = hist_cache.get(ticker)
        bar = latest_bar_from_history(hist, ticker) if hist is not None else None

        updates = build_updates(props, bar, today)
        if not updates:
            continue

        if dry_run:
            print(f"  ↪ {ticker}: {updates}")
            if hist is not None:
                _preview_chart(ticker, hist, props, updates)
            continue

        notion.pages.update(page["id"], properties=updates)
        updated_count += 1

        # チャート画像の更新（価格更新後のベストエフォート処理）
        if hist is not None:
            _update_page_chart(notion, page["id"], ticker, hist, props, updates, today)

    if dry_run:
        print(f"✅ 処理が完了しました（dry-runのため書き込みは行っていません）。")
    else:
        print(f"✅ 処理が完了しました（{updated_count}ページ更新）。")


if __name__ == "__main__":
    main()
