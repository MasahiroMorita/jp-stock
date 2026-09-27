"""
NotionのポートフォリオDBを参照し、保有銘柄の株価情報を更新するスクリプト。

処理内容:
1. 購入日が本日と同じ & 購入株価が空欄
   → 購入株価 = 本日の始値, 購入数 = 1 を書き込む
2. 購入日が記入済み & 売却日が空欄
   → 現在日 = 本日, 現在株価 = 本日の終値,
     現在損益 = (現在株価 - 購入株価) × 購入数 を書き込む

株価は yfinance から取得する。休日などで本日の日足が無い場合は、
最新の取引日の値を代用して警告を表示する。

Usage:
  python stock-price-checker.py            # 実際にNotionへ書き込む
  python stock-price-checker.py --dry-run  # 書き込まずに内容だけ表示
"""

import math
import os
import sys
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

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


def fetch_latest_bar(ticker_code: str) -> dict | None:
    """
    銘柄コードから最新の日足1本を取得し、{"date": 取引日, "open": 始値, "close": 終値} を返す。
    取得できない場合は None。レートリミットには指数バックオフでリトライする。
    """
    delay = 20
    for attempt in range(1, 5):
        try:
            hist = yf.Ticker(f"{ticker_code}.T", session=_get_yf_session()).history(period="5d")
            if hist.empty:
                print(f"⚠️ {ticker_code}: 日足データがありません。")
                return None
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

    ケース1: 購入日が本日 & 購入株価が空欄 → 購入株価 = 本日の始値, 購入数 = 1
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
            updates[COL_QTY] = {"number": 1}
            # ケース2の現在損益計算用に値を反映
            buy_price = bar["open"]
            qty = 1.0
            print(f"📝 {ticker}: 購入株価={bar['open']}（本日の始値）・購入数=1 を記録します。")

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
    bar_cache: dict[str, dict | None] = {}
    updated_count = 0
    for page in pages:
        props = page["properties"]
        if not needs_update(props, today):
            continue

        ticker = get_rich_text_value(props, COL_TICKER)
        if ticker and ticker not in bar_cache:
            bar_cache[ticker] = fetch_latest_bar(ticker)
        bar = bar_cache.get(ticker)

        updates = build_updates(props, bar, today)
        if not updates:
            continue

        if dry_run:
            print(f"  ↪ {ticker}: {updates}")
            continue

        notion.pages.update(page["id"], properties=updates)
        updated_count += 1

    if dry_run:
        print(f"✅ 処理が完了しました（dry-runのため書き込みは行っていません）。")
    else:
        print(f"✅ 処理が完了しました（{updated_count}ページ更新）。")


if __name__ == "__main__":
    main()
