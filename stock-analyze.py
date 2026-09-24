import sys
import os
import re
import math
import json
import tempfile
import time
from datetime import date, timedelta
from urllib.parse import parse_qs, unquote, urljoin, urlparse
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from dotenv import load_dotenv
from notion_client import Client
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

# .envの読み込み
load_dotenv()

NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

# DeepSeek API の設定
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # deepseek-chat / deepseek-reasoner
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_TIMEOUT = 300  # 推論待ちのタイムアウト（秒）

# 定量スコアがこの値以下の場合は AI による定性評価をスキップする
AI_EVAL_SKIP_THRESHOLD = 22


# --- Step 1: データ収集（銘柄データ ＋ 市場全体地合いデータ） ---
_YF_SESSION = None


def _get_yf_session():
    """
    Yahoo Finance 用のブラウザ偽装セッションを返す（全銘柄・地合いデータで使い回す）。
    CI等のデータセンターIPでは TLS フィンガープリントの違いから 429/401 の
    Bot 判定を受けやすく、curl_cffi の chrome 偽装セッションで回避できる場合がある。
    """
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = cffi_requests.Session(impersonate="chrome")
    return _YF_SESSION


def _fetch_info_with_retry(stock, max_retries: int = 5) -> dict:
    """
    stock.info を取得する。Yahoo Finance のレートリミット(429)には
    指数バックオフ（20s→40s→80s→120s）でリトライする。
    """
    delay = 20
    for attempt in range(1, max_retries + 1):
        try:
            info = stock.info
            if not info.get("shortName") and not (info.get("currentPrice") or info.get("previousClose")):
                print("⚠️ Yahoo Finance から基本情報を取得できていない可能性があります（定量スコアを疑ってください）。")
            return info
        except YFRateLimitError:
            if attempt == max_retries:
                raise
            print(f"⏳ Yahoo Finance からレートリミットされました。{delay}秒待ってリトライします（{attempt}/{max_retries - 1}回目）...")
            time.sleep(delay)
            delay = min(delay * 2, 120)


def _pct_change(latest: float, past: float | None) -> float | None:
    """latest が past から何%変動したかを返す。past が無効値の場合は None。"""
    if not past:
        return None
    return (latest / past - 1) * 100


def _calc_rsi(close, period: int = 14) -> float | None:
    """終値のSeriesからRSI（単純移動平均ベース）を算出する。データ不足時は None。"""
    import pandas as pd
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if pd.isna(gain) or pd.isna(loss):
        return None
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def fetch_stock_technicals(ticker_code: str) -> dict:
    """
    yfinance の1年分日足から、押し目判定に使う定量テクニカル指標を算出する。
    データ取得に失敗した場合は警告を出して空dictを返す（分析自体は継続する）。
    """
    try:
        hist = yf.Ticker(f"{ticker_code}.T", session=_get_yf_session()).history(period="1y")
    except Exception as e:
        print(f"⚠️ テクニカル指標の日足取得に失敗しました（スキップします）: {e}")
        return {}
    if hist.empty or len(hist) < 30:
        print("⚠️ 日足データが不足しているためテクニカル指標をスキップします。")
        return {}

    close = hist["Close"]
    latest = float(close.iloc[-1])
    high_52w = float(hist["High"].max())
    low_52w = float(hist["Low"].min())
    return {
        "price": latest,
        "sma25": float(close.rolling(window=25).mean().iloc[-1]),
        "sma75": float(close.rolling(window=75).mean().iloc[-1]),
        "sma25_dev": _pct_change(latest, float(close.rolling(window=25).mean().iloc[-1])),
        "sma75_dev": _pct_change(latest, float(close.rolling(window=75).mean().iloc[-1])),
        "rsi14": _calc_rsi(close),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "drawdown_from_high": _pct_change(latest, high_52w),
        "change_1mo": _pct_change(latest, float(close.iloc[-22])) if len(close) >= 22 else None,
        "change_3mo": _pct_change(latest, float(close.iloc[-64])) if len(close) >= 64 else None,
    }


def fetch_stock_and_market_data(ticker_code: str):
    """
    対象銘柄データと日経平均(地合い)データを取得する
    """
    symbol = f"{ticker_code}.T"
    stock = yf.Ticker(symbol, session=_get_yf_session())
    try:
        info = _fetch_info_with_retry(stock)
    except YFRateLimitError:
        print("❌ Yahoo Finance のレートリミットが継続しています。")
        print("   しばらく（数十分〜数時間）待ってから再実行してください。")
        sys.exit(1)

    # 1. 個別銘柄データ
    data = {
        "ticker": ticker_code,
        "name": info.get("shortName", f"銘柄コード {ticker_code}"),
        "date": date.today().isoformat(),
        "sector": info.get("sector", "不明"),
        "price": info.get("currentPrice") or info.get("previousClose", 0),
        "per": info.get("trailingPE"),
        "forward_per": info.get("forwardPE"),
        "pbr": info.get("priceToBook"),
        "eps_growth": info.get("earningsGrowth"),
        "market_cap": info.get("marketCap", 0),
        "volume": info.get("volume", 0),
        "avg_volume": info.get("averageVolume", 0),
    }

    # 2. 地合いデータ（日経平均 ^N225 の25日移動平均線チェック）
    market_data = {"nikkei_above_sma25": False, "nikkei_price": 0, "sma25": 0}
    try:
        n225 = yf.Ticker("^N225", session=_get_yf_session())
        hist = n225.history(period="3mo")
        if len(hist) >= 25:
            hist['SMA25'] = hist['Close'].rolling(window=25).mean()
            latest_close = hist['Close'].iloc[-1]
            latest_sma25 = hist['SMA25'].iloc[-1]

            market_data["nikkei_price"] = latest_close
            market_data["sma25"] = latest_sma25
            market_data["nikkei_above_sma25"] = latest_close > latest_sma25
            market_data["nikkei_rsi14"] = _calc_rsi(hist['Close'])
            market_data["nikkei_change_1mo"] = _pct_change(float(latest_close), float(hist['Close'].iloc[-22])) if len(hist) >= 22 else None
    except Exception as e:
        print(f"⚠️ 地合いデータ取得でエラー（スキップします）: {e}")

    return data, market_data


# --- Step 1 補助: 株探(kabutan.jp) 決算速報ニュース取得 ---
KABUTAN_BASE_URL = "https://kabutan.jp"
KABUTAN_KESSAN_LIMIT = 3  # 取得する決算速報の最大件数
KABUTAN_INDUSTRY_RANKING_PAGES = (1, 2, 3)  # 業種別ランキングのページ（全33業種分）
# 業種別ランキングのキャッシュファイル。取得結果は日付付きで保存し、同日の再実行では
# Web取得をせずキャッシュを使い回す（ analyze_signals.py が本スクリプトを銘柄数分起動するため、
# 株探への大量リクエスト集中を避けるための対策）。
KABUTAN_INDUSTRY_RANKING_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kabutan_industry_ranking.json"
)


def fetch_kabutan_kessan_news(ticker_code: str, limit: int = KABUTAN_KESSAN_LIMIT) -> list[dict]:
    """
    株探の「決算速報」タブ(nmode=2)から直近の記事を最大 `limit` 件抽出し、
    各記事ページの本文テキストまで取得する。
    一覧の取得自体に失敗した場合は警告を出して空リストを返す（分析自体は継続する）。
    """
    news_list_url = f"{KABUTAN_BASE_URL}/stock/news?code={ticker_code}&nmode=2"
    try:
        resp = cffi_requests.get(news_list_url, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        items = []
        for row in soup.select("table.s_news_list tr"):
            a = row.select_one("a[href]")
            t = row.select_one("time")
            if not a or not a.get_text(strip=True):
                continue
            items.append({
                "datetime": t.get("datetime") if t else "",
                "title": a.get_text(strip=True),
                "url": urljoin(KABUTAN_BASE_URL, a["href"]),
            })
            if len(items) >= limit:
                break
    except Exception as e:
        print(f"⚠️ 株探 決算速報一覧の取得に失敗しました（スキップします）: {e}")
        return []

    # 各記事ページの本文テキストを取得する
    news = []
    for item in items:
        try:
            resp = cffi_requests.get(item["url"], impersonate="chrome", timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            body = soup.select_one("article div.body")
            if body is not None:
                lines = [ln.strip() for ln in body.get_text("\n", strip=True).splitlines()]
                item["body"] = "\n".join(ln for ln in lines if ln)
        except Exception as e:
            print(f"⚠️ 記事本文の取得に失敗しました（タイトルのみ保持します）: {item['title']} ({e})")
            item["body"] = ""
        news.append(item)
        time.sleep(1)  # 株探への負荷軽減のための待機

    return news


def _parse_num(text: str) -> float | None:
    """'1,234.5' や '%' 付きの表示文字列を float に変換する。空・不正値（nan/inf含む）は None。"""
    try:
        value = float(text.replace(",", "").replace("%", "").strip())
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_kabutan_industry_ranking_cache() -> dict | None:
    """業種別ランキングのキャッシュファイルを読み込む。存在しない・破損している場合は None。"""
    try:
        with open(KABUTAN_INDUSTRY_RANKING_CACHE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("industries"), list):
        return None
    return payload


def _save_kabutan_industry_ranking_cache(fetched_date: str, industries: list[dict]) -> None:
    """業種別ランキングを日付付きでキャッシュファイルに保存する。保存失敗時は警告のみ出して継続する。"""
    try:
        with open(KABUTAN_INDUSTRY_RANKING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched_date": fetched_date, "industries": industries}, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"⚠️ 業種別ランキングのキャッシュ保存に失敗しました（スキップします）: {e}")


def fetch_kabutan_industry_ranking(pages: tuple = KABUTAN_INDUSTRY_RANKING_PAGES) -> list[dict]:
    """
    株探の「業種別ランキング」(mode=9_1)を `pages` ページ分取得し、
    table.stock_table から各業種の平均株価・前日比（増減）・PER・PBR・利回りを抽出する。
    ページ取得に失敗した場合は警告を出してそのページをスキップする（分析自体は継続する）。

    取得結果は日付付きでキャッシュファイルに保存し、同日中の再実行ではWeb取得せずキャッシュを
    使い回す。Web取得に失敗した場合は、日付が異なる（古い）キャッシュでもフォールバックとして使う。
    """
    today = date.today().isoformat()
    cache = _load_kabutan_industry_ranking_cache()
    if cache is not None and cache.get("fetched_date") == today:
        industries = cache["industries"]
        print(f"💾 業種別ランキングはキャッシュファイルから読み込みました（{today} 取得分・全{len(industries)}業種）")
        return industries

    industries = []
    for page in pages:
        url = f"{KABUTAN_BASE_URL}/warning/?mode=9_1&page={page}"
        try:
            resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"⚠️ 株探 業種別ランキング(page={page})の取得に失敗しました（スキップします）: {e}")
            continue

        for row in soup.select("table.stock_table tr")[1:]:  # 先頭行はヘッダー
            cells = row.select("td, th")
            if len(cells) < 11:
                continue
            industries.append({
                "code": cells[0].get_text(strip=True),
                "name": cells[1].get_text(strip=True),
                "count": int(_parse_num(cells[2].get_text(strip=True)) or 0),
                "price": _parse_num(cells[4].get_text(strip=True)),
                "change": _parse_num(cells[6].get_text(strip=True)),
                "change_pct": _parse_num(cells[7].get_text(strip=True)),
                "per": _parse_num(cells[8].get_text(strip=True)),
                "pbr": _parse_num(cells[9].get_text(strip=True)),
                "dividend_yield": _parse_num(cells[10].get_text(strip=True)),
            })
        time.sleep(1)  # 株探への負荷軽減のための待機

    if industries:
        _save_kabutan_industry_ranking_cache(today, industries)
        return industries

    if cache is not None:
        industries = cache["industries"]
        print(f"⚠️ 業種別ランキングの取得に失敗したため、キャッシュ（{cache.get('fetched_date', '日付不明')} 取得分・全{len(industries)}業種）を使用します。")
        return industries
    return []


def fetch_kabutan_next_earnings_date(ticker_code: str) -> dict | None:
    """
    株探の「業績・財務推移」ページにある過去の四半期・半期決算の発表日から、
    次回決算発表予定日を推定する（多くの銘柄はほぼ同じ時期に発表するため）。
    取得に失敗した場合は警告を出して None を返す（分析自体は継続する）。
    """
    url = f"{KABUTAN_BASE_URL}/stock/finance?code={ticker_code}"
    try:
        resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ 株探 決算推移ページの取得に失敗しました（スキップします）: {e}")
        return None

    # (月, 日) -> 実績の生文字列一覧。「予」行の公表日は結果発表日ではないため除外する
    history: dict[tuple[int, int], list[str]] = {}
    for table in soup.select("table"):
        head = table.select_one("thead")
        if head is None or "発表日" not in head.get_text():
            continue
        for tr in table.select("tr"):
            cells = tr.select("th, td")
            if len(cells) < 2:
                continue
            period = cells[0].get_text(strip=True)
            reported = cells[-1].get_text(strip=True)
            if "予" in period or not re.search(r"\d{2}\.\d{2}-\d{2}", period):
                continue
            m = re.fullmatch(r"\d{2}/(\d{2})/(\d{2})", reported)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                history.setdefault(key, [])
                if reported not in history[key]:
                    history[key].append(reported)
    if not history:
        print("⚠️ 過去の決算発表日が見つからなかったため、次回決算日の推定をスキップします。")
        return None

    today = date.today()
    best = None  # (候補日, 実績文字列一覧)
    for (month, day), raws in history.items():
        for year in (today.year, today.year + 1):
            try:
                cand = date(year, month, day)
            except ValueError:
                continue
            # 直前に通過した候補は発表済みとみなして除外する
            if cand < today - timedelta(days=7):
                continue
            if best is None or cand < best[0]:
                best = (cand, raws)
    if best is None:
        print("⚠️ 次回決算日の候補が見つからなかったため、推定をスキップします。")
        return None

    estimated, raws = best
    return {
        "estimated_date": estimated.isoformat(),
        "days_until": (estimated - today).days,
        "history": sorted(raws)[-3:],
    }


# --- Step 1 補助: セクター関連ニュース検索（kabutan / Yahoo!ファイナンス） ---
# Yahoo Finance の sector は英語（GICS系: "Industrials" 等）で返るため、
# 検索クエリ用に日本語の業種キーワードへ変換する。
_SECTOR_JA_QUERY_TERMS = {
    "Basic Materials": "素材",
    "Communication Services": "通信",
    "Consumer Cyclical": "消費",
    "Consumer Defensive": "食品",
    "Energy": "資源",
    "Financial Services": "金融",
    "Healthcare": "医薬品",
    "Industrials": "機械",
    "Real Estate": "不動産",
    "Technology": "電気機器",
    "Utilities": "電力",
}
SECTOR_SEARCH_MAX_RESULTS = 3  # 1クエリあたりの取得件数
SECTOR_SEARCH_QUERIES = (
    "site:kabutan.jp {sector} セクター",
    "site:finance.yahoo.co.jp {sector} セクター 騰落率",
)
# セクター検索結果のキャッシュファイル。取得結果は日付付きで保存し、同日の再実行では
# Web取得をせずキャッシュを使い回す（analyze_signals.py が本スクリプトを銘柄数分起動するため、
# 検索エンジンへのリクエスト集中を避けるための対策）。
SECTOR_SEARCH_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sector_trend_search_cache.json"
)


def _load_sector_search_cache() -> dict | None:
    """セクター検索結果のキャッシュファイルを読み込む。存在しない・破損している場合は None。"""
    try:
        with open(SECTOR_SEARCH_CACHE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("sectors"), dict):
        return None
    return payload


def _save_sector_search_cache(fetched_date: str, sectors: dict[str, list[dict]]) -> None:
    """セクター検索結果を日付付きでキャッシュファイルに保存する。保存失敗時は警告のみ出して継続する。"""
    try:
        with open(SECTOR_SEARCH_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"fetched_date": fetched_date, "sectors": sectors}, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"⚠️ セクター検索結果のキャッシュ保存に失敗しました（スキップします）: {e}")


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGoのリダイレクトURL(//duckduckgo.com/l/?uddg=...)から実URLを取り出す。"""
    parsed = urlparse(href)
    if "duckduckgo.com" not in parsed.netloc:
        return href
    return unquote(parse_qs(parsed.query).get("uddg", [href])[0])


def _search_sector_news_yahoo(query: str) -> list[dict]:
    """
    Yahoo! JAPAN検索で `query` を検索し、kabutan / Yahoo!ファイナンス配下の結果を
    最大 SECTOR_SEARCH_MAX_RESULTS 件抽出する。失敗時は警告を出して空リストを返す。
    """
    try:
        resp = cffi_requests.get(
            "https://search.yahoo.co.jp/search",
            params={"p": query, "ei": "UTF-8"},
            impersonate="chrome",
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ Yahoo!検索の実行に失敗しました（スキップします）: {e}")
        return []

    items = []
    for card in soup.select("div.sw-CardBase"):
        a = card.select_one("a.sw-Card__titleInner")
        title = card.select_one("h3.sw-Card__titleMain")
        if not a or not a.get("href") or not title:
            continue
        url = a["href"]
        host = urlparse(url).netloc
        if "kabutan" not in host and "yahoo" not in host:
            continue  # site:指定が効かない結果や広告等を除外
        items.append({
            "source": "kabutan.jp" if "kabutan" in host else "finance.yahoo.co.jp",
            "title": title.get_text(strip=True),
            "snippet": "",
            "url": url,
        })
        if len(items) >= SECTOR_SEARCH_MAX_RESULTS:
            break
    return items


def _search_sector_news_ddg(query: str) -> list[dict]:
    """
    DuckDuckGo(html)で `query` を検索し、kabutan / Yahoo!ファイナンス配下の結果を
    最大 SECTOR_SEARCH_MAX_RESULTS 件抽出する（Yahoo!検索のフォールバック用）。
    失敗時は警告を出して空リストを返す。
    """
    try:
        resp = cffi_requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            impersonate="chrome",
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ DuckDuckGo検索の実行に失敗しました（スキップします）: {e}")
        return []

    if not soup.select("div.result"):
        print(f"⚠️ DuckDuckGo検索の結果が取得できませんでした（ボット判定等の可能性）: {query}")
        return []

    items = []
    for div in soup.select("div.result"):
        a = div.select_one("a.result__a")
        snip = div.select_one("a.result__snippet")
        if not a or not a.get("href"):
            continue
        url = _decode_ddg_href(a["href"])
        host = urlparse(url).netloc
        if "kabutan" not in host and "yahoo" not in host:
            continue
        items.append({
            "source": "kabutan.jp" if "kabutan" in host else "finance.yahoo.co.jp",
            "title": a.get_text(strip=True),
            "snippet": (snip.get_text(strip=True) if snip else "")[:300],
            "url": url,
        })
        if len(items) >= SECTOR_SEARCH_MAX_RESULTS:
            break
    return items


def _fetch_page_excerpt(url: str, max_chars: int = 400) -> str:
    """
    検索結果ページの本文先頭をスニペット代わりに抜粋する（検索エンジンがスニペットを
    返さない場合の補完）。サイトごとに本文コンテナが異なるため優先度付きで試し、
    本文が取れない場合は meta description で補完する。失敗時は空文字を返す。
    """
    try:
        resp = cffi_requests.get(url, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return ""
    node = None
    for sel in ("article div.body", "div#main", "article", "div.body", "main"):
        node = soup.select_one(sel)
        if node is not None:
            break
    if node is None:
        node = soup.body or soup
    lines = [ln.strip() for ln in node.get_text("\n", strip=True).splitlines() if ln.strip()]
    excerpt = "\n".join(lines)[:max_chars].strip()
    # 本文が取れなかった場合（JS描画ページ等）は meta description で補完する
    if len(excerpt) < 40:
        meta = soup.select_one("meta[name='description']")
        if meta and meta.get("content"):
            excerpt = meta["content"].strip()[:max_chars]
    return excerpt


def _search_sector_news(query: str) -> list[dict]:
    """
    `query` をYahoo! JAPAN検索（失敗時はDuckDuckGo）で実行し、結果一覧を返す。
    各結果ページの本文先頭を抜粋してスニペットとして添付する（検索結果ページの
    スニペットだけでは不足するため）。全滅した場合は空リストを返す。
    """
    items = _search_sector_news_yahoo(query)
    if not items:
        items = _search_sector_news_ddg(query)
    if not items:
        print(f"⚠️ セクター検索の結果が取得できませんでした: {query}")
        return []
    for item in items:
        item["snippet"] = _fetch_page_excerpt(item["url"])
        time.sleep(1)  # 検索結果ページへの負荷軽減のための待機
    return items


def fetch_sector_trend_search(sector: str) -> list[dict]:
    """
    対象セクターの直近動向を kabutan.jp / finance.yahoo.co.jp に site: 絞り込みした
    Web検索で取得し、タイトル・スニペット・URLのリストを返す。
    検索に失敗した場合は警告を出して空リストを返す（分析自体は継続する）。

    取得結果は日付付きでキャッシュし、同日中の再実行ではWeb取得せずキャッシュを使い回す。
    結果が空だった場合はキャッシュせず、別銘柄での再実行時に再取得を試みる。
    """
    if not sector or sector == "不明":
        return []
    query_term = _SECTOR_JA_QUERY_TERMS.get(sector, sector)
    today = date.today().isoformat()
    cache = _load_sector_search_cache()
    if cache is not None and cache.get("fetched_date") == today:
        cached = cache.get("sectors", {}).get(sector)
        if cached:
            print(f"💾 セクター検索結果はキャッシュから読み込みました（{today} 取得分・{sector}）")
            return cached

    results = []
    for query_tpl in SECTOR_SEARCH_QUERIES:
        results.extend(_search_sector_news(query_tpl.format(sector=query_term)))
        time.sleep(1)  # 検索エンジンへの負荷軽減のための待機

    if results:
        # 同日中に別セクターの結果がキャッシュ済みなら、それを保持したまま追記する
        sectors = cache.get("sectors", {}) if cache is not None and cache.get("fetched_date") == today else {}
        sectors[sector] = results
        _save_sector_search_cache(today, sectors)
    return results


# --- Step 1 補助: JPX 投資部門別週次売買動向（海外投資家・個人）取得 ---
JPX_INVESTOR_TYPE_URL = "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"


def fetch_jpx_investor_weekly() -> dict | None:
    """
    JPX「投資部門別売買状況（週間）」の最新週から、東証プライムにおける
    海外投資家・個人の差引き売買額（億円）を取得する。
    取得に失敗した場合は警告を出して None を返す（分析自体は継続する）。
    """
    try:
        resp = cffi_requests.get(JPX_INVESTOR_TYPE_URL, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ JPX 投資部門別ページの取得に失敗しました（スキップします）: {e}")
        return None

    xls_url = None
    week_label = ""
    for tr in soup.select("table tr"):
        txt = tr.get_text(strip=True)
        if not re.search(r"\d{4}年.+第\d週", txt):
            continue
        a = tr.find("a", href=re.compile(r"stock_val_\d+_\d+\.xls"))
        if a:
            xls_url = urljoin(JPX_INVESTOR_TYPE_URL, a["href"])
            week_label = txt
            break
    if not xls_url:
        print("⚠️ JPXの週次XLSリンクが見つからなかったため、投資部門別データをスキップします。")
        return None

    try:
        import io
        import pandas as pd
        resp = cffi_requests.get(xls_url, impersonate="chrome", timeout=60)
        resp.raise_for_status()
        xl = pd.ExcelFile(io.BytesIO(resp.content))
        sheet = "TSE Prime" if "TSE Prime" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet, header=None)
    except Exception as e:
        print(f"⚠️ JPX週次XLSの取得・解析に失敗しました（スキップします）: {e}")
        return None

    def _balance(section: str) -> float | None:
        for i in range(len(df) - 1):
            label = str(df.iat[i, 0]).replace("\u3000", "").strip()
            if label != section:
                continue
            # 差引き列（売り or 買い行のどちらか一方に記載）を優先し、なければ買い-売りで算出する。
            # 列配置: 今週ブロックの金額=col.8, 比率=col.9, 差引き=col.10
            for r in (i, i + 1):
                balance = _parse_num(str(df.iat[r, 10]))
                if balance is not None:
                    return balance / 100_000  # 千円 -> 億円
            sell = _parse_num(str(df.iat[i, 8]))
            buy = _parse_num(str(df.iat[i + 1, 8]))
            if sell is not None and buy is not None:
                return (buy - sell) / 100_000  # 千円 -> 億円
        return None

    prime = {
        "海外投資家": _balance("海外投資家"),
        "個人": _balance("個人"),
    }
    if prime["海外投資家"] is None and prime["個人"] is None:
        print("⚠️ JPX週次XLSから海外投資家・個人のデータを抽出できませんでした（スキップします）。")
        return None
    return {"week": week_label, "prime": prime, "url": xls_url}


# --- Step 2: 地合い・セクターを含む定量スコア計算 (50点満点) ---
def calculate_quantitative_score(data: dict, market_data: dict):
    """
    ファンダメンタル・地合い・流動性の定量評価 (50点満点)
    """
    score = 0
    details = []

    # A. 市場全体の地合い評価 (10点)[cite: 1]
    if market_data.get("nikkei_above_sma25"):
        score += 10
        details.append("地合い良好(日経平均>25日線) +10pt")
    else:
        details.append("地合い低調(日経平均<25日線・逆風注意) +0pt")

    # B. PER・バリュエーション (10点)
    per = data.get("forward_per") or data.get("per")
    if per:
        if 0 < per <= 15:
            score += 10
            details.append(f"PER割安({per:.1f}倍) +10pt")
        elif 15 < per <= 25:
            score += 5
            details.append(f"PER適正({per:.1f}倍) +5pt")
        else:
            details.append(f"PER割高({per:.1f}倍) +0pt")

    # C. EPS成長率 (10点)[cite: 2, 3]
    eps_growth = data.get("eps_growth")
    if eps_growth is not None:
        growth_pct = eps_growth * 100
        if growth_pct >= 15:
            score += 10
            details.append(f"EPS高成長({growth_pct:.1f}%) +10pt")
        elif growth_pct > 0:
            score += 5
            details.append(f"EPS増加傾向({growth_pct:.1f}%) +5pt")
        else:
            details.append(f"EPS減益傾向({growth_pct:.1f}%) +0pt")

    # D. 出来高・流動性 (10点)[cite: 1, 2]
    volume = data.get("volume", 0)
    avg_vol = data.get("avg_volume", 1)
    if volume > avg_vol * 1.3:
        score += 10
        details.append("出来高急増(買い意欲旺盛) +10pt")
    elif volume >= avg_vol:
        score += 5
        details.append("出来高標準 +5pt")

    # E. 時価総額・流動性リスク (10点)[cite: 2, 3]
    market_cap = data.get("market_cap", 0)
    if market_cap >= 100_000_000_000:
        score += 10
        details.append("時価総額1000億超(流動性高) +10pt")
    elif market_cap >= 20_000_000_000:
        score += 5
        details.append("時価総額200億超 +5pt")

    return score, details


# --- Step 3: DeepSeek による地合い・セクター・決算リスク定性分析 ---
def _parse_json_response(raw_text: str) -> dict:
    raw_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # 前後に説明文が付いた場合のフォールバック: 最初の '{' から最後の '}' までを抽出して解析
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw_text[start:end + 1])
        raise


def _build_analysis_prompt(data: dict, market_data: dict, quant_score: int, quant_details: list, context: dict | None = None) -> str:
    """
    DeepSeek API 用の分析プロンプトを組み立てる（Web検索はPython側で事前実行し、結果を埋め込む前提）。
    株探・JPX・yfinance から取得済みのデータがあれば、根拠データとしてプロンプト内に埋め込む。
    """
    context = context or {}
    kessan_news = context.get("kessan_news") or []
    industry_ranking = context.get("industry_ranking") or []
    sector_news = context.get("sector_news") or []
    technicals = context.get("technicals") or {}
    next_earnings = context.get("next_earnings")
    investor_weekly = context.get("investor_weekly")

    def _signed(v, digits=1):
        return "--" if v is None else f"{v:+.{digits}f}%"

    def _oku(v):
        return "--" if v is None else f"{v:+,.0f}億円"

    kessan_section = ""
    if kessan_news:
        articles = "\n\n".join(
            f"{i}. [{n['datetime'][:10]}] {n['title']}\n{n['url']}\n{n['body']}"
            for i, n in enumerate(kessan_news, 1)
        )
        kessan_section = (
            "\n【株探 決算速報（取得済みデータ・最新順）】\n"
            "株探(kabutan.jp)の「決算速報」から取得した直近記事です。"
            "決算イベントリスクや業績トレンドの評価には必ず以下を根拠として使用してください。\n\n"
            + articles
        )

    industry_section = ""
    if industry_ranking:
        lines = []
        for r in industry_ranking:
            pct = f"{r['change_pct']:+.2f}%" if r["change_pct"] is not None else "--"
            per = f"{r['per']}" if r["per"] is not None else "--"
            lines.append(f"- {r['name']}: 前日比 {pct} / PER {per}")
        up = sum(1 for r in industry_ranking if (r["change_pct"] or 0) > 0)
        down = sum(1 for r in industry_ranking if (r["change_pct"] or 0) < 0)
        industry_section = (
            f"\n【株探 業種別ランキング（取得済みデータ・全{len(industry_ranking)}業種・値上がり{up}/値下がり{down}）】\n"
            "株探(kabutan.jp)の業種別ランキングから取得した全業種の平均株価の増減（前日比%）とPERです。"
            "セクター資金流動の評価には必ず以下を根拠として使用してください。\n\n"
            + "\n".join(lines)
        )

    sector_news_section = ""
    if sector_news:
        entries = "\n\n".join(
            f"{i}. [{r['source']}] {r['title']}\n{r['url']}\n{r['snippet']}"
            for i, r in enumerate(sector_news, 1)
        )
        sector_news_section = (
            f"\n【セクター関連Web検索結果（{data['sector']}セクター・取得済みデータ）】\n"
            "kabutan.jp / finance.yahoo.co.jp を対象に事前検索した直近記事のタイトルと本文抜粋です。"
            "セクターの直近1〜2週間トレンドの評価には必ず以下を根拠として使用してください。\n\n"
            + entries
        )

    technical_section = ""
    if technicals:
        technical_section = (
            "\n【対象銘柄テクニカル指標（yfinance・取得済みデータ）】\n"
            "押し目か下落トレンド転換かの判定には必ず以下を根拠として使用してください。\n\n"
            f"- SMA25乖離: {_signed(technicals.get('sma25_dev'))} / SMA75乖離: {_signed(technicals.get('sma75_dev'))}\n"
            f"- RSI14: {technicals.get('rsi14'):.1f}\n"
            f"- 52週高値 {technicals.get('high_52w'):,.0f}円 からの下落率: {_signed(technicals.get('drawdown_from_high'))}"
            f"（52週安値: {technicals.get('low_52w'):,.0f}円）\n"
            f"- 直近1ヶ月騰落率: {_signed(technicals.get('change_1mo'))} / 直近3ヶ月騰落率: {_signed(technicals.get('change_3mo'))}"
        )

    earnings_section = ""
    if next_earnings:
        earnings_section = (
            "\n【次回決算発表予定日（株探の過去発表実績からの推定）】\n"
            f"推定: {next_earnings['estimated_date']}（約{next_earnings['days_until']}日後）。"
            f"根拠: 例年の発表日 {', '.join(next_earnings['history'])}。\n"
            "※推定値のため実際は前後する可能性があります。14日以内と判定した場合は推定誤差を考慮し、厳めに評価してください。"
        )

    investor_section = ""
    if investor_weekly:
        prime = investor_weekly.get("prime") or {}
        investor_section = (
            "\n【投資部門別週次売買動向（JPX・東証プライム・取得済みデータ）】\n"
            f"{investor_weekly.get('week', '')}: 海外投資家 {_oku(prime.get('海外投資家'))} / 個人 {_oku(prime.get('個人'))}"
            "（差引き。プラス=買い越し、マイナス=売り越し）\n"
            "全体地合いの判断には必ず以下を根拠として使用してください。"
        )

    tone_parts = [f"日経平均25日線クリア = {market_data.get('nikkei_above_sma25')}"]
    if market_data.get("nikkei_rsi14") is not None:
        tone_parts.append(f"日経RSI14 = {market_data['nikkei_rsi14']:.1f}")
    if market_data.get("nikkei_change_1mo") is not None:
        tone_parts.append(f"日経1ヶ月騰落率 = {market_data['nikkei_change_1mo']:+.1f}%")
    nikkei_tone = " / ".join(tone_parts)

    return f"""
あなたは勝率重視の日本株ファンダメンタル・アナリストです。
押し目買い(Dip Buying)戦略を実行するにあたり、対象銘柄を取り巻く「全体地合い」および「セクターへの資金流動（追い風・逆風）」を調査し、「ダマシ」や「地合い・セクター逆風による下落連動」を回避するための定性調査を実施して総合判断を下してください。

【評価対象】
- 銘柄: {data['ticker']} ({data['name']}) / セクター: {data['sector']}
- 株価: {data['price']}円 / PER: {data['per']}倍
- 事前定量スコア: {quant_score}/50点 ({', '.join(quant_details)})
- 市場全体地合い状況: {nikkei_tone}
{kessan_section}
{industry_section}
{sector_news_section}
{technical_section}
{earnings_section}
{investor_section}

【定性分析指示】
※主要な一次データはすべて上記【取得済みデータ】として埋め込み済みです。セクター動向のWeb検索も事前に実行済みで、結果を埋め込んでいます。外部ツールは利用できないため、判断はすべて埋め込み済みデータのみを根拠として行ってください。
1. **全体地合いの確認**:
   - 取得済みデータ（日経平均の25日線・RSI14・1ヶ月騰落率、業種別前日比の分布、海外投資家・個人の週次売買動向）のみで、市場全体が「買われすぎ」「中立」「冷え込み（売り優勢）」のどちらかを判断してください。
2. **セクター資金流動チェック**:
   - 【株探 業種別ランキング】の全業種前日比を最重視し、対象セクターが当日「資金流入（値上がり上位）」か「資金流出（逆風）」かを確認してください。
   - 【{data['sector']}】セクターの直近1〜2週間のトレンドは、【セクター関連Web検索結果】の記事タイトル・本文抜粋を根拠に判断してください（当日データだけでは週間の資金流動を判別できないため）。検索結果が取得できていない場合は、業種別ランキングの前日比と対象銘柄の直近1ヶ月・3ヶ月騰落率、日経平均の1ヶ月騰落率から推測してください。
   - セクター全体が下降トレンドまたは資金流出局面にある場合、「安値追い（さらなる値下がり）」のリスクが高いか？
3. **該当銘柄の押し目判定**:
   - 【対象銘柄テクニカル指標】のSMA25/75乖離・RSI14・52週高値からの下落率・直近1/3ヶ月騰落率を根拠に、対象銘柄の下落が「一時的な健全な押し目（全体の地合い悪化に連動した一時的調整）」か「個別材料悪化による本格的な下落トレンド転換」かを判定してください。
   - 個別材料の有無は【株探 決算速報】の記事と取得済みデータのみで判断してください。
4. **決算イベントリスクの調査**:
   - 上記【株探 決算速報】の取得済み記事があれば、直近業績・会社計画の一次情報として最重視してください。
   - 【次回決算発表予定日（推定）】が14日以内の場合は、推定誤差を考慮した上で減点対象（または「WAIT」判定）としてください。
5. **総合判断（100点満点）の算出**:
   - 事前定量スコア({quant_score}点)に、定性評価(50点満点)を加算・調整してください。
   - ※セクターから激しい資金流出がある場合や、決算発表直前(14日以内)の場合は、総合スコアを大きく減点（または「WAIT」判定）にしてください。

【出力形式】
必ず以下のJSONフォーマットのみを出力してください（Markdown装飾コードブロックを含めないでください）。

{{
  "final_score": 82,
  "judgement": "GO" または "WAIT" または "NO_GO",
  "reason": "150字程度で地合い・セクター風向き・ファンダメンタルズを含めた評価理由",
  "sector_trend": "セクター資金流動の状況（例: 化学セクターへ資金流入傾向あり）",
  "earnings_risk": "決算リスクの有無（例: 直近10日後に発表予定のため減点）"
}}
"""


def evaluate_with_deepseek(data: dict, market_data: dict, quant_score: int, quant_details: list, context: dict | None = None):
    """
    DeepSeek API（OpenAI互換のchat/completions）を直接呼び出して定性評価を行う。
    JSON出力を強制する response_format を指定し、それでもJSONとして解析できない場合は
    1回だけリトライする（2回目のプロンプトにはJSONのみの出力を強く指示する）。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY が設定されていません。.env に DeepSeek のAPIキーを設定してください。"
        )
    prompt = _build_analysis_prompt(data, market_data, quant_score, quant_details, context)

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "あなたは勝率重視の日本株ファンダメンタル・アナリストです。"
                    "ユーザーのプロンプトに埋め込まれた【取得済みデータ】のみを根拠に定性評価を行い、"
                    "プロンプトに指定されたJSONのみを出力してください。"
                    "前後の説明文やマークダウンのコードブロック装飾は付けないこと。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},  # JSON出力を強制する
        "temperature": 0.3,
        "max_tokens": 2000,
    }

    for attempt in (1, 2):
        try:
            resp = cffi_requests.post(
                url, headers=headers, json=payload, impersonate="chrome", timeout=DEEPSEEK_TIMEOUT
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"DeepSeek API の呼び出しに失敗しました: {e}")

        obj = resp.json()
        content = (obj.get("choices") or [{}])[0].get("message", {}).get("content", "")
        try:
            return _parse_json_response(content)
        except json.JSONDecodeError:
            if attempt == 2:
                raise RuntimeError(
                    f"DeepSeek の応答からJSONを抽出できませんでした。\n応答: {content[:500]}"
                )
            # 2回目はJSONのみを強制するリトライ
            payload["messages"].append(
                {"role": "assistant", "content": content},
            )
            payload["messages"].append(
                {"role": "user", "content": "前回の応答はJSONとして解析できませんでした。指示されたJSONオブジェクトのみを出力してください。"}
            )


# --- Step 4 準備: 日足チャート生成 & Notion への画像アップロード ---
def generate_daily_chart(ticker_code: str, out_path: str) -> bool:
    """
    対象銘柄の日足ローソク足チャート（25日移動平均・出来高付き）をPNGで保存する。
    データ取得に失敗した場合は False を返し、呼び出し側で画像添付をスキップする。
    """
    import matplotlib
    matplotlib.use("Agg")  # ディスプレイ非依存のバックエンドで描画する
    import mplfinance as mpf

    try:
        hist = yf.Ticker(f"{ticker_code}.T", session=_get_yf_session()).history(period="6mo")
    except Exception as e:
        print(f"⚠️ 日足データの取得に失敗しました（チャートをスキップします）: {e}")
        return False
    if hist.empty:
        print("⚠️ 日足データが空のためチャートをスキップします。")
        return False

    mpf.plot(
        hist,
        type="candle",
        style="yahoo",
        title=f"{ticker_code} Daily Chart (6mo)",
        ylabel="Price (JPY)",
        volume=True,
        mav=(25,),
        savefig=dict(fname=out_path, dpi=110, bbox_inches="tight"),
    )
    return True


def upload_chart_to_notion(notion: Client, png_path: str, filename: str) -> str:
    """
    PNGをNotionのFile Upload APIでアップロードし、file_upload のIDを返す。
    single_part アップロードは send 完了で即利用可能になるため complete は不要。
    ページへの紐付けは image ブロックの {"type": "file_upload", "file_upload": {"id": ...}} で行う。
    """
    upload = notion.file_uploads.create(filename=filename, content_type="image/png")
    with open(png_path, "rb") as f:
        notion.file_uploads.send(upload["id"], file=f)
    return upload["id"]


# --- Step 4: Notion データベースへの記録（地合い・セクター列を追加） ---
def record_to_notion(data: dict, quant_score: int, ai_result: dict, chart_path: str | None = None):
    """
    Notionへ評価結果を保存。chart_path を指定すると日足チャート画像をページ内に添付する。
    """
    notion = Client(auth=NOTION_API_KEY)

    properties = {
        "ページタイトル": {"title": [{"text": {"content": f"{data['ticker']}: {data['name']}({data['date']})"}}]},
        "日付": {"date": {"start": data.get("date", None)}},
        "銘柄コード": {"rich_text": [{"text": {"content": data["ticker"]}}]},
        "銘柄名": {"rich_text": [{"text": {"content": data["name"]}}]},
        "セクター": {"rich_text": [{"text": {"content": str(data["sector"])}}]},
        "株価": {"number": float(data["price"]) if data["price"] else 0},
        "判定": {"select": {"name": ai_result.get("judgement", "WAIT")}},
        "最終スコア": {"number": int(ai_result.get("final_score", 0))},
        "定量スコア": {"number": quant_score},
        "セクター風向き": {"rich_text": [{"text": {"content": ai_result.get("sector_trend", "")}}]},
        "理由・リスク概要": {"rich_text": [{"text": {"content": ai_result.get("reason", "")}}]},
        "決算リスク": {"rich_text": [{"text": {"content": ai_result.get("earnings_risk", "")}}]},
    }

    children = []
    if chart_path:
        try:
            file_upload_id = upload_chart_to_notion(
                notion, chart_path, f"{data['ticker']}_daily_{data['date']}.png"
            )
            children.append({
                "type": "image",
                "image": {"type": "file_upload", "file_upload": {"id": file_upload_id}},
            })
        except Exception as e:
            print(f"⚠️ チャート画像のアップロードに失敗しました（画像なしでページを作成します）: {e}")

    notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties=properties,
        children=children,
    )
    print("✅ Notionへの保存が完了しました。")


# --- メイン実行処理 ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <銘柄コード>")
        sys.exit(1)

    ticker = sys.argv[1]
    print(f"🔍 銘柄コード: {ticker} の分析を開始します...")

    # 1. データ取得（個別 ＋ 地合い）
    data, market_data = fetch_stock_and_market_data(ticker)
    print(f"📊 データ取得完了: {data['name']} / セクター: {data['sector']}")

    # 1-2. 株探 決算速報ニュース取得（失敗しても分析自体は継続する）
    print("📰 株探(kabutan.jp)から決算速報を取得しています...")
    kessan_news = fetch_kabutan_kessan_news(ticker)
    if kessan_news:
        for n in kessan_news:
            print(f"   - [{n['datetime'][:10]}] {n['title']}")
    else:
        print("   決算速報は見つかりませんでした。")

    # 1-3. 株探 業種別ランキング取得（失敗しても分析自体は継続する）
    print("🏭 株探(kabutan.jp)から業種別ランキングを取得しています...")
    industry_ranking = fetch_kabutan_industry_ranking()
    if industry_ranking:
        best = max(industry_ranking, key=lambda r: r["change_pct"] if r["change_pct"] is not None else float("-inf"))
        worst = min(industry_ranking, key=lambda r: r["change_pct"] if r["change_pct"] is not None else float("inf"))
        print(f"   全{len(industry_ranking)}業種の取得完了: 値上がり首位 {best['name']}({best['change_pct']:+.2f}%) / 値下がり首位 {worst['name']}({worst['change_pct']:+.2f}%)")
    else:
        print("   業種別ランキングは取得できませんでした。")

    # 1-3b. 対象セクターの関連ニュース検索（kabutan / Yahoo!ファイナンス。失敗しても分析自体は継続する）
    print("🔎 対象セクターの直近動向をWeb検索しています（kabutan / Yahoo!ファイナンス）...")
    sector_news = fetch_sector_trend_search(data["sector"])
    if sector_news:
        for r in sector_news:
            print(f"   - [{r['source']}] {r['title']}")
    else:
        print("   セクター関連の検索結果は取得できませんでした。")

    # 1-4. テクニカル指標・次回決算日（推定）・投資部門別週次データの取得（失敗しても分析自体は継続する）
    print("📉 yfinanceからテクニカル指標を算出しています...")
    technicals = fetch_stock_technicals(ticker)
    if technicals:
        print(f"   SMA25乖離 {technicals['sma25_dev']:+.1f}% / RSI14 {technicals['rsi14']:.1f} / 52週高値から{technicals['drawdown_from_high']:+.1f}%")
    else:
        print("   テクニカル指標は取得できませんでした。")

    print("🗓️ 株探の過去発表実績から次回決算発表予定日を推定しています...")
    next_earnings = fetch_kabutan_next_earnings_date(ticker)
    if next_earnings:
        print(f"   推定: {next_earnings['estimated_date']}（あと約{next_earnings['days_until']}日）")
    else:
        print("   次回決算日は推定できませんでした。")

    print("🌏 JPXから投資部門別週次売買動向を取得しています...")
    investor_weekly = fetch_jpx_investor_weekly()
    if investor_weekly:
        prime = investor_weekly["prime"]
        foreign_txt = f"{prime['海外投資家']:+,.0f}億円" if prime["海外投資家"] is not None else "--"
        individual_txt = f"{prime['個人']:+,.0f}億円" if prime["個人"] is not None else "--"
        print(f"   {investor_weekly['week']}: 海外投資家 {foreign_txt} / 個人 {individual_txt}")
    else:
        print("   投資部門別データは取得できませんでした。")

    # 2. 定量スコア計算
    quant_score, quant_details = calculate_quantitative_score(data, market_data)
    print(f"📈 定量スコア: {quant_score}/50点\n   内訳: {', '.join(quant_details)}")

    # 3. AI による定性・セクター・決算リスク分析
    #    定量スコアが閾値以下の銘柄は定量面で基準未満のため、AI評価をスキップして NO_GO とする
    context = {
        "kessan_news": kessan_news,
        "industry_ranking": industry_ranking,
        "sector_news": sector_news,
        "technicals": technicals,
        "next_earnings": next_earnings,
        "investor_weekly": investor_weekly,
    }
    if quant_score <= AI_EVAL_SKIP_THRESHOLD:
        print(f"⏭️ 定量スコアが{quant_score}点（{AI_EVAL_SKIP_THRESHOLD}点以下）のため、AIによる定性評価をスキップします。")
        ai_result = {
            "final_score": quant_score,
            "judgement": "NO_GO",
            "reason": f"定量スコア{quant_score}点（{AI_EVAL_SKIP_THRESHOLD}点以下）のためAI定性評価をスキップ。定量面で基準未満。",
            "sector_trend": "",
            "earnings_risk": "",
        }
    else:
        print("🤖 DeepSeek APIで分析中（取得済みデータを優先し、数分かかる場合があります）...")
        ai_result = evaluate_with_deepseek(data, market_data, quant_score, quant_details, context)

    print("\n--- 最終分析結果 ---")
    print(f"【判定】: {ai_result.get('judgement')}")
    print(f"【総合スコア】: {ai_result.get('final_score')} / 100点")
    print(f"【セクター風向き】: {ai_result.get('sector_trend')}")
    print(f"【理由】: {ai_result.get('reason')}")
    print(f"【決算リスク】: {ai_result.get('earnings_risk')}")

    if kessan_news:
        print("\n--- 株探 決算速報（直近3件） ---")
        for n in kessan_news:
            print(f"\n■ [{n['datetime'][:10]}] {n['title']}")
            print(n['body'] or "（本文の取得に失敗しました）")
        print()

    # 4. Notionへ記録（NO_GOの場合は書き込まない）
    if ai_result.get("judgement") == "NO_GO":
        print("⏭️ 判定がNO_GOのため、Notionへの書き込みをスキップします。")
    elif NOTION_API_KEY and NOTION_DATABASE_ID:
        # 4-1. 日足チャート生成（失敗しても記録自体は継続する）
        chart_path = None
        tmp_chart = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_chart.close()
        print("📈 日足チャートを生成しています...")
        if generate_daily_chart(ticker, tmp_chart.name):
            chart_path = tmp_chart.name

        # 4-2. Notionへ書き込み
        print("📝 Notionデータベースへ書き込んでいます...")
        try:
            record_to_notion(data, quant_score, ai_result, chart_path)
        finally:
            if os.path.exists(tmp_chart.name):
                os.unlink(tmp_chart.name)

if __name__ == "__main__":
    main()