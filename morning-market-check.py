"""
朝の市場地合い判定スクリプト（新規エントリーのサーキットブレーカー判定）。

前日大引け（15:00）後〜当日朝にかけての海外市場・為替・日経平均先物・ニュースを
スクリプト内でWebから収集し、そのテキストをプロンプトに埋め込んで DeepSeek API に投げ、
当日の新規買いエントリーの可否を【GO / CAUTION / NO GO】で判定する。

処理内容:
1. 市場データ収集（yfinance）
   - 米国主要指数（NYダウ / S&P500 / NASDAQ / SOX半導体）の騰落率
   - 日経平均先物（CME NKD=F）の前日東証終値比
   - ドル円（JPY=X）の前日15:00時点比 / VIX（^VIX）/ 米10年債利回り（^TNX）
2. 機械的NO GO判定（客観的な足切り基準）
   - 日経平均先物 前日東証終値比 -1.5%以下
   - SOX -3.0%以上 / NASDAQ -2.0%以上の急落
   - VIX急騰（20超え または 前日比 +10%以上）
   - ドル円が前日15:00時点比 -1.5円以上の急激な円高
   → いずれかに該当した場合はAIの判定にかかわらず強制NO GO
3. ニュース収集（すべてベストエフォート。1ソース失敗しても判定は継続）
   - 株探 市場ニュース（記事本文まで取得）
   - Yahoo!ファイナンス トップのニュース見出し
   - Google News RSS / DuckDuckGo検索（夜間〜朝の市況・地政学ニュース補完）
4. DeepSeek API に判定プロンプトを投げて最終判定
5. 判定結果を標準出力し、DISCORD_WEBHOOK_URL が設定されていればDiscordへ通知

Usage:
  python morning-market-check.py          # 朝8:00〜8:55頃の実行を想定
  python morning-market-check.py --no-ai  # AI判定をスキップし収集データと機械判定のみ表示（動作確認用）
"""

import json
import math
import os
import re
import sys
import time
from datetime import datetime, time as dtime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

# .envの読み込み
load_dotenv()

# DeepSeek API の設定（stock-analyze.py と共通）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # deepseek-chat / deepseek-reasoner
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_TIMEOUT = 300  # 推論待ちのタイムアウト（秒）

# Discord通知（任意）。.env に DISCORD_WEBHOOK_URL があれば判定結果を投稿する
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

JST = ZoneInfo("Asia/Tokyo")

KABUTAN_BASE_URL = "https://kabutan.jp"
KABUTAN_MARKET_NEWS_URL = "https://kabutan.jp/news/marketnews/"
YAHOO_FINANCE_TOP_URL = "https://finance.yahoo.co.jp/"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# 夜間〜朝の動向を拾う検索クエリ（Google News RSS / DuckDuckGo 共通）
OVERNIGHT_QUERIES = [
    "日経平均先物 円相場",
    "米国株 ダウ ナスダック 半導体",
    "ドル円 為替",
    "地政学リスク 原油 米国債",
]

# 各ソースの取得件数上限（プロンプト肥大化を防ぐための目安）
KABUTAN_NEWS_LIMIT = 6
KABUTAN_BODY_MAX_CHARS = 500
YAHOO_TOPICS_LIMIT = 10
GOOGLE_NEWS_PER_QUERY = 4
DDG_PER_QUERY = 3
DDG_SNIPPET_MAX_CHARS = 250

# 判定結果の表示ラベル
VERDICT_LABELS = {
    "GO": "GO（エントリー可能）",
    "CAUTION": "CAUTION（ロット半減・指値限定）",
    "NO GO": "NO GO（エントリー停止・見送り）",
}


# --- Step 1: 市場データ収集（yfinance） ---
# (yfinanceシンボル, 表示名, 変動の単位: "pct"=%, "yen"=円)
MARKET_SYMBOLS = [
    ("^DJI", "NYダウ", "pct"),
    ("^GSPC", "S&P500", "pct"),
    ("^IXIC", "NASDAQ総合", "pct"),
    ("^SOX", "SOX半導体", "pct"),
    ("^VIX", "VIX恐怖指数", "pct"),
    ("^TNX", "米10年債利回り", "pct"),
    ("JPY=X", "ドル円", "yen"),
    ("NKD=F", "日経225先物(CME)", "pct"),
    ("^N225", "日経平均(東証)", "pct"),
]

_YF_SESSION = None


def _get_yf_session():
    """
    Yahoo Finance 用のブラウザ偽装セッションを返す（全シンボルで使い回す）。
    stock-analyze.py と同様のBot対策。
    """
    global _YF_SESSION
    if _YF_SESSION is None:
        _YF_SESSION = cffi_requests.Session(impersonate="chrome")
    return _YF_SESSION


def _fetch_live_price(ticker, fallback: float) -> float:
    """fast_info からリアルタイムの直近値を取得する。失敗時は直近終値を返す。"""
    try:
        price = float(ticker.fast_info["last_price"])
        if price and not math.isnan(price):
            return price
    except Exception:
        pass
    return fallback


def _fetch_usdjpy_prev_1500(ticker) -> float | None:
    """
    ドル円の「前日15:00時点」の値を5分足から取得する（急激な円高判定の基準値）。
    前日が土日祝などで該当バーが無い場合は None を返す（呼び出し側で前日終値にフォールバック）。
    """
    target = datetime.now(JST) - timedelta(days=1)
    if target.weekday() >= 5:  # 土日は為替市場がほぼ動かないためスキップ
        return None
    try:
        hist = ticker.history(period="2d", interval="5m")
        if hist.empty:
            return None
        idx = hist.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx = idx.tz_convert(JST)
        day_mask = idx.date == target.date()  # numpy bool配列
        time_mask = np.array([t.time() <= dtime(15, 0) for t in idx])
        candidates = np.flatnonzero(day_mask & time_mask)
        if len(candidates) == 0:
            return None
        return float(hist["Close"].iloc[candidates[-1]])
    except Exception as e:
        print(f"⚠️ ドル円の前日15:00時点の値の取得に失敗しました（前日終値で代用します）: {e}")
        return None


def fetch_market_data() -> dict:
    """
    夜間〜朝の市場データを yfinance で取得し、シンボルをキーにしたdictで返す。
    各エントリ: {"label", "unit", "prev_close", "last_close", "last_close_date",
                "latest", "change", "change_pct"}（失敗したシンボルは値がNoneのまま）。
    加えて "nkd_vs_tse_pct"（CME先物の前日東証終値比）をトップレベルに持つ。
    """
    data: dict = {}
    for symbol, label, unit in MARKET_SYMBOLS:
        entry = {
            "label": label, "unit": unit,
            "prev_close": None, "last_close": None, "last_close_date": None,
            "latest": None, "change": None, "change_pct": None,
        }
        try:
            ticker = yf.Ticker(symbol, session=_get_yf_session())
            hist = ticker.history(period="5d")
            if hist.empty or len(hist) < 2:
                print(f"⚠️ {label}({symbol}): 日足データが不足しているためスキップします。")
                data[symbol] = entry
                continue
            prev_close = float(hist["Close"].iloc[-2])
            last_close = float(hist["Close"].iloc[-1])
            latest = _fetch_live_price(ticker, last_close)
            entry.update({
                "prev_close": prev_close,
                "last_close": last_close,
                "last_close_date": hist.index[-1].date().isoformat(),
                "latest": latest,
                "change": latest - prev_close,
                "change_pct": (latest / prev_close - 1) * 100 if prev_close else None,
            })
        except Exception as e:
            print(f"⚠️ {label}({symbol}): データ取得に失敗しました（スキップします）: {e}")
        data[symbol] = entry

        # ドル円は機械判定用に「前日15:00時点」の値も取得する（急激な円高判定の基準値）
        if symbol == "JPY=X" and entry["latest"] is not None:
            entry["prev_1500"] = _fetch_usdjpy_prev_1500(ticker)

    # 日経平均先物(CME)の対東証終値比（前日東証終値は ^N225 の前日終値で近似する）
    nkd = data.get("NKD=F", {})
    n225 = data.get("^N225", {})
    if nkd.get("latest") and n225.get("prev_close"):
        data["nkd_vs_tse_pct"] = (nkd["latest"] / n225["prev_close"] - 1) * 100
    else:
        data["nkd_vs_tse_pct"] = None

    return data


# --- Step 2: 機械的NO GO判定（足切り基準） ---
def mechanical_no_go_rules(market: dict) -> list[dict]:
    """
    あらかじめ定めた客観的な足切り基準（前日大引け後〜朝の急変検知）を機械的に判定する。
    該当したルールを [{"name": ルール名, "detail": 数値根拠}, ...] で返す。
    """
    rules = []

    # 1. 先物の急落: 前日東証終値比 -1.5%以下
    nkd = market.get("NKD=F", {})
    nkd_vs_tse = market.get("nkd_vs_tse_pct")
    if nkd.get("latest") is not None and nkd_vs_tse is not None and nkd_vs_tse <= -1.5:
        rules.append({
            "name": "先物急落",
            "detail": f"日経225先物(CME) {nkd['latest']:,.0f}円 は前日東証終値比 {nkd_vs_tse:+.2f}%（基準: -1.5%以下で発動）",
        })
    elif nkd.get("latest") is None:
        # CME先物が取得できない場合は日経平均(^N225)の前日比で代用判定する
        n225 = market.get("^N225", {})
        if n225.get("change_pct") is not None and n225["change_pct"] <= -1.5:
            rules.append({
                "name": "先物急落（日経平均で代用）",
                "detail": f"日経平均(^N225) 前日比 {n225['change_pct']:+.2f}%（CME先物が取得できなかったため代用・基準: -1.5%以下で発動）",
            })

    # 2. 市場急変指標: SOX -3.0%以上 / NASDAQ -2.0%以上の急落
    sox = market.get("^SOX", {})
    if sox.get("change_pct") is not None and sox["change_pct"] <= -3.0:
        rules.append({
            "name": "SOX急落",
            "detail": f"SOX半導体指数 前日比 {sox['change_pct']:+.2f}%（基準: -3.0%以下で発動）",
        })
    ixic = market.get("^IXIC", {})
    if ixic.get("change_pct") is not None and ixic["change_pct"] <= -2.0:
        rules.append({
            "name": "NASDAQ急落",
            "detail": f"NASDAQ総合 前日比 {ixic['change_pct']:+.2f}%（基準: -2.0%以下で発動）",
        })

    # 3. VIXの急騰: 20超え または 前日比 +10%以上
    vix = market.get("^VIX", {})
    if vix.get("latest") is not None and vix.get("change_pct") is not None:
        if vix["latest"] > 20 or vix["change_pct"] >= 10:
            rules.append({
                "name": "VIX急騰",
                "detail": f"VIX {vix['latest']:.1f}（前日比 {vix['change_pct']:+.1f}%）（基準: 20超え または +10%以上で発動）",
            })

    # 4. 為替ショック: 前日15:00時点比 -1.5円以上の急激な円高
    usdjpy = market.get("JPY=X", {})
    base = usdjpy.get("prev_1500") or usdjpy.get("prev_close")
    if usdjpy.get("latest") is not None and base is not None:
        change_vs_1500 = usdjpy["latest"] - base
        usdjpy["change_vs_1500"] = change_vs_1500  # プロンプト表示用に保持
        if change_vs_1500 <= -1.5:
            rules.append({
                "name": "急激な円高",
                "detail": f"ドル円 {usdjpy['latest']:.2f}円 は前日15:00時点比 {change_vs_1500:+.2f}円（基準: -1.5円以上で発動）",
            })

    return rules


# --- Step 3: ニュース収集（ベストエフォート） ---
def overnight_cutoff(now: datetime) -> datetime:
    """ニュース収集の開始時点「前日15:00」を返す。"""
    prev = now - timedelta(days=1)
    return prev.replace(hour=15, minute=0, second=0, microsecond=0)


def _parse_jst_datetime(text: str) -> datetime | None:
    """'2026-10-01T20:11:13+09:00' 形式の日時をJSTのdatetimeに変換する。"""
    try:
        return datetime.fromisoformat(text).astimezone(JST)
    except (ValueError, AttributeError):
        return None


def fetch_kabutan_market_news(cutoff: datetime | None = None, limit: int = KABUTAN_NEWS_LIMIT) -> list[dict]:
    """
    株探の市場ニュース(/news/marketnews/)から、cutoff以降に配信された記事を最大limit件取得し、
    各記事ページの本文（先頭KABUTAN_BODY_MAX_CHARS文字）まで取得する。
    失敗時は警告を出して空リストを返す（判定自体は継続する）。
    """
    try:
        resp = cffi_requests.get(KABUTAN_MARKET_NEWS_URL, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ 株探 市場ニュース一覧の取得に失敗しました（スキップします）: {e}")
        return []

    items = []
    for row in soup.select("table tr"):
        a = row.select_one("a[href*='marketnews']")
        t = row.select_one("time[datetime]")
        if not a or not t:
            continue
        title = a.get_text(strip=True)
        if not title:
            continue
        published = _parse_jst_datetime(t["datetime"])
        if cutoff is not None and published is not None and published < cutoff:
            continue
        items.append({
            "datetime": t["datetime"],
            "title": title,
            "url": urljoin(KABUTAN_BASE_URL, a["href"]),
        })
        if len(items) >= limit:
            break

    news = []
    for item in items:
        item["body"] = ""
        try:
            resp = cffi_requests.get(item["url"], impersonate="chrome", timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            body = soup.select_one("div.body")
            if body is not None:
                lines = [ln.strip() for ln in body.get_text("\n", strip=True).splitlines()]
                item["body"] = "\n".join(ln for ln in lines if ln)[:KABUTAN_BODY_MAX_CHARS]
        except Exception as e:
            print(f"⚠️ 株探 記事本文の取得に失敗しました（タイトルのみ保持します）: {item['title']} ({e})")
        news.append(item)
        time.sleep(1)  # 株探への負荷軽減のための待機

    return news


def fetch_yahoo_finance_topics(cutoff: datetime | None = None, limit: int = YAHOO_TOPICS_LIMIT) -> list[dict]:
    """
    Yahoo!ファイナンス トップのニュース見出し（/news/detail/リンク）を最大limit件取得する。
    リンクテキスト先頭の「HH:MM」を時刻として抽出し、実行時刻より大幅に未来の時刻は
    前日の記事とみなす（ページ側の時刻表示ゆらぎは90分まで許容）。失敗時は警告を出して空リストを返す。
    """
    try:
        resp = cffi_requests.get(YAHOO_FINANCE_TOP_URL, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ Yahoo!ファイナンス トップの取得に失敗しました（スキップします）: {e}")
        return []

    now = datetime.now(JST)
    items = []
    seen: set[str] = set()
    for a in soup.select("a[href*='/news/detail/']"):
        text = a.get_text(" ", strip=True)
        href = a["href"]
        if not text or href in seen:
            continue
        seen.add(href)
        published = None
        title = text
        # リンクテキストは「タイトル … HH:MM 配信元 …」の形式のため、末尾側の時刻を配信時刻とみなす
        matches = list(re.finditer(r"\b(\d{1,2}):(\d{2})\b", text))
        if matches:
            m = matches[-1]
            hour, minute = int(m.group(1)), int(m.group(2))
            title = text[: m.start()].strip()
            candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate > now + timedelta(minutes=90):
                candidate -= timedelta(days=1)
            published = candidate
        if cutoff is not None and published is not None and published < cutoff:
            continue
        items.append({
            "datetime": published,
            "title": title,
            "url": urljoin(YAHOO_FINANCE_TOP_URL, href),
        })
        if len(items) >= limit:
            break
    return items


def fetch_google_news_rss(query: str, cutoff: datetime | None = None, limit: int = GOOGLE_NEWS_PER_QUERY) -> list[dict]:
    """
    Google News RSSで `query` を検索し、cutoff以降に配信された記事を最大limit件返す。
    pubDateはGMTのためJSTに変換してからフィルタする。失敗時は警告を出して空リストを返す。
    """
    try:
        resp = cffi_requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={"q": query, "hl": "ja", "gl": "JP", "ceid": "JP:ja"},
            impersonate="chrome",
            timeout=30,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
    except Exception as e:
        print(f"⚠️ Google News RSSの取得に失敗しました（スキップします）: {query} ({e})")
        return []

    items = []
    for it in soup.find_all("item"):
        title = it.title.get_text(strip=True) if it.title else ""
        link = it.link.get_text(strip=True) if it.link else ""
        if not title or not link:
            continue
        published = None
        pubdate = it.pubDate.get_text(strip=True) if it.pubDate else ""
        try:
            published = parsedate_to_datetime(pubdate).astimezone(JST)
        except (ValueError, TypeError):
            pass
        if cutoff is not None and published is not None and published < cutoff:
            continue
        items.append({
            "datetime": published.isoformat() if published else "",
            "title": title,
            "url": link,
            "query": query,
        })
        if len(items) >= limit:
            break
    return items


def _decode_ddg_href(href: str) -> str:
    """DuckDuckGoのリダイレクトURL(//duckduckgo.com/l/?uddg=...)から実URLを取り出す。"""
    parsed = urlparse(href)
    if "duckduckgo.com" not in parsed.netloc:
        return href
    return unquote(parse_qs(parsed.query).get("uddg", [href])[0])


def fetch_ddg_news(query: str, limit: int = DDG_PER_QUERY) -> list[dict]:
    """
    DuckDuckGo(html)で `query` を検索し、タイトル・スニペットを最大limit件返す。
    夜間〜朝の市況・地政学ニュースの補完用。失敗時は警告を出して空リストを返す。
    """
    try:
        resp = cffi_requests.get(DDG_HTML_URL, params={"q": query}, impersonate="chrome", timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"⚠️ DuckDuckGo検索の実行に失敗しました（スキップします）: {e}")
        return []

    items = []
    for div in soup.select("div.result"):
        a = div.select_one("a.result__a")
        snip = div.select_one("a.result__snippet")
        if not a or not a.get("href"):
            continue
        url = _decode_ddg_href(a["href"])
        items.append({
            "title": a.get_text(strip=True),
            "snippet": (snip.get_text(strip=True) if snip else "")[:DDG_SNIPPET_MAX_CHARS],
            "url": url,
            "query": query,
        })
        if len(items) >= limit:
            break
    return items


def fetch_all_news(cutoff: datetime) -> dict:
    """全ニュースソースをベストエフォートで収集し、ソース名をキーにしたdictで返す。"""
    news: dict = {}

    news["kabutan_market"] = fetch_kabutan_market_news(cutoff=cutoff)
    news["yahoo_topics"] = fetch_yahoo_finance_topics(cutoff=cutoff)

    news["google_news"] = []
    seen: set[str] = set()
    for q in OVERNIGHT_QUERIES:
        for item in fetch_google_news_rss(q, cutoff=cutoff):
            if item["url"] not in seen:
                seen.add(item["url"])
                news["google_news"].append(item)
        time.sleep(1)  # 検索エンジンへの負荷軽減のための待機

    news["ddg"] = []
    seen_ddg: set[str] = set()
    for q in OVERNIGHT_QUERIES:
        for item in fetch_ddg_news(q):
            if item["url"] not in seen_ddg:
                seen_ddg.add(item["url"])
                news["ddg"].append(item)
        time.sleep(1)  # 検索エンジンへの負荷軽減のための待機

    return news


# --- Step 4: 判定プロンプトの組み立て & DeepSeek API呼び出し ---
def _fmt_jst_iso(text: str) -> str:
    """'2026-10-01T15:15:00+09:00' 形式の日時文字列を 'MM-DD HH:MM' に整形する。"""
    try:
        return datetime.fromisoformat(text).strftime("%m-%d %H:%M")
    except (ValueError, AttributeError):
        return "--"


def _build_market_check_prompt(market: dict, no_go_rules: list[dict], news: dict, now: datetime) -> str:
    """DeepSeek API用の朝の地合い判定プロンプトを組み立てる（Web検索はPython側で事前実行し、結果を埋め込む前提）。"""

    # ■ 市場データ（yfinance取得分）
    market_lines = []
    for symbol, label, unit in MARKET_SYMBOLS:
        entry = market.get(symbol, {})
        latest = entry.get("latest")
        prev = entry.get("prev_close")
        if latest is None or prev is None:
            market_lines.append(f"  - {label}: データ取得不可")
            continue
        if unit == "yen":
            chg = f"{entry['change']:+.2f}円" if entry.get("change") is not None else "--"
            market_lines.append(f"  - {label}: 直近 {latest:.2f}円（前日終値比 {chg}）")
        else:
            pct = f"{entry['change_pct']:+.2f}%" if entry.get("change_pct") is not None else "--"
            market_lines.append(
                f"  - {label}: 直近 {latest:,.2f}（前日終値比 {pct}・データ基準日 {entry.get('last_close_date', '--')}）"
            )
    nkd = market.get("NKD=F", {})
    nkd_vs_tse = market.get("nkd_vs_tse_pct")
    if nkd.get("latest") is not None and nkd_vs_tse is not None:
        market_lines.append(f"  - 日経225先物(CME)の前日東証終値比: {nkd_vs_tse:+.2f}%（東証終値は ^N225 の前日終値で近似）")
    usdjpy = market.get("JPY=X", {})
    if usdjpy.get("latest") is not None:
        base = usdjpy.get("prev_1500") or usdjpy.get("prev_close")
        if base is not None:
            note = "" if usdjpy.get("prev_1500") else "（前日15:00の5分足が取得できなかったため前日終値で代用）"
            market_lines.append(
                f"  - ドル円の前日15:00時点比: {usdjpy['latest'] - base:+.2f}円（基準値 {base:.2f}円{note}）"
            )

    # ■ 機械判定結果
    if no_go_rules:
        rule_lines = "\n".join(f"  - 【{r['name']}】{r['detail']}" for r in no_go_rules)
        machine_section = (
            "以下のルールに【該当】しています（無条件NO GOの対象。AIの判定は参考情報として扱われます）:\n"
            + rule_lines
        )
    else:
        machine_section = "発動したルールはありません。"

    # ■ ニュース
    kabutan_news = news.get("kabutan_market") or []
    kabutan_section = ""
    if kabutan_news:
        entries = "\n\n".join(
            f"{i}. [{n['datetime'][11:16]}] {n['title']}\n   {n['body']}"
            for i, n in enumerate(kabutan_news, 1)
        )
        kabutan_section = f"\n【株探 市場ニュース（本文つき・新しい順）】\n{entries}"

    yahoo_topics = news.get("yahoo_topics") or []
    yahoo_section = ""
    if yahoo_topics:
        entries = "\n".join(
            f"{i}. [{t['datetime'].strftime('%H:%M') if t['datetime'] else '--:--'}] {t['title']}"
            for i, t in enumerate(yahoo_topics, 1)
        )
        yahoo_section = f"\n【Yahoo!ファイナンス トップ見出し】\n{entries}"

    google_news = news.get("google_news") or []
    google_section = ""
    if google_news:
        entries = "\n".join(
            f"{i}. [{_fmt_jst_iso(g['datetime'])}] {g['title']}"
            for i, g in enumerate(google_news, 1)
        )
        google_section = f"\n【Google News 検索結果（夜間〜朝の関連ニュース）】\n{entries}"

    ddg = news.get("ddg") or []
    ddg_section = ""
    if ddg:
        entries = "\n\n".join(
            f"{i}. {d['title']}\n   {d['snippet']}"
            for i, d in enumerate(ddg, 1)
        )
        ddg_section = f"\n【DuckDuckGo 検索結果（市況・地政学関連）】\n{entries}"

    return f"""
あなたは規律と資産防衛を最優先する日本株のプロデイトレーダー／スイングトレーダーです。
本日の東証寄り付き（9:00）を前に、昨日の15:00以降から現在にかけてのグローバル市場のニュース・指標を精査し、「本日の新規買いエントリーを見送るべきか否か」を冷徹に判定してください。

【取得済みデータ（昨夕15:00〜現在（{now.strftime('%H:%M')}）までにPython側で収集したもの）】
※外部ツールは利用できないため、判定はすべて以下の埋め込みデータのみを根拠として行ってください。

■ 市場データ
{chr(10).join(market_lines)}

■ 機械判定結果（客観的な足切り基準）
{machine_section}
{kabutan_section}
{yahoo_section}
{google_section}
{ddg_section}

【判定ルール】
以下のいずれかに該当する場合は、原則【NO GO（エントリー全面停止）】と判定してください。
- 日経平均先物が前日比 -1.5% 以上の大幅安
- 米国SOX指数が -3.0% 以上の急落、またはNASDAQが -2.0% 以上の下落
- VIX指数が急騰し、リスクオフの投げ売り相場となっている
- 日本市場全体に大きな下押し圧力をかける重大な悪材料ニュース（地政学リスクの急変、想定外のタカ派発言、金融不安など）が未明〜早朝に発生している
※重大な悪材料が無いものの地合いが弱い場合は CAUTION（ロット半減・指値限定）としてください。

【出力形式】
以下のJSONのみを出力してください。
{{
  "verdict": "GO" または "CAUTION" または "NO GO",
  "market_summary": "朝のマーケット環境サマリー（先物騰落率、為替水準、米市場結果）",
  "news_and_warnings": "注目すべき市場ニュース・警戒材料（夜間に何が起きたか）",
  "strategy_impact": "本日のトレード戦略への影響（なぜその判定に至ったか、どのようなリスクが警戒されるか）"
}}
"""


def _parse_json_response(raw_text: str) -> dict:
    """DeepSeekの応答からJSONオブジェクトを抽出する（stock-analyze.py と共通）。"""
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


def evaluate_with_deepseek(prompt: str) -> dict:
    """
    DeepSeek API（OpenAI互換のchat/completions）を直接呼び出して朝の地合い判定を行う。
    JSON出力を強制する response_format を指定し、それでもJSONとして解析できない場合は
    1回だけリトライする（2回目のプロンプトにはJSONのみの出力を強く指示する）。
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY が設定されていません。.env に DeepSeek のAPIキーを設定してください。"
        )

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
                    "あなたは規律と資産防衛を最優先する日本株のプロデイトレーダーです。"
                    "ユーザーのプロンプトに埋め込まれた【取得済みデータ】と【機械判定結果】のみを根拠に判定を行い、"
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
            payload["messages"].append({"role": "assistant", "content": content})
            payload["messages"].append(
                {"role": "user", "content": "前回の応答はJSONとして解析できませんでした。指示されたJSONオブジェクトのみを出力してください。"}
            )


# --- Step 5: 最終判定 & 出力 ---
def decide_verdict(ai_result: dict | None, no_go_rules: list[dict]) -> tuple[str, str]:
    """
    最終判定を返す。機械的な足切り基準に該当した場合、およびAI判定が取得できない場合は
    安全側に倒してNO GOにする。戻り値は (verdict, 補足メッセージ)。
    """
    if no_go_rules:
        return "NO GO", f"機械判定（足切り基準）に{len(no_go_rules)}件該当したため強制NO GO"
    if ai_result is None:
        return "NO GO", "AI判定が取得できなかったため安全側でNO GO"
    verdict = str(ai_result.get("verdict", "")).strip().upper()
    if verdict not in ("GO", "CAUTION", "NO GO"):
        return "NO GO", f"AIの判定が不正でした（「{verdict or '空'}」）ため安全側でNO GO"
    return verdict, ""


def format_result(now: datetime, verdict: str, note: str, ai_result: dict | None, no_go_rules: list[dict]) -> str:
    """判定結果を指定の出力形式（最終判定 / サマリー / 警戒材料 / 戦略への影響）で組み立てる。"""
    ai_result = ai_result or {}
    label = VERDICT_LABELS.get(verdict, verdict)
    parts = [
        f"🌅 朝の市場地合い判定（{now.strftime('%Y-%m-%d %H:%M')} JST）",
        "",
        f"■ 最終判定：【 {label} 】",
    ]
    if note:
        parts.append(f"   ※ {note}")
    if no_go_rules:
        parts.append("")
        parts.append("🚨 発動した機械判定（足切り基準）:")
        for r in no_go_rules:
            parts.append(f"   ✕ {r['name']}: {r['detail']}")
    parts.append("")
    parts.append("■ 朝のマーケット環境サマリー")
    parts.append(ai_result.get("market_summary", "（AI判定が取得できませんでした）"))
    parts.append("")
    parts.append("■ 注目すべき市場ニュース・警戒材料")
    parts.append(ai_result.get("news_and_warnings", "（AI判定が取得できませんでした）"))
    parts.append("")
    parts.append("■ 本日のトレード戦略への影響")
    parts.append(ai_result.get("strategy_impact", "（AI判定が取得できませんでした）"))
    return "\n".join(parts)


def notify_discord(text: str) -> bool:
    """
    DISCORD_WEBHOOK_URL が設定されていれば判定結果をDiscordへ通知する。
    Discordの1メッセージ上限(2000文字)を超える分は切り詰める。失敗時は警告のみ出して継続する。
    """
    if not DISCORD_WEBHOOK_URL:
        return False
    try:
        resp = cffi_requests.post(
            DISCORD_WEBHOOK_URL, json={"content": text[:1900]}, timeout=30
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"⚠️ Discordへの通知に失敗しました: {e}")
        return False


# --- メイン実行処理 ---
def main():
    no_ai = "--no-ai" in sys.argv
    now = datetime.now(JST)
    cutoff = overnight_cutoff(now)
    print(f"🌅 朝の市場地合い判定を開始します（{now.strftime('%Y-%m-%d %H:%M')} JST・ニュース対象期間: {cutoff.strftime('%m/%d %H:%M')} 以降）...")

    # 1. 市場データ収集
    print("📊 市場データを収集しています...")
    market = fetch_market_data()

    # 2. 機械的NO GO判定
    no_go_rules = mechanical_no_go_rules(market)
    if no_go_rules:
        print(f"🚨 機械判定（足切り基準）に{len(no_go_rules)}件該当しました:")
        for r in no_go_rules:
            print(f"   ✕ {r['name']}: {r['detail']}")
    else:
        print("✅ 機械判定（足切り基準）: 該当なし")

    # 3. ニュース収集
    print("📰 ニュースを収集しています...")
    news = fetch_all_news(cutoff)
    print(
        f"   → 株探 {len(news.get('kabutan_market', []))}件 / Yahoo!ファイナンス {len(news.get('yahoo_topics', []))}件"
        f" / Google News {len(news.get('google_news', []))}件 / DuckDuckGo {len(news.get('ddg', []))}件"
    )

    # 4. 判定プロンプトの組み立て & AI判定
    prompt = _build_market_check_prompt(market, no_go_rules, news, now)
    ai_result = None
    ai_error = None
    if no_ai:
        print("ℹ️ --no-ai のためAI判定をスキップします。収集データ入りプロンプトを表示します:\n")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
    else:
        print("🤖 DeepSeek API で地合いを判定しています...")
        try:
            ai_result = evaluate_with_deepseek(prompt)
        except RuntimeError as e:
            ai_error = str(e)
            print(f"❌ {e}")

    # 5. 最終判定 & 出力
    verdict, note = decide_verdict(ai_result, no_go_rules)
    if ai_error and not no_go_rules:
        note = f"{note}（{ai_error}）"
    text = format_result(now, verdict, note, ai_result, no_go_rules)
    print()
    print(text)

    # 6. Discord通知（任意・デバッグモードでは通知しない）
    if not no_ai and notify_discord(text):
        print("\n📨 Discordへ通知しました。")


if __name__ == "__main__":
    main()
