#!/usr/bin/env python3
"""pullback_buy_signals.csv から STRONG_BUY / BUY 銘柄を抽出し、
stock-analyze.py を順次実行するランナー。

環境変数:
  MAX_ANALYZE_STOCKS: 1回の実行で分析する最大銘柄数 (デフォルト: 30)
  SIGNALS_CSV: 銘柄リストCSVのパス (デフォルト: pullback_buy_signals.csv)。
               スクリーナーを実行せず analyze の動作確認だけ行う場合に
               テスト用フィクスチャCSVを指定するために使う。
"""
import csv
import os
import subprocess
import sys

CSV_PATH = os.getenv("SIGNALS_CSV", "pullback_buy_signals.csv")
TARGET_RATINGS = {"STRONG_BUY", "BUY"}
MAX_STOCKS = int(os.getenv("MAX_ANALYZE_STOCKS", "30"))


def main() -> int:
    if not os.path.isfile(CSV_PATH):
        print(f"⏭️ {CSV_PATH} が存在しないため分析をスキップします。")
        return 0

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    targets = [
        r
        for r in rows
        if (r.get("analyst_rating") or "").strip().upper() in TARGET_RATINGS
    ]
    # 上限超過時に STRONG_BUY が除外されないよう先に分析する
    targets.sort(
        key=lambda r: (r.get("analyst_rating") or "").strip().upper() != "STRONG_BUY"
    )

    if not targets:
        print("対象銘柄（STRONG_BUY / BUY）はありません。")
        return 0

    if len(targets) > MAX_STOCKS:
        print(
            f"⚠️ 対象 {len(targets)} 銘柄 > 上限 {MAX_STOCKS} のため、"
            "STRONG_BUY 優先で絞り込みます（MAX_ANALYZE_STOCKS で変更可）。"
        )
        targets = targets[:MAX_STOCKS]

    if not os.getenv("KIMI_MODEL_API_KEY") and not os.getenv("KIMI_CLI_MODEL"):
        print("⚠️ KIMI_MODEL_API_KEY が未設定です。定性評価は失敗する可能性があります。")

    print(f"=== {len(targets)} 銘柄の分析を開始 ===")
    failures = 0
    for i, row in enumerate(targets, 1):
        ticker = (row.get("ticker") or "").strip()
        if not ticker:
            continue
        rating = (row.get("analyst_rating") or "").strip().upper()
        print(f"\n--- [{i}/{len(targets)}] {ticker} ({rating}) ---", flush=True)
        proc = subprocess.run([sys.executable, "stock-analyze.py", ticker])
        if proc.returncode != 0:
            failures += 1
            print(f"⚠️ {ticker} の分析に失敗 (exit {proc.returncode})。次へ進みます。", flush=True)

    print(f"\n=== 分析完了: {len(targets) - failures}/{len(targets)} 成功 ===")
    # 全銘柄が失敗した場合のみ異常終了（認証ミス等の全体的な問題を検知するため）
    return 1 if targets and failures == len(targets) else 0


if __name__ == "__main__":
    sys.exit(main())
