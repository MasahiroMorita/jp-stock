#!/usr/bin/env bash
# GitHub Actions (ubuntu) 上で実行されるジョブの外部通信をすべて VPN Gate の日本リレー経由にする。
#
# 背景: 株探(kabutan.jp)はデータセンターIPをWAFで拒否(HTTP 405)するため、CIランナーからは
# 直接アクセスできない。VPN Gate の日本リレーは日本のISP/家庭用IPから egress されるため
# ブロックを回避できる。リレーはボランティア運営で不安定なため、スコア順に複数台試行する。
#
# 注意: このスクリプトはフルトンネル化する。つまり Notion/Kimi の API キーを含む通信も
# リレーを通る。TLS により通信内容が傍受されることはないが、通信相手のメタデータは
# リレー運用者に見える点と、リレーが途中で切断した場合はジョブ全体の通信が失われる点に注意。
set -euo pipefail

VPNGATE_API="https://www.vpngate.net/api/iphone/"
# 動作確認に使う株探のURL（決算速報一覧）。CIのIPブロック回避が目的のためこれが200になることを確認する
PROBE_URL="https://kabutan.jp/stock/news?code=2170&nmode=2"
MAX_TRY_SERVERS="${MAX_TRY_SERVERS:-10}"
CONNECT_WAIT_SEC="${CONNECT_WAIT_SEC:-40}"

log() { echo "[vpn-gate] $*"; }

if [ "$(id -u)" -ne 0 ]; then
    echo "このスクリプトは root で実行してください（CI: sudo bash ci/setup-vpn-gate.sh）" >&2
    exit 1
fi

if ! command -v openvpn >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq openvpn curl
fi

# --- VPN Gate リレー一覧を取得（日本・TCPリレーのみスコア順に候補化） ---
curl -fsSL --retry 3 --max-time 60 "$VPNGATE_API" -o /tmp/vpngate.csv
log "リレー一覧: $(grep -c ',' /tmp/vpngate.csv) 件"

MAX_TRY_SERVERS="$MAX_TRY_SERVERS" python3 - > /tmp/vpngate_candidates.tsv <<'PY'
import base64, os, re

max_servers = int(os.environ["MAX_TRY_SERVERS"])
with open("/tmp/vpngate.csv", encoding="utf-8", errors="replace") as f:
    lines = [ln for ln in f.read().splitlines() if ln.strip()]

candidates = []
for ln in lines:
    if ln.startswith(("*vpn_servers", "#")):
        continue
    parts = ln.split(",")
    if len(parts) < 15 or parts[5] != "Japan":
        continue
    try:
        cfg = base64.b64decode(parts[14]).decode("utf-8", "replace")
    except Exception:
        continue
    # CIのデータセンター出口ではTCP(443)が最も通りやすい
    if not re.search(r"^proto tcp", cfg, flags=re.M):
        continue
    m = re.search(r"^remote ([^ ]+) (\d+)", cfg, flags=re.M)
    if not m:
        continue
    score = int(parts[2]) if parts[2].isdigit() else 0
    candidates.append((score, parts[0], m.group(1), m.group(2)))

candidates.sort(key=lambda c: -c[0])
for score, host, ip, port in candidates[:max_servers]:
    print(f"{host}\t{ip}\t{port}")
PY

if [ ! -s /tmp/vpngate_candidates.tsv ]; then
    log "ERROR: 日本のTCPリレーが見つかりません"
    exit 1
fi
log "候補リレー: $(wc -l < /tmp/vpngate_candidates.tsv) 台（最大 $MAX_TRY_SERVERS 台まで試行）"

write_ovpn_config() {
    local host="$1"
    HOST="$host" python3 - > /tmp/vpngate.ovpn <<'PY'
import base64, os
with open("/tmp/vpngate.csv", encoding="utf-8", errors="replace") as f:
    lines = [ln for ln in f.read().splitlines() if ln.strip()]
for ln in lines:
    parts = ln.split(",")
    if len(parts) >= 15 and parts[0] == os.environ["HOST"]:
        print(base64.b64decode(parts[14]).decode("utf-8", "replace"))
        break
PY
    {
        # サーバー側の指示に関わらず、デフォルトルートを必ずトンネル側に取る（フルトンネル化）
        echo "pull-filter ignore \"redirect-gateway\""
        echo "redirect-gateway def1"
        echo "auth-nocache"
    } >> /tmp/vpngate.ovpn
}

stop_vpn() {
    if [ -f /tmp/vpngate.pid ]; then
        kill "$(cat /tmp/vpngate.pid)" 2>/dev/null || true
        rm -f /tmp/vpngate.pid
        sleep 2
    fi
}

route_via_tun() {
    # デフォルト経路が tun デバイス側になっているか（8.8.8.8 はルーティング確認用の宛先）
    ip route get 8.8.8.8 2>/dev/null | grep -q 'dev tun'
}

attempt=0
while IFS=$'\t' read -r host ip port; do
    attempt=$((attempt + 1))
    log "[$attempt] 接続試行: $host ($ip:$port)"
    stop_vpn
    write_ovpn_config "$host"
    rm -f /tmp/vpngate.log
    openvpn --config /tmp/vpngate.ovpn --daemon --writepid /tmp/vpngate.pid --log-append /tmp/vpngate.log

    connected=0
    for _ in $(seq 1 "$CONNECT_WAIT_SEC"); do
        if route_via_tun; then connected=1; break; fi
        sleep 1
    done
    if [ "$connected" -ne 1 ]; then
        log "  ルート確立が ${CONNECT_WAIT_SEC}s 以内に完了しませんでした。次のリレーへ。"
        tail -n 5 /tmp/vpngate.log 2>/dev/null || true
        continue
    fi

    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$PROBE_URL" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then
        log "✅ 接続成功: $host 経由で $PROBE_URL が HTTP 200 で応答します"
        exit 0
    fi
    log "  株探の応答が HTTP $code でした。次のリレーへ。"
    stop_vpn
done < /tmp/vpngate_candidates.tsv

log "ERROR: ${attempt}台すべてのリレーで接続に失敗しました"
exit 1
