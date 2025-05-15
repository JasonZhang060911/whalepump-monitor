import math
import requests
import streamlit as st
import pandas as pd
from binance.client import Client
from streamlit_autorefresh import st_autorefresh

# —— CONFIG ——  
THRESHOLD = 400
INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# —— 从 Secrets 里读 Key/Secret 和可选代理 ——  
API_KEY    = st.secrets["BINANCE_API_KEY"]
API_SECRET = st.secrets["BINANCE_API_SECRET"]
# 如果有设置代理（比如 “http://127.0.0.1:7890”），填在 Secrets 的 BINANCE_PROXY
PROXY = st.secrets.get("BINANCE_PROXY", None)
HEADERS = {"User-Agent": "Mozilla/5.0"}    # 加一个常见浏览器头

def get_requests_kwargs():
    """统一返回 requests.get() 的补充参数：headers + 可能的 proxies"""
    kw = {"timeout": 10, "headers": HEADERS}
    if PROXY:
        kw["proxies"] = {"https": PROXY}
    return kw

# —— 缓存加载所有 USDT 交易对 ——  
@st.cache_data(ttl=300)
def load_all_usdt_symbols():
    """
    优先用原生 requests 拉 exchangeInfo，
    如果报错 or 拿不到 symbols，再回退去用 python-binance 客户端。
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    # 1) 原生请求
    try:
        resp = requests.get(url, **get_requests_kwargs())
        resp.raise_for_status()
        data = resp.json()
        if "symbols" not in data:
            raise RuntimeError(f"ExchangeInfo 返回无 symbols：{data}")
        syms = [
            s["symbol"] for s in data["symbols"]
            if s["symbol"].endswith("USDT")
               and s.get("status") == "TRADING"
               and s.get("isSpotTradingAllowed", False)
        ]
        # 如果拿到了，就直接返回
        if syms:
            return syms

    except Exception as err_raw:
        # 原生接口拉取失败（比如 451、502、超时 等），尝试 python-binance
        try:
            client = Client(API_KEY, API_SECRET)
            info = client.get_exchange_info()
            return [
                s["symbol"] for s in info["symbols"]
                if s["symbol"].endswith("USDT")
                   and s.get("status") == "TRADING"
                   and s.get("isSpotTradingAllowed", False)
            ]
        except Exception as err_fb:
            # 双重失败
            raise RuntimeError(
                f"原生请求失败: {err_raw}\npython-binance 回退失败: {err_fb}"
            )

# —— 核心计算函数（与你原来一模一样） ——  
def calculate_whale_pump(lows: list[float], closes: list[float]) -> float:
    if len(lows) < 90:
        return 0.0

    low = pd.Series(lows)
    diff1 = (low - low.shift(1)).abs().fillna(0.0)
    diff2 = (low - low.shift(1)).clip(lower=0).fillna(0.0)

    def xsa_series(src, length, weight):
        prev_sum = prev_out = 0.0
        res = []
        for i, v in enumerate(src):
            v = 0.0 if math.isnan(v) else v
            if i == 0:
                s = v
            else:
                s = prev_sum - (src[i-length] if i >= length else 0.0) + v
            ma = s / length
            out = ma if i == 0 else (v * weight + prev_out * (length - weight)) / length
            res.append(out)
            prev_sum, prev_out = s, out
        return res

    x1 = xsa_series(diff1.tolist(), 3, 1)
    x2 = xsa_series(diff2.tolist(), 3, 1)
    ratio = [(x1[i]/x2[i]*100 if x2[i] != 0 else 0.0) for i in range(len(low))]
    ewm = pd.Series(ratio).mul(10).ewm(span=3, adjust=False).mean()
    min38 = low.rolling(38, min_periods=38).min()
    max38 = ewm.rolling(38, min_periods=38).max()
    ok90  = (~low.rolling(90, min_periods=90).min().isna()).astype(int)
    base  = ((ewm + max38 * 2) / 2).where(low <= min38, 0.0)
    val   = base.ewm(span=3, adjust=False).mean().div(618).mul(ok90)
    nonz  = val[val != 0]
    return round(nonz.iloc[-1] if not nonz.empty else 0.0, 4)

def fetch_whale_data_for(symbol: str) -> list[dict]:
    client = Client(API_KEY, API_SECRET)
    rows = []
    for iv in INTERVALS:
        kl_url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": iv, "limit": 100}
        klines = requests.get(kl_url, params=params, **get_requests_kwargs()).json()
        lows   = [float(k[3]) for k in klines]
        closes = [float(k[4]) for k in klines]
        wp     = calculate_whale_pump(lows, closes)
        rows.append({
            "Symbol":         symbol,
            "Timeframe":      iv,
            "WhalePumpValue": wp,
            "Status":         "🟡 警报" if wp > THRESHOLD else "🟢 正常"
        })
    return rows

# —— Streamlit UI ——  
st.set_page_config(page_title="Whale Pump Monitor", layout="wide")
st.title("🦈 Whale Pump Monitor")
st_autorefresh(interval=60_000, key="refresh")

# 加载交易对列表
try:
    all_usdt = load_all_usdt_symbols()
except Exception as e:
    st.error(f"加载交易对列表失败：{e}")
    st.stop()

# 侧边栏：多选交易对
selected = st.sidebar.multiselect(
    "请选择要监控的交易对 (USDT)", all_usdt, default=all_usdt[:20]
)
if not selected:
    st.sidebar.warning("请至少选择一个交易对")
    st.stop()

# 侧边栏：排序方式
sort_mode = st.sidebar.radio(
    "排序方式", ("默认 (Symbol→Timeframe)", "警报优先 (🟡→🟢)")
)

# 抓取并计算
with st.spinner("Fetching & calculating..."):
    data = []
    for sym in selected:
        data += fetch_whale_data_for(sym)

df = pd.DataFrame(data)
df["Timeframe"] = pd.Categorical(df["Timeframe"], categories=INTERVALS, ordered=True)
df["WhalePumpValue"] = df["WhalePumpValue"].map(lambda x: float(f"{x:.4f}"))

if sort_mode.startswith("警报优先"):
    df = df.sort_values(["Status", "Timeframe"], ascending=[True, True])
else:
    df = df.sort_values(["Symbol", "Timeframe"], ascending=[True, True])

st.subheader("📋 数据一览")
st.dataframe(df, use_container_width=True)
