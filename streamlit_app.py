import math
import streamlit as st
import pandas as pd
from binance.client import Client
from streamlit_autorefresh import st_autorefresh

# —— CONFIG ——  
THRESHOLD = 400
INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d", "1w"]

# 从 Streamlit Secrets 读取 API Key
API_KEY    = st.secrets["BINANCE_API_KEY"]
API_SECRET = st.secrets["BINANCE_API_SECRET"]

# —— 核心计算函数 ——  
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
        klines = client.get_klines(symbol=symbol, interval=iv, limit=100)
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
st_autorefresh(interval=10_000, key="refresh")

# 初始化 Binance 客户端并拉取所有交易对
client      = Client()
symbols_inf = client.get_exchange_info()["symbols"]

# 过滤：symbol 以 USDT 结尾、状态为 TRADING、可现货交易
all_usdt = [
    s["symbol"]
    for s in symbols_inf
    if s["symbol"].endswith("USDT")
       and s.get("status") == "TRADING"
       and s.get("isSpotTradingAllowed", False)
]

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

# 抓取并计算所有选中交易对的数据
with st.spinner("Fetching & calculating..."):
    data = []
    for sym in selected:
        data += fetch_whale_data_for(sym)

df = pd.DataFrame(data)

# 将 Timeframe 列转为“有序分类”，按 INTERVALS 顺序排列
df["Timeframe"] = pd.Categorical(
    df["Timeframe"],
    categories=INTERVALS,
    ordered=True
)

# 保留四位小数
df["WhalePumpValue"] = df["WhalePumpValue"].map(lambda x: float(f"{x:.4f}"))

# 排序：先按 Symbol，再按 Timeframe；警报优先时先按 Status
if sort_mode.startswith("警报优先"):
    df = df.sort_values(["Status", "Timeframe"], ascending=[True, True])
else:
    df = df.sort_values(["Symbol", "Timeframe"], ascending=[True, True])

# 展示表格
st.subheader("📋 数据一览")
st.dataframe(df, use_container_width=True)
