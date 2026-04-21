app_code = r'''
import os
import re
import math
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
import torch.nn as nn
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="QuantGenius AI",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# ROOT PATHS
# =========================================================
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path("/content/quantgenius-demo")

TABLES_DIR = PROJECT_ROOT / "tables"
MODELS_DIR = PROJECT_ROOT / "models"
MAPPING_PATH = PROJECT_ROOT / "step11_asset_mapping.json"

DEVICE = torch.device("cpu")
NEWS_LOOKBACK_DAYS = 45

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
:root {
    --bg1: #04101c;
    --bg2: #081726;
    --panel: rgba(255,255,255,0.045);
    --border: rgba(255,255,255,0.09);
    --text: #f7fbff;
    --muted: #bfd0e3;
}

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(1200px 500px at 20% 0%, rgba(96,165,250,0.10), transparent 55%),
        radial-gradient(900px 500px at 90% 10%, rgba(94,234,212,0.08), transparent 55%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"] {
    display: none !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}
.block-container {
    max-width: 100% !important;
    padding-top: 0.7rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    padding-bottom: 2rem !important;
}

.main-hero {
    background: linear-gradient(135deg, rgba(96,165,250,0.14), rgba(16,185,129,0.08));
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 22px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}
.main-hero h1 {
    margin: 0;
    color: white;
    font-size: 46px;
    font-weight: 800;
}
.main-hero p {
    margin-top: 14px;
    color: #d9e7f5;
    font-size: 18px;
    line-height: 1.75;
}

.section-title {
    color: white;
    font-size: 22px;
    font-weight: 800;
    margin-top: 12px;
    margin-bottom: 12px;
}

.card-shell {
    background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.03));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.16);
}

.result-label {
    display: inline-block;
    padding: 10px 16px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 14px;
    margin-top: 8px;
}
.label-bull { background: rgba(52,211,153,0.14); color: #86efac; border: 1px solid rgba(52,211,153,0.3); }
.label-neutral { background: rgba(250,204,21,0.12); color: #fde68a; border: 1px solid rgba(250,204,21,0.3); }
.label-bear { background: rgba(248,113,113,0.12); color: #fca5a5; border: 1px solid rgba(248,113,113,0.3); }

.explain-summary {
    background: rgba(94,234,212,0.06);
    border-left: 4px solid rgba(94,234,212,0.9);
    border-radius: 16px;
    padding: 20px;
    color: #ecfffb;
    line-height: 1.8;
    margin-bottom: 18px;
}

.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    min-height: 158px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}
.metric-title {
    font-size: 12px;
    color: #b7c4d6;
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.metric-value {
    font-size: 34px;
    font-weight: 800;
    color: white;
}
.metric-sub {
    margin-top: 10px;
    font-size: 14px;
    color: #d0d9e5;
    line-height: 1.6;
}

.bullet-card ul {
    margin-top: 8px;
    margin-bottom: 0;
    padding-left: 22px;
}
.bullet-card li {
    margin-bottom: 12px;
    color: #e4edf8;
    line-height: 1.7;
}

label, .stSelectbox label, .stMarkdown p {
    color: var(--muted) !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    min-height: 52px !important;
}
div[data-baseweb="select"] * {
    color: #f8fbff !important;
    fill: #f8fbff !important;
}
div[data-baseweb="popover"] * {
    color: #f8fbff !important;
    background: #102033 !important;
}
button[kind="primary"] {
    border-radius: 14px !important;
    font-weight: 700 !important;
}

.dark-table-wrap {
    background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.03));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px;
    overflow-x: auto;
    margin-bottom: 16px;
}
.dark-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}
.dark-table thead th {
    background: rgba(255,255,255,0.06);
    color: #d9e6f4;
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.dark-table tbody td {
    color: #eef5fc;
    padding: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    vertical-align: top;
}
.dark-table tbody tr:hover {
    background: rgba(255,255,255,0.03);
}

.small-note {
    color: #9fb0c5;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def latest_match(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {folder}")
    return matches[-1]

@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def get_current_utc():
    return datetime.now(timezone.utc)

def squash_tanh(x, scale=1.0):
    return math.tanh(x / scale)

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def label_class(label: str) -> str:
    if "Bullish" in label:
        return "label-bull"
    if "Bearish" in label:
        return "label-bear"
    return "label-neutral"

def get_alpha_vantage_key():
    try:
        key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if key:
        return str(key).strip()

    key_file = PROJECT_ROOT / "alpha_vantage_key.txt"
    if key_file.exists():
        txt = key_file.read_text().strip()
        if txt:
            return txt

    return ""

# =========================================================
# LOAD PROJECT ARTIFACTS
# =========================================================
CELL10_META = load_json(latest_match(TABLES_DIR, "cell10_meta_*.json"))
SEQ_META = load_json(latest_match(TABLES_DIR, "cell8_seq_meta_*.json"))
ASSET_MAPPING = load_json(MAPPING_PATH)

LOOKBACK = int(SEQ_META["lookback"])
SCALER_PARAMS = SEQ_META["scaler_params_per_horizon"]

ASSET_UNIVERSE = ASSET_MAPPING["asset_universe"]
TICKER_TO_ASSET_ID = {str(k).upper(): int(v) for k, v in ASSET_MAPPING["ticker_to_asset_id"].items()}

# =========================================================
# MODEL DEFINITIONS
# =========================================================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, n_assets, asset_emb_dim=8, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, asset_emb_dim)
        in_dim = n_features + asset_emb_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, asset_id):
        emb = self.asset_emb(asset_id)
        emb = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, emb], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        t = x.size(1)
        x = x + self.pe[:, :t, :]
        return self.dropout(x)

class TransformerRegressor(nn.Module):
    def __init__(self, n_features, lookback=60, d_model=128, n_heads=8, n_layers=3, dropout=0.15, ff_mult=4):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model, max_len=lookback + 5, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.in_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        last = h[:, -1, :]
        mean = h.mean(dim=1)
        z = torch.cat([last, mean], dim=1)
        return self.head(z).squeeze(-1)

def extract_state_dict_and_cfg(ckpt):
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"], ckpt.get("config", ckpt.get("cfg", {}))
        if "state_dict" in ckpt:
            return ckpt["state_dict"], ckpt.get("config", ckpt.get("cfg", {}))
    return ckpt, {}

def remap_transformer_keys_if_needed(sd):
    if "pos.pe" in sd:
        return sd
    if "pe" in sd:
        sd2 = dict(sd)
        sd2["pos.pe"] = sd2.pop("pe")
        return sd2
    return sd

@st.cache_resource
def load_models():
    models = {}

    lstm_path = MODELS_DIR / "lstm_UPDATEDv3_short_3d_20260117_145043.pt"
    ckpt = torch.load(lstm_path, map_location="cpu", weights_only=False)
    sd, _ = extract_state_dict_and_cfg(ckpt)

    emb_w = sd["asset_emb.weight"]
    n_assets = emb_w.shape[0]
    asset_emb_dim = emb_w.shape[1]
    hidden = sd["lstm.weight_ih_l0"].shape[0] // 4
    layers = 0
    while f"lstm.weight_ih_l{layers}" in sd:
        layers += 1

    lstm_model = LSTMRegressor(
        n_features=44,
        n_assets=n_assets,
        asset_emb_dim=asset_emb_dim,
        hidden=hidden,
        layers=layers,
        dropout=0.2,
    ).to(DEVICE)
    lstm_model.load_state_dict(sd, strict=True)
    lstm_model.eval()
    models["lstm_short_3d"] = lstm_model

    tr_path = MODELS_DIR / "transformer_reg_intermediate_15d_20260117_153517.pt"
    ckpt = torch.load(tr_path, map_location="cpu", weights_only=False)
    sd, cfg = extract_state_dict_and_cfg(ckpt)
    sd = remap_transformer_keys_if_needed(sd)

    tr_model = TransformerRegressor(
        n_features=44,
        lookback=int(cfg.get("lookback", 60)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 8)),
        n_layers=int(cfg.get("n_layers", 3)),
        dropout=float(cfg.get("dropout", 0.15)),
        ff_mult=int(cfg.get("ff_mult", 4)),
    ).to(DEVICE)
    tr_model.load_state_dict(sd, strict=True)
    tr_model.eval()
    models["transformer_intermediate_15d"] = tr_model

    xgb_path = MODELS_DIR / "xgb_reg_y_volnorm_intermediate_10d_cuda_20260117_142507.json"
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))
    models["xgb_intermediate_10d"] = booster

    return models

MODELS = load_models()

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EMBEDDER = load_embedder()

# =========================================================
# MARKET DATA + FEATURES
# =========================================================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_market_data(ticker: str) -> pd.DataFrame:
    last_error = None

    for _ in range(3):
        try:
            hist = yf.download(
                ticker,
                period="2y",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if hist is not None and not hist.empty:
                hist = hist.reset_index()
                hist.columns = [
                    "_".join([str(c) for c in col if str(c) != ""]).strip("_")
                    if isinstance(col, tuple) else str(col)
                    for col in hist.columns
                ]

                rename_map = {}
                for c in hist.columns:
                    cl = c.lower()
                    if cl.startswith("date"):
                        rename_map[c] = "Date"
                    elif cl.startswith("open"):
                        rename_map[c] = "Open"
                    elif cl.startswith("high"):
                        rename_map[c] = "High"
                    elif cl.startswith("low"):
                        rename_map[c] = "Low"
                    elif cl.startswith("close"):
                        rename_map[c] = "Close"
                    elif cl.startswith("volume"):
                        rename_map[c] = "Volume"

                hist = hist.rename(columns=rename_map)
                required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                hist = hist[required].copy()
                hist["Date"] = pd.to_datetime(hist["Date"])
                hist = hist.sort_values("Date").reset_index(drop=True)
                return hist

            last_error = f"No market data returned for {ticker}"

        except Exception as e:
            last_error = str(e)
            time.sleep(1.5)

    raise ValueError(
        f"Market data for {ticker} is temporarily unavailable. "
        f"Yahoo Finance may be rate-limiting requests. Details: {last_error}"
    )

def build_features(hist: pd.DataFrame):
    df = hist.copy()

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_2"] = df["Close"].pct_change(2)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["log_ret_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_ret_3"] = np.log(df["Close"] / df["Close"].shift(3))
    df["log_ret_5"] = np.log(df["Close"] / df["Close"].shift(5))

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_range"] = (df["Close"] - df["Open"]) / df["Open"]
    df["high_close"] = df["High"] / df["Close"]
    df["low_close"] = df["Low"] / df["Close"]
    df["open_close"] = df["Open"] / df["Close"]

    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
        df[f"close_sma_ratio_{w}"] = df["Close"] / df[f"sma_{w}"]

    for w in [5, 10, 20]:
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
        df[f"close_ema_ratio_{w}"] = df["Close"] / df[f"ema_{w}"]

    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["ret_1"].rolling(w).std()
        df[f"mom_{w}"] = df["Close"] / df["Close"].shift(w) - 1

    df["vol_chg_1"] = df["Volume"].pct_change(1)
    df["vol_chg_5"] = df["Volume"].pct_change(5)

    for w in [5, 20]:
        df[f"vol_sma_{w}"] = df["Volume"].rolling(w).mean()
        df[f"vol_ratio_{w}"] = df["Volume"] / df[f"vol_sma_{w}"]

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper_ratio"] = (bb_mid + 2 * bb_std) / df["Close"]
    df["bb_lower_ratio"] = (bb_mid - 2 * bb_std) / df["Close"]
    df["bb_width"] = (4 * bb_std) / df["Close"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["rsi_14_scaled"] = (100 - (100 / (1 + rs))) / 100.0

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["close_rank_20"] = df["Close"].rolling(20).rank(pct=True)
    df["ret_z_20"] = (df["ret_1"] - df["ret_1"].rolling(20).mean()) / (df["ret_1"].rolling(20).std() + 1e-12)
    df["vol_z_20"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / (df["Volume"].rolling(20).std() + 1e-12)

    feature_cols = [
        "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
        "log_ret_1", "log_ret_3", "log_ret_5",
        "hl_range", "oc_range", "high_close", "low_close", "open_close",
        "close_sma_ratio_5", "close_sma_ratio_10", "close_sma_ratio_20", "close_sma_ratio_50",
        "close_ema_ratio_5", "close_ema_ratio_10", "close_ema_ratio_20",
        "vol_5", "vol_10", "vol_20",
        "mom_5", "mom_10", "mom_20",
        "vol_chg_1", "vol_chg_5", "vol_ratio_5", "vol_ratio_20",
        "bb_upper_ratio", "bb_lower_ratio", "bb_width",
        "rsi_14_scaled",
        "macd", "macd_signal", "macd_hist",
        "close_rank_20", "ret_z_20", "vol_z_20",
        "sma_5", "sma_10", "sma_20", "ema_10",
    ]

    feat_df = df[["Date"] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat_df, feature_cols

def scale_and_window(feat_df, feature_cols):
    scaler_cfg = SCALER_PARAMS["short_3d"]
    mean_ = np.array(scaler_cfg["mean_"], dtype=np.float32)
    scale_ = np.array(scaler_cfg["scale_"], dtype=np.float32)

    x = feat_df[feature_cols].values.astype(np.float32)
    x_scaled = (x - mean_) / (scale_ + 1e-12)

    x_seq = x_scaled[-LOOKBACK:]
    x_flat = x_seq.reshape(1, -1)
    return np.expand_dims(x_seq, axis=0), x_flat

# =========================================================
# QUANT MODELS
# =========================================================
def run_quant_models(ticker: str, feat_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    x_seq, x_flat = scale_and_window(feat_df, feature_cols)
    asset_id = TICKER_TO_ASSET_ID[ticker]

    booster = MODELS["xgb_intermediate_10d"]
    n_expected = booster.num_features()
    x_used = x_flat.copy()
    if x_used.shape[1] < n_expected:
        pad = np.zeros((x_used.shape[0], n_expected - x_used.shape[1]), dtype=x_used.dtype)
        x_used = np.concatenate([x_used, pad], axis=1)
    elif x_used.shape[1] > n_expected:
        x_used = x_used[:, :n_expected]
    dmat = xgb.DMatrix(x_used)
    ml_pred = float(booster.predict(dmat).reshape(-1)[0])

    with torch.no_grad():
        xb = torch.from_numpy(x_seq.astype(np.float32)).to(DEVICE)
        aid = torch.from_numpy(np.array([asset_id], dtype=np.int64)).to(DEVICE)
        y_lstm_norm = MODELS["lstm_short_3d"](xb, aid).cpu().numpy().reshape(-1)[0]

    mu = float(CELL10_META["best_models"]["lstm"]["short_3d"]["target_norm_mu"])
    sd = float(CELL10_META["best_models"]["lstm"]["short_3d"]["target_norm_sd"])
    dl_pred = float(y_lstm_norm * sd + mu)

    with torch.no_grad():
        xb = torch.from_numpy(x_seq.astype(np.float32)).to(DEVICE)
        y_tr_norm = MODELS["transformer_intermediate_15d"](xb).cpu().numpy().reshape(-1)[0]
    tr_pred = float(y_tr_norm * 0.05732867504361837 + 0.013723999097657915)

    return {
        "ml_pred": ml_pred,
        "dl_pred": dl_pred,
        "transformer_pred": tr_pred,
        "latest_date": str(feat_df["Date"].iloc[-1].date()),
    }

# =========================================================
# LIVE NEWS + SEC
# =========================================================
@st.cache_data(ttl=60 * 60 * 6)
def load_ticker_cik_map():
    headers = {"User-Agent": "QuantGenius Demo contact@example.com"}
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    mapping = {}
    for entry in data.values():
        t = str(entry.get("ticker", "")).upper().strip()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if t:
            mapping[t] = cik
    return mapping

def fetch_live_docs(ticker: str):
    api_key = get_alpha_vantage_key()
    docs = []

    news_status = {
        "source": "Alpha Vantage",
        "enabled": bool(api_key),
        "status": "not_called",
        "message": "",
        "items_returned": 0,
        "request_url": "NEWS_SENTIMENT"
    }

    sec_status = {
        "source": "SEC",
        "enabled": True,
        "status": "not_called",
        "message": "",
        "items_returned": 0,
        "ticker_found": ticker.upper().strip(),
        "cik_found": None
    }

    # Alpha Vantage
    if api_key:
        try:
            end_dt = get_current_utc()
            start_dt = end_dt - timedelta(days=NEWS_LOOKBACK_DAYS)
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": start_dt.strftime("%Y%m%dT%H%M"),
                "time_to": end_dt.strftime("%Y%m%dT%H%M"),
                "limit": "50",
                "sort": "LATEST",
                "apikey": api_key,
            }

            r = requests.get("https://www.alphavantage.co/query", params=params, timeout=60)
            data = r.json()

            if "feed" in data:
                news_status["status"] = "ok"
                news_status["items_returned"] = len(data["feed"])

                for item in data["feed"]:
                    docs.append({
                        "source": "NEWS",
                        "ticker": ticker,
                        "date": item.get("time_published", ""),
                        "title": item.get("title", ""),
                        "text": normalize_text(f"{item.get('title','')} {item.get('summary','')}"),
                        "sentiment_score": safe_float(item.get("overall_sentiment_score", 0.0), 0.0),
                        "url": item.get("url", ""),
                    })

            elif "Note" in data:
                news_status["status"] = "rate_limited"
                news_status["message"] = str(data["Note"])
            elif "Information" in data:
                news_status["status"] = "info"
                news_status["message"] = str(data["Information"])
            elif "Error Message" in data:
                news_status["status"] = "error"
                news_status["message"] = str(data["Error Message"])
            else:
                news_status["status"] = "unexpected"
                news_status["message"] = str(data)[:500]

        except Exception as e:
            news_status["status"] = "exception"
            news_status["message"] = str(e)

    # SEC
    try:
        ticker_cik = load_ticker_cik_map()
        cik10 = ticker_cik.get(ticker.upper().strip())
        sec_status["cik_found"] = cik10

        if not cik10:
            sec_status["status"] = "no_cik"
            sec_status["message"] = f"No CIK mapping found for {ticker}"
            return docs, news_status, sec_status

        headers = {
            "User-Agent": "QuantGenius Research Demo contact@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov"
        }

        url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        r = requests.get(url, headers=headers, timeout=30)

        if r.status_code != 200:
            sec_status["status"] = f"http_{r.status_code}"
            sec_status["message"] = r.text[:300]
            return docs, news_status, sec_status

        data = r.json()
        recent = (data.get("filings", {}) or {}).get("recent", {}) or {}

        df = pd.DataFrame({
            "form": recent.get("form", []),
            "filingDate": recent.get("filingDate", []),
            "primaryDocDescription": recent.get("primaryDocDescription", []),
        })

        if df.empty:
            sec_status["status"] = "empty_recent"
            sec_status["message"] = "SEC returned no recent filing rows"
            return docs, news_status, sec_status

        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")
        df = (
            df[df["form"].isin(["10-K", "10-Q", "8-K"])]
            .dropna(subset=["filingDate"])
            .sort_values("filingDate", ascending=False)
            .head(12)
        )

        if df.empty:
            sec_status["status"] = "no_target_forms"
            sec_status["message"] = "No 10-K / 10-Q / 8-K filings found"
            return docs, news_status, sec_status

        sec_status["status"] = "ok"
        sec_status["items_returned"] = len(df)

        for _, row in df.iterrows():
            text = normalize_text(f"{row['form']} filing. {row.get('primaryDocDescription', '')}")
            docs.append({
                "source": "SEC",
                "ticker": ticker,
                "date": str(row["filingDate"].date()),
                "title": f"{row['form']} Filing",
                "text": text,
                "sentiment_score": None,
                "url": "",
            })

    except Exception as e:
        sec_status["status"] = "exception"
        sec_status["message"] = str(e)

    return docs, news_status, sec_status

# =========================================================
# RAG
# =========================================================
def embed_documents(docs: List[Dict[str, Any]]):
    if not docs:
        return np.zeros((0, 384), dtype=np.float32)
    texts = [d["text"] for d in docs]
    emb = EMBEDDER.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)

def retrieve_rag_evidence_balanced(ticker: str, horizon: str, docs: List[Dict[str, Any]], doc_embeddings: np.ndarray, top_k=8):
    if not docs or len(doc_embeddings) == 0:
        return []

    query = f"{ticker} {horizon} earnings guidance revenue margins risks regulation competition demand outlook"
    q_emb = EMBEDDER.encode([query], normalize_embeddings=True)[0].astype(np.float32)

    scores = doc_embeddings @ q_emb

    enriched = []
    for i, d in enumerate(docs):
        item = d.copy()
        item["retrieval_score"] = float(scores[i])
        enriched.append(item)

    news_docs = sorted([d for d in enriched if d["source"] == "NEWS"], key=lambda x: -x["retrieval_score"])
    sec_docs = sorted([d for d in enriched if d["source"] == "SEC"], key=lambda x: -x["retrieval_score"])

    n_news = min(4, len(news_docs))
    n_sec = min(4, len(sec_docs))

    combined = news_docs[:n_news] + sec_docs[:n_sec]
    combined = sorted(combined, key=lambda x: -x["retrieval_score"])[:top_k]
    return combined

# =========================================================
# CONTEXT SCORES
# =========================================================
RISK_WORDS = [
    "risk", "uncertainty", "litigation", "slowdown", "weakness", "decline",
    "adverse", "volatility", "pressure", "challenge", "regulation",
    "margin pressure", "supply chain", "competition", "cost inflation"
]

POSITIVE_WORDS = [
    "growth", "strong", "beat", "improved", "raised", "expansion",
    "optimistic", "demand", "record", "momentum", "profitability"
]

def compute_context_scores(retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    news_docs = [d for d in retrieved_docs if d["source"] == "NEWS"]
    sec_docs = [d for d in retrieved_docs if d["source"] == "SEC"]

    news_sentiments = [safe_float(d.get("sentiment_score", 0.0), 0.0) for d in news_docs if d.get("sentiment_score") is not None]
    avg_news_sent = float(np.mean(news_sentiments)) if news_sentiments else 0.0

    sec_text = " ".join(d.get("text", "") for d in sec_docs).lower()
    risk_hits = sum(sec_text.count(w) for w in RISK_WORDS)
    pos_hits = sum(sec_text.count(w) for w in POSITIVE_WORDS)
    sec_score = float(np.clip((pos_hits - risk_hits) / 10.0, -1.0, 1.0))

    evidence_strength = float(np.mean([d["retrieval_score"] for d in retrieved_docs])) if retrieved_docs else 0.0
    evidence_strength = float(np.clip(evidence_strength, 0.0, 1.0))

    return {
        "avg_news_sentiment": avg_news_sent,
        "news_signal": float(np.clip(avg_news_sent, -0.35, 0.35)),
        "sec_context_score": sec_score,
        "evidence_strength": evidence_strength,
        "news_count": len(news_docs),
        "sec_count": len(sec_docs),
    }

# =========================================================
# FUSION
# =========================================================
def fuse_signals(quant_outputs: Dict[str, Any], context_scores: Dict[str, Any]) -> Dict[str, Any]:
    ml_score = squash_tanh(quant_outputs["ml_pred"], scale=1.5)
    dl_score = squash_tanh(quant_outputs["dl_pred"], scale=0.02)
    tr_score = squash_tanh(quant_outputs["transformer_pred"], scale=0.04)
    news_score = squash_tanh(context_scores["news_signal"], scale=0.75)
    sec_score = squash_tanh(context_scores["sec_context_score"], scale=0.75)
    evidence_score = squash_tanh(context_scores["evidence_strength"], scale=0.75)

    weights = {
        "Price pattern model": 0.22,
        "Deep learning model": 0.20,
        "Sequence intelligence model": 0.23,
        "Recent news context": 0.15,
        "Regulatory filing context": 0.12,
        "Evidence quality": 0.08,
    }

    final_score = (
        weights["Price pattern model"] * ml_score +
        weights["Deep learning model"] * dl_score +
        weights["Sequence intelligence model"] * tr_score +
        weights["Recent news context"] * news_score +
        weights["Regulatory filing context"] * sec_score +
        weights["Evidence quality"] * evidence_score
    )

    if final_score >= 0.35:
        label = "Strong Bullish Support"
    elif final_score >= 0.12:
        label = "Mild Bullish Support"
    elif final_score <= -0.35:
        label = "Strong Bearish Support"
    elif final_score <= -0.12:
        label = "Mild Bearish Support"
    else:
        label = "Neutral / Mixed"

    return {
        "final_score": float(final_score),
        "label": label,
        "component_scores": {
            "Price pattern model": float(ml_score),
            "Deep learning model": float(dl_score),
            "Sequence intelligence model": float(tr_score),
            "Recent news context": float(news_score),
            "Regulatory filing context": float(sec_score),
            "Evidence quality": float(evidence_score),
        },
        "weights": weights,
    }

# =========================================================
# EXPLAINABILITY
# =========================================================
def generate_front_explanation_business(ticker, horizon, fusion_outputs, context_scores):
    label = fusion_outputs["label"]
    score = fusion_outputs["final_score"]
    news_count = context_scores["news_count"]
    sec_count = context_scores["sec_count"]

    return f"""
    <div class="explain-summary">
    <b>Business interpretation:</b><br><br>
    QuantGenius evaluated <b>{ticker}</b> for the <b>{horizon}</b> decision horizon by combining
    historical price behaviour, machine learning forecasts, deep learning sequence analysis,
    and live external market evidence.<br><br>

    The final system decision is <b>{label}</b> with a score of <b>{score:.4f}</b>.<br><br>

    From a business perspective, this means the system sees the current market setup for
    <b>{ticker}</b> as leaning in a supportive direction based on both:
    <ul>
      <li>internal quantitative forecasting signals, and</li>
      <li>external evidence from <b>{news_count}</b> newly fetched news items and <b>{sec_count}</b> SEC-related evidence items.</li>
    </ul>

    This output is designed as a decision-support layer, not as a standalone trading instruction.
    It helps a business user or analyst understand whether recent market conditions and external disclosures
    are reinforcing or weakening the model-driven view.
    </div>
    """

def build_single_reason_card(fusion_outputs, context_scores):
    comp = fusion_outputs["component_scores"]

    def score_text(value):
        if value > 0.25:
            return "strongly supportive"
        elif value > 0.08:
            return "mildly supportive"
        elif value < -0.25:
            return "strongly negative"
        elif value < -0.08:
            return "mildly negative"
        return "neutral or mixed"

    bullets = [
        f"<li>The <b>price pattern model</b> was <b>{score_text(comp['Price pattern model'])}</b>, meaning historical market structure materially influenced the final result.</li>",
        f"<li>The <b>deep learning model</b> was <b>{score_text(comp['Deep learning model'])}</b>, showing how nonlinear pattern recognition contributed to the decision.</li>",
        f"<li>The <b>sequence intelligence model</b> was <b>{score_text(comp['Sequence intelligence model'])}</b>, which means time-based market behaviour was a relevant part of the signal.</li>",
        f"<li>The system included <b>{context_scores['news_count']}</b> newly fetched news items, and the <b>recent news context</b> was <b>{score_text(comp['Recent news context'])}</b>.</li>",
        f"<li>The system also included <b>{context_scores['sec_count']}</b> SEC-related evidence items, and the <b>regulatory filing context</b> was <b>{score_text(comp['Regulatory filing context'])}</b>.</li>",
        f"<li>The overall <b>evidence quality</b> score was <b>{comp['Evidence quality']:.3f}</b>, showing how strongly the retrieved evidence matched the decision context.</li>",
        f"<li>After combining all quantitative and contextual inputs, the final decision became <b>{fusion_outputs['label']}</b> with a score of <b>{fusion_outputs['final_score']:.4f}</b>.</li>",
    ]

    return f"""
    <div class="card-shell bullet-card">
        <ul>
            {''.join(bullets)}
        </ul>
    </div>
    """

def build_client_breakdown(component_scores):
    rows = []
    for name, score in component_scores.items():
        if score > 0.25:
            meaning = "This pushed the final result upward strongly."
        elif score > 0.08:
            meaning = "This supported the final result slightly."
        elif score < -0.25:
            meaning = "This pulled the final result downward strongly."
        elif score < -0.08:
            meaning = "This added some downside pressure."
        else:
            meaning = "This had little or mixed impact."

        rows.append({
            "Decision driver": name,
            "Influence score": round(score, 4),
            "What it means": meaning
        })
    return pd.DataFrame(rows)

def build_evidence_table_df(retrieved_docs):
    if not retrieved_docs:
        return pd.DataFrame(columns=["Source", "Date", "Title", "Relevance", "Sentiment", "URL"])

    rows = []
    for d in retrieved_docs:
        rows.append({
            "Source": d.get("source", ""),
            "Date": d.get("date", ""),
            "Title": d.get("title", ""),
            "Relevance": round(safe_float(d.get("retrieval_score", np.nan), np.nan), 4),
            "Sentiment": None if d.get("sentiment_score") is None else round(safe_float(d.get("sentiment_score", 0.0), 0.0), 4),
            "URL": d.get("url", "")
        })
    return pd.DataFrame(rows)

def render_dark_html_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data available.")
        return

    html = '<div class="dark-table-wrap"><table class="dark-table"><thead><tr>'
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                val = ""
            html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

def render_price_chart(hist: pd.DataFrame, ticker: str):
    price_view = hist.tail(60).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_view["Date"],
        y=price_view["Close"],
        mode="lines",
        name=ticker,
        line=dict(color="#5eead4", width=3)
    ))

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#e8f2ff"),
        xaxis=dict(title="", showgrid=False, zeroline=False),
        yaxis=dict(title="Price", showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="main-hero">
    <h1>QuantGenius AI</h1>
    <p>
        Quantitative Forecasting + Live News / SEC Retrieval + Explainable Fusion
        <br><br>
        This application combines mathematical forecasting models with live market evidence.
        It is built to help a user understand not just <b>what</b> the system predicts,
        but also <b>why</b> the system reached that conclusion.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# INPUTS
# =========================================================
left, right = st.columns([1.2, 1])
with left:
    ticker = st.selectbox("Select ticker", ASSET_UNIVERSE, index=0)
with right:
    horizon = st.selectbox("Select horizon", ["short_3d", "intermediate_10d", "intermediate_15d"], index=1)

run_button = st.button("Run End-to-End Decision Pipeline", type="primary")

# =========================================================
# MAIN
# =========================================================
if run_button:
    try:
        with st.spinner("Running QuantGenius pipeline..."):
            hist = fetch_market_data(ticker)
            feat_df, feature_cols = build_features(hist)

            quant_outputs = run_quant_models(ticker, feat_df, feature_cols)
            live_docs, news_status, sec_status = fetch_live_docs(ticker)
            doc_embeddings = embed_documents(live_docs)
            retrieved_docs = retrieve_rag_evidence_balanced(ticker, horizon, live_docs, doc_embeddings, top_k=8)
            context_scores = compute_context_scores(retrieved_docs)
            fusion_outputs = fuse_signals(quant_outputs, context_scores)

        st.markdown(generate_front_explanation_business(ticker, horizon, fusion_outputs, context_scores), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card-shell">
            <div class="small-note">Final Result</div>
            <div class="result-label {label_class(fusion_outputs['label'])}">{fusion_outputs['label']}</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Price pattern model</div>
                <div class="metric-value">{quant_outputs['ml_pred']:.4f}</div>
                <div class="metric-sub">Pattern-based numerical forecasting from historical market features.</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Deep learning model</div>
                <div class="metric-value">{quant_outputs['dl_pred']:.4f}</div>
                <div class="metric-sub">Nonlinear recognition across the recent price sequence window.</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Sequence intelligence model</div>
                <div class="metric-value">{quant_outputs['transformer_pred']:.4f}</div>
                <div class="metric-sub">Temporal relationship and sequence-structure understanding.</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Final decision score</div>
                <div class="metric-value">{fusion_outputs['final_score']:.4f}</div>
                <div class="metric-sub">Combined result after blending quantitative models with live evidence.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Why this system reached this result</div>', unsafe_allow_html=True)
        st.markdown(build_single_reason_card(fusion_outputs, context_scores), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Client-Friendly Fusion Summary</div>', unsafe_allow_html=True)
        render_dark_html_table(build_client_breakdown(fusion_outputs["component_scores"]))

        st.markdown('<div class="section-title">Recent Price Context</div>', unsafe_allow_html=True)
        render_price_chart(hist, ticker)

        st.markdown('<div class="section-title">Retrieved Evidence</div>', unsafe_allow_html=True)
        render_dark_html_table(build_evidence_table_df(retrieved_docs))

        with st.expander("Data source checks"):
            st.write("News pipeline")
            st.json(news_status)
            st.write("SEC pipeline")
            st.json(sec_status)

    except Exception as e:
        st.error(str(e))
        st.info("Please try again in a few minutes or choose another ticker.")

st.caption("For academic research and decision-support demonstration only. Not investment advice.")
'''
