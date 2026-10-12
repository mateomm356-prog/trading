"""
App Streamlit — SMCI en vivo con Alpaca. Modelo 1 (GBM) y Modelo 2 (Montecarlo) en pestañas separadas.
Correr:  streamlit run app.py
"""
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

import config
import data_alpaca as da
import gbm_model as gbm
import montecarlo_model as mc

st.set_page_config(page_title="SMCI · GBM + Montecarlo", page_icon="📈", layout="wide")

COLOR = {"COMPRA": "#1F8A5B", "VENTA": "#C8452B", "NEUTRO": "#6B7480"}


def badge(s):
    return (f'<div style="display:inline-block;padding:10px 22px;border-radius:8px;background:{COLOR[s]};'
            f'color:#fff;font-size:28px;font-weight:700;letter-spacing:.02em">{s}</div>')


def _secret(nombre, default):
    """Streamlit Cloud (pestaña Secrets) o .streamlit/secrets.toml local; si no, lo que diga config.py."""
    try:
        return st.secrets.get(nombre, default) or default
    except Exception:
        return default


KEY_DEF = _secret("APCA_API_KEY_ID", config.ALPACA_KEY)
SECRET_DEF = _secret("APCA_API_SECRET_KEY", config.ALPACA_SECRET)
EN_CLOUD = "PEGA_TU" not in KEY_DEF

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Conexión")
    if EN_CLOUD:
        st.caption("Llaves de Alpaca cargadas desde Secrets.")
        key, secret = KEY_DEF, SECRET_DEF
    else:
        key = st.text_input("Alpaca key", value="", type="password")
        secret = st.text_input("Alpaca secret", value="", type="password")
    symbol = st.text_input("Símbolo", value=config.SYMBOL).upper().strip()
    feed = st.selectbox("Feed", ["iex", "sip"], index=0 if config.FEED == "iex" else 1,
                        help="iex = tiempo real gratis. sip = mercado completo, pero sin los últimos 15 min en plan gratis.")
    dias = st.slider("Sesiones de histórico (1 min)", 1, 20, config.DIAS_HISTORIA)
    solo_reg = st.checkbox("Solo horario regular (9:30–16:00 NY)", value=config.SOLO_HORARIO_REGULAR)

    st.header("Modelo")
    horizonte = st.slider("Horizonte (min)", 5, 390, config.HORIZONTE, step=5)
    ancla = st.slider("Ancla: medir desde hace (min)", 5, 390, config.ANCLA, step=5)
    umbral_c = st.slider("Umbral compra (percentil ≤)", 1, 40, config.UMBRAL_COMPRA)
    umbral_v = st.slider("Umbral venta (percentil ≥)", 60, 99, config.UMBRAL_VENTA)

    st.header("Montecarlo")
    n_sims = st.select_slider("Caminos", [1000, 5000, 10000, 20000, 50000], value=config.N_SIMULACIONES)
    modo = st.radio("Incrementos", ["bootstrap", "normal"], index=0 if config.MODO_MC == "bootstrap" else 1,
                    help="bootstrap = remuestrea retornos reales (colas gordas). normal = gaussiano.")

    st.header("Actualización")
    auto = st.toggle("Auto-refrescar", value=True)
    intervalo = st.slider("Cada (seg)", 5, 120, 15)
    if st.button("Refrescar ahora"):
        st.cache_data.clear()


@st.cache_data(ttl=55, show_spinner=False)
def cargar_barras(symbol, key, secret, dias, feed, solo_reg):
    return da.get_bars(symbol, key, secret, dias=dias, feed=feed, solo_regular=solo_reg)


def ultimo_precio(symbol, key, secret, feed, df):
    try:
        return da.get_ultimo_precio(symbol, key, secret, feed=feed)
    except Exception:
        return float(df["close"].iloc[-1]), df["t"].iloc[-1]


def tabla_niveles(res):
    niv = res["niveles_horizonte"]
    return pd.DataFrame({
        "Nivel": list(niv.keys()),
        "Precio": [f"{v:.2f}" for v in niv.values()],
        "vs actual": [f"{(v / res['precio'] - 1) * 100:+.2f}%" for v in niv.values()],
    })


def panel_modelo(res, fig, extra_metrics):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(res["senal"]), unsafe_allow_html=True)
        st.caption(res["explicacion"])
        m1, m2 = st.columns(2)
        m1.metric("Precio", f"{res['precio']:.2f}",
                  f"{(res['precio'] / res['precio_ancla'] - 1) * 100:+.2f}% vs hace {res['ancla']} min")
        m2.metric("Percentil", f"{res['percentil']:.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Take-profit", f"{res['take_profit']:.2f}", f"{(res['take_profit'] / res['precio'] - 1) * 100:+.2f}%")
        m4.metric("Stop-loss", f"{res['stop_loss']:.2f}", f"{(res['stop_loss'] / res['precio'] - 1) * 100:+.2f}%")
        for lbl, val in extra_metrics:
            st.write(f"**{lbl}:** {val}")
        p = res["params"]
        st.caption(f"σ/min {p['sigma'] * 100:.3f}% · σ/día {p['sigma_dia'] * 100:.2f}% · "
                   f"drift/hora {p['m'] * 60 * 100:+.3f}% · {p['n']} retornos")
        st.dataframe(tabla_niveles(res), hide_index=True, width="stretch")
    with c2:
        st.plotly_chart(fig, width="stretch")


@st.fragment(run_every=dt.timedelta(seconds=intervalo) if auto else None)
def cuerpo():
    try:
        df = cargar_barras(symbol, key, secret, dias, feed, solo_reg)
        px, ts = ultimo_precio(symbol, key, secret, feed, df)
    except Exception as e:
        st.error(f"No pude traer datos de Alpaca.\n\n{e}")
        st.info("Revisa: (1) llaves en Secrets o en la barra lateral, (2) feed = iex en plan gratis, (3) símbolo correcto.")
        return

    closes = df["close"].values
    st.caption(f"{symbol} · {len(df)} velas de 1 min ({df['t'].iloc[0]:%d %b} → {df['t'].iloc[-1]:%d %b %H:%M} NY) · "
               f"último trade {px:.2f} @ {ts:%H:%M:%S} NY · feed {feed} · actualizado {time.strftime('%H:%M:%S')}")

    t1, t2, t3 = st.tabs(["Modelo 1 · GBM analítico", "Modelo 2 · Montecarlo", "Exportar datos"])

    with t1:
        r1 = gbm.analizar(closes, px, horizonte, ancla, umbral_c, umbral_v)
        panel_modelo(r1, gbm.figura(df, r1), [
            ("P(precio sube en el horizonte)", f"{r1['prob_subir'] * 100:.0f}%"),
            ("P(toca TP antes que SL) — sin límite de tiempo", f"{r1['prob_tp_antes_sl'] * 100:.0f}%"),
        ])

    with t2:
        r2 = mc.analizar(closes, px, horizonte, ancla, umbral_c, umbral_v, n_sims=n_sims, modo=modo)
        bz = r2["barreras"]
        panel_modelo(r2, mc.figura(df, r2), [
            ("P(precio sube en el horizonte)", f"{r2['prob_subir'] * 100:.0f}%"),
            ("Dentro del horizonte toca", f"TP primero {bz['tp_primero'] * 100:.0f}% · SL primero {bz['sl_primero'] * 100:.0f}% · ninguno {bz['ninguno'] * 100:.0f}%"),
            ("Peor 5% / mejor 5% de escenarios", f"{r2['peor_5pct'] * 100:+.2f}% / {r2['mejor_5pct'] * 100:+.2f}%"),
        ])

    with t3:
        st.write("Copia esto y pégalo en la versión web (o descarga el CSV).")
        csv = df[["t", "close"]].to_csv(index=False)
        st.download_button("Descargar closes CSV", csv, file_name=f"{symbol}_1min.csv", mime="text/csv")
        st.code("\n".join(f"{c:.4f}" for c in closes[-800:]), language="text")


st.title(f"{symbol} · dónde comprar y vender")
st.caption("Los dos modelos dan probabilidades, no certezas. Esto no es asesoría financiera.")
cuerpo()
