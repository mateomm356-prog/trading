"""
App Streamlit — SMCI en vivo con Alpaca.
  · Decisión de ahora (con "ventana segura": no compra mientras sigue cayendo)
  · Modelo 1 (GBM) y Modelo 2 (Montecarlo)
  · Simulador: qué habría pasado operando con la estrategia en los últimos días (precios reales)
  · Mi cuenta de papel: comprar/vender 1 o 2 acciones con dinero ficticio y ver cuánto ganas o pierdes en vivo
Correr:  streamlit run app.py
"""
import time
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data_alpaca as da
import gbm_model as gbm
import montecarlo_model as mc
import estrategia as es
import broker

st.set_page_config(page_title="SMCI · GBM + Montecarlo", page_icon="📈", layout="wide")

COLOR = {**es.COLORES, "MANTENER": "#2F5F9E", "VENDER": "#C8452B", "SIN POSICIÓN": "#6B7480"}


def badge(s, size=28):
    return (f'<div style="display:inline-block;padding:10px 22px;border-radius:8px;background:{COLOR.get(s, "#6B7480")};'
            f'color:#fff;font-size:{size}px;font-weight:700;letter-spacing:.02em">{s}</div>')


def _secret(nombre, default):
    """Streamlit Cloud (pestaña Secrets) o .streamlit/secrets.toml local; si no, lo que diga config.py."""
    try:
        return st.secrets.get(nombre, default) or default
    except Exception:
        return default


def cfg(nombre, default):
    return getattr(config, nombre, default)


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
    dias = st.slider("Sesiones de histórico (1 min)", 2, 20, config.DIAS_HISTORIA)
    solo_reg = st.checkbox("Solo horario regular (9:30–16:00 NY)", value=config.SOLO_HORARIO_REGULAR)

    st.header("Modelo")
    horizonte = st.slider("Horizonte (min)", 5, 390, config.HORIZONTE, step=5,
                          help="Cuánto tiempo hacia adelante se proyecta y cuánto dura como máximo una operación.")
    ancla = st.slider("Ancla: medir desde hace (min)", 5, 390, config.ANCLA, step=5,
                      help="Ventana con la que se mide si la última caída o subida fue rara (el percentil).")
    umbral_c = st.slider("Umbral compra (percentil ≤)", 1, 40, config.UMBRAL_COMPRA)
    umbral_v = st.slider("Umbral venta (percentil ≥)", 60, 99, config.UMBRAL_VENTA)
    usar_drift = st.toggle("Usar drift (tendencia de los últimos días)", value=cfg("USAR_DRIFT", False),
                           help="Apagado = tendencia 0. En pocos días el drift es ruido y sesga la señal.")

    st.header("Ventana segura")
    usar_conf = st.toggle("Esperar rebote antes de comprar", value=cfg("USAR_CONFIRMACION", True),
                          help="No compra mientras el precio sigue cayendo: espera a que la pendiente de los últimos minutos sea positiva.")
    corto = st.slider("Pendiente corta (min)", 3, 30, 10,
                      help="Ventana para decidir si el precio ya dejó de caer.")
    armado = st.slider("Oportunidad válida por (min)", 5, 90, cfg("ARMADO_MIN", 30), step=5,
                       help="Después de tocar la zona de compra, cuánto tiempo se espera el rebote antes de descartarla.")
    usar_tend = st.toggle("Bloquear compra si la tendencia de 2 h es muy bajista", value=cfg("USAR_TENDENCIA", False))
    usar_sal = st.toggle("Salida temprana si sigue cayendo", value=cfg("USAR_SALIDA_TEMPRANA", True),
                         help="Si compraste, estás en pérdida y el precio sigue cayendo fuerte, la señal pasa a VENDER sin esperar el stop-loss.")

    st.header("Montecarlo")
    n_sims = st.select_slider("Caminos", [1000, 5000, 10000, 20000], value=config.N_SIMULACIONES)
    modo = st.radio("Incrementos", ["bootstrap", "normal"], index=0 if config.MODO_MC == "bootstrap" else 1,
                    help="bootstrap = copia tramos de retornos reales. normal = gaussiano.")
    bloque = st.slider("Bloque bootstrap (min)", 5, 60, cfg("BLOQUE_MC", 30), step=5)

    st.header("Actualización")
    auto = st.toggle("Auto-refrescar", value=True)
    intervalo = st.slider("Cada (seg)", 5, 120, 15)
    if st.button("Refrescar ahora"):
        st.cache_data.clear()

PRM = dict(ancla=ancla, umbral_compra=umbral_c, umbral_venta=umbral_v, horizonte=horizonte, corto=corto,
           armado_min=armado, usar_confirmacion=usar_conf, usar_tendencia=usar_tend, usar_salida_temprana=usar_sal,
           z_salida=cfg("Z_SALIDA", -1.5))
RUN = dt.timedelta(seconds=intervalo) if auto else None


# ---------------- datos ----------------
@st.cache_data(ttl=55, show_spinner=False)
def cargar_barras(symbol, key, secret, dias, feed, solo_reg):
    return da.get_bars(symbol, key, secret, dias=dias, feed=feed, solo_regular=solo_reg)


@st.cache_data(ttl=5, show_spinner=False)
def precio_vivo(symbol, key, secret, feed):
    return da.get_ultimo_precio(symbol, key, secret, feed=feed)


def datos():
    df = cargar_barras(symbol, key, secret, dias, feed, solo_reg)
    try:
        px, ts = precio_vivo(symbol, key, secret, feed)
    except Exception:
        px, ts = float(df["close"].iloc[-1]), df["t"].iloc[-1]
    return df, px, ts


def gbm_ahora(df, px):
    return gbm.analizar(df["close"].values, px, horizonte, ancla, umbral_c, umbral_v,
                        sesiones=df["t"].dt.date.values, usar_drift=usar_drift)


def error_datos(e):
    st.error(f"No pude traer datos de Alpaca.\n\n{e}")
    st.info("Revisa: (1) llaves en Secrets o en la barra lateral, (2) feed = iex en plan gratis, (3) símbolo correcto.")


def fmt_z(z):
    return "—" if z is None else f"{z:+.1f}"


# ---------------- decisión de ahora ----------------
@st.fragment(run_every=RUN)
def decision():
    try:
        df, px, ts = datos()
    except Exception as e:
        error_datos(e); return
    r1 = gbm_ahora(df, px)
    ah = es.ahora(df["close"].values, df["t"].dt.date.values, px, r1["params"]["sigma"], PRM)
    st.session_state["ahora"] = ah

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(ah["estado"], 30), unsafe_allow_html=True)
        st.write(ah["explicacion"])
    with c2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio", f"{px:.2f}", f"{(px / r1['precio_ancla'] - 1) * 100:+.2f}% en {r1['ancla']} min")
        m2.metric("Percentil", "—" if ah["pct"] is None else f"{ah['pct']:.0f}",
                  help="Qué tan rara fue la movida de la última ventana. ≤10 barato, ≥90 caro.")
        m3.metric(f"Pendiente {corto} min", fmt_z(ah["z_corto"]),
                  help="Hacia dónde va el precio en los últimos minutos, en desviaciones normales. Positiva = subiendo.")
        m4.metric("Tendencia 2 h", fmt_z(ah["z_largo"]),
                  help="Lo mismo pero en 2 horas. Menor a -1.5 = tendencia bajista fuerte.")
    st.caption(f"{symbol} · {len(df)} velas de 1 min · último trade {px:.2f} @ {ts:%H:%M:%S} NY · feed {feed} · "
               f"actualizado {time.strftime('%H:%M:%S')}")


# ---------------- modelos ----------------
def tabla_niveles(res):
    niv = res["niveles_horizonte"]
    return pd.DataFrame({"Nivel": list(niv.keys()), "Precio": [f"{v:.2f}" for v in niv.values()],
                         "vs actual": [f"{(v / res['precio'] - 1) * 100:+.2f}%" for v in niv.values()]})


def panel_modelo(res, fig, extra_metrics):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(badge(res["senal"], 22), unsafe_allow_html=True)
        st.caption(res["explicacion"] + " (Señal pura del modelo, sin ventana segura.)")
        m1, m2 = st.columns(2)
        m1.metric("Take-profit", f"{res['take_profit']:.2f}", f"{(res['take_profit'] / res['precio'] - 1) * 100:+.2f}%")
        m2.metric("Stop-loss", f"{res['stop_loss']:.2f}", f"{(res['stop_loss'] / res['precio'] - 1) * 100:+.2f}%")
        if res["nota"]:
            st.info(res["nota"])
        for lbl, val in extra_metrics:
            st.write(f"**{lbl}:** {val}")
        p, c = res["params"], res["calibracion"]
        st.write(f"**Calibración ({c['n']} ventanas de {p['escala']} min):** zona COMPRA {c['compra']:.0f}% · "
                 f"zona VENTA {c['venta']:.0f}% de las veces")
        st.caption(f"σ/min bruta {p['sigma1'] * 100:.3f}% → calibrada {p['sigma'] * 100:.3f}% (ratio {p['vr']:.2f}) · "
                   f"σ/{p['escala']}min {p['sigma_h'] * 100:.2f}% · σ/día {p['sigma_dia'] * 100:.2f}%")
        st.dataframe(tabla_niveles(res), hide_index=True, width="stretch")
    with c2:
        st.plotly_chart(fig, width="stretch")


@st.fragment(run_every=RUN)
def tab_gbm():
    try:
        df, px, _ = datos()
    except Exception as e:
        error_datos(e); return
    r1 = gbm_ahora(df, px)
    panel_modelo(r1, gbm.figura(df, r1), [
        ("P(precio sube en el horizonte)", f"{r1['prob_subir'] * 100:.0f}%"),
        ("P(toca TP antes que SL)", f"{r1['prob_tp_antes_sl'] * 100:.0f}%"),
    ])


@st.fragment(run_every=RUN)
def tab_mc():
    try:
        df, px, _ = datos()
    except Exception as e:
        error_datos(e); return
    r2 = mc.analizar(df["close"].values, px, horizonte, ancla, umbral_c, umbral_v, n_sims=n_sims, modo=modo,
                     bloque=bloque, sesiones=df["t"].dt.date.values, usar_drift=usar_drift)
    bz = r2["barreras"]
    panel_modelo(r2, mc.figura(df, r2), [
        ("P(precio sube en el horizonte)", f"{r2['prob_subir'] * 100:.0f}%"),
        ("Dentro del horizonte toca", f"TP primero {bz['tp_primero'] * 100:.0f}% · SL primero {bz['sl_primero'] * 100:.0f}% · "
                                      f"ninguno {bz['ninguno'] * 100:.0f}%"),
        ("Peor 5% / mejor 5%", f"{r2['peor_5pct'] * 100:+.2f}% / {r2['mejor_5pct'] * 100:+.2f}%"),
    ])


# ---------------- simulador (backtest) ----------------
@st.cache_data(ttl=600, show_spinner=False)
def barras_largas(symbol, key, secret, n_dias, feed):
    return da.get_bars(symbol, key, secret, dias=n_dias, feed=feed, solo_regular=True)


def tab_simulador():
    st.write("Qué habría pasado si hubieras seguido la estrategia en los últimos días, con los **precios reales** de "
             f"{symbol}. Compra cuando la señal dice COMPRA, vende por take-profit, stop-loss, salida temprana, "
             "horizonte o cierre del día. Una operación a la vez, nada queda abierto de un día para otro.")
    c1, c2, c3 = st.columns(3)
    n_acc = c1.number_input("Acciones por operación", 1, 1000, 2, key="sim_acc")
    n_dias = c2.slider("Días a simular", 5, 60, 20, key="sim_dias")
    costo = c3.number_input("Costo por lado (%)", 0.0, 1.0, 0.05, 0.01, key="sim_costo",
                            help="Spread + deslizamiento: lo que pierdes al comprar y al vender por no conseguir el precio exacto.")
    if st.button("Correr simulación", type="primary", key="sim_run"):
        with st.spinner("Bajando velas y simulando…"):
            try:
                df = barras_largas(symbol, key, secret, n_dias + 1, feed)
            except Exception as e:
                error_datos(e); return
            base = {k: v for k, v in PRM.items() if not k.startswith("usar_")}
            res = {}
            for nombre, flags in es.VARIANTES.items():
                ops = es.backtest(df, {**base, **flags}, acciones=n_acc, costo_pct=costo)
                res[nombre] = (ops, es.resumen(ops))
            ops = es.backtest(df, PRM, acciones=n_acc, costo_pct=costo)
            res["Tu configuración actual"] = (ops, es.resumen(ops))
            st.session_state["sim"] = dict(res=res, dias=df["t"].dt.date.nunique() - 1, acc=n_acc,
                                           desde=df["t"].iloc[0], hasta=df["t"].iloc[-1])
    sim = st.session_state.get("sim")
    if not sim:
        st.info("Dale a **Correr simulación**.")
        return

    st.caption(f"{sim['dias']} días operables ({sim['desde']:%d %b} → {sim['hasta']:%d %b}) · "
               f"{sim['acc']} acciones por operación · el primer día solo se usa para calibrar.")
    filas = []
    for nombre, (ops, r) in sim["res"].items():
        filas.append({"Estrategia": nombre, "Operaciones": r["operaciones"],
                      "Ganadoras": "—" if r["operaciones"] == 0 else f"{r['ganadoras_pct']:.0f}%",
                      "Promedio por operación": "—" if r["operaciones"] == 0 else f"{r['ret_prom_pct']:+.2f}%",
                      "Ganancia total": f"${r['ganancia_usd']:+.2f}",
                      "Peor caída acumulada": f"${r['peor_racha_usd']:.2f}"})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    elegida = st.selectbox("Ver detalle de", list(sim["res"].keys()), index=len(sim["res"]) - 1, key="sim_ver")
    ops = sim["res"][elegida][0]
    if not ops:
        st.warning("Esta estrategia no hizo ninguna operación en ese periodo.")
        return
    d = pd.DataFrame(ops)
    fig = go.Figure()
    for nombre, (o, _) in sim["res"].items():
        if o:
            fig.add_trace(go.Scatter(y=np.cumsum([x["ganancia_usd"] for x in o]), mode="lines", name=nombre,
                                     line=dict(width=3 if nombre == elegida else 1.3)))
    fig.add_hline(y=0, line=dict(color="#999", dash="dot"))
    fig.update_layout(title="Ganancia acumulada por operación ($)", xaxis_title="Operación #", yaxis_title="$",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    mot = d.groupby("motivo")["ret_pct"].agg(["count", "mean"]).reset_index()
    mot.columns = ["Salió por", "Veces", "Promedio %"]
    mot["Promedio %"] = mot["Promedio %"].map(lambda x: f"{x:+.2f}%")
    st.dataframe(mot, hide_index=True)
    st.dataframe(d.rename(columns={"dia": "Día", "entrada_hora": "Compra", "salida_hora": "Venta", "entrada": "Precio compra",
                                   "salida": "Precio venta", "motivo": "Motivo", "minutos": "Min",
                                   "ret_pct": "% neto", "ganancia_usd": "$ neto"}),
                 hide_index=True, width="stretch")
    st.caption("Con pocos días los resultados son ruido: con menos de ~30 operaciones no se puede concluir nada. "
               "Rendimiento pasado no garantiza el futuro.")


# ---------------- cuenta de papel ----------------
@st.fragment(run_every=RUN)
def tab_papel():
    st.write("Compra y vende con **dinero ficticio** en tu cuenta de papel de Alpaca, a precios reales. "
             "Las órdenes con el mercado cerrado se ejecutan en la apertura.")
    try:
        cta = broker.cuenta(key, secret)
        clk = broker.reloj(key, secret)
        pos = broker.posicion(symbol, key, secret)
        ords = broker.ordenes(symbol, key, secret)
        df, px, _ = datos()
    except Exception as e:
        st.error(f"No pude leer tu cuenta de papel: {e}")
        st.info("Las llaves deben ser de Paper Trading (empiezan con PK).")
        return

    abierto = clk.get("is_open", False)
    st.caption(f"Mercado {'ABIERTO' if abierto else 'CERRADO'} · cuenta de papel: valor total ${float(cta['equity']):,.2f} · "
               f"efectivo ${float(cta['cash']):,.2f}")

    c1, c2, c3 = st.columns([1, 1, 2])
    qty = c1.number_input("Acciones", 1, 10000, 1, key="pap_qty")
    if c2.button(f"Comprar {qty} {symbol}", type="primary", key="pap_buy"):
        try:
            broker.comprar(symbol, qty, key, secret)
            st.success("Orden de compra enviada." + ("" if abierto else " Se ejecuta en la apertura."))
        except Exception as e:
            st.error(str(e))
    if c3.button(f"Vender todo {symbol}", key="pap_sell", disabled=pos is None):
        try:
            broker.vender_todo(symbol, key, secret)
            st.success("Orden de venta enviada." + ("" if abierto else " Se ejecuta en la apertura."))
        except Exception as e:
            st.error(str(e))

    pend = [o for o in ords if o.get("status") in ("new", "accepted", "pending_new", "partially_filled")]
    if pend:
        st.info(f"{len(pend)} orden(es) pendiente(s): " + ", ".join(f"{o['side']} {o['qty']}" for o in pend))

    st.subheader("Posición abierta")
    if pos is None:
        st.markdown(badge("SIN POSICIÓN", 20), unsafe_allow_html=True)
    else:
        q, entrada = float(pos["qty"]), float(pos["avg_entry_price"])
        pl = (px - entrada) * q
        compras = [o for o in ords if o["side"] == "buy" and o.get("status") == "filled"]
        t_ent = pd.to_datetime(compras[0]["filled_at"], utc=True) if compras else None
        mins = int((pd.Timestamp.now(tz="UTC") - t_ent).total_seconds() // 60) if t_ent is not None else 0
        ah = st.session_state.get("ahora") or {}
        sigma = ah.get("sigma") or gbm_ahora(df, px)["params"]["sigma"]
        hora_ny = pd.Timestamp.now(tz="America/New_York")
        motivo = es.salida(px, entrada, mins, ah.get("z_corto"), sigma, hora_ny.hour * 60 + hora_ny.minute, PRM)
        tp, sl = es.niveles(entrada, sigma, PRM)
        b1, b2 = st.columns([1, 2])
        with b1:
            st.markdown(badge("VENDER" if motivo else "MANTENER", 24), unsafe_allow_html=True)
            st.write(f"Motivo: **{motivo}**" if motivo else "Ninguna regla de salida se ha activado.")
        with b2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Acciones", f"{q:g}", f"compradas a {entrada:.2f}")
            m2.metric("Ganancia ahora", f"${pl:+.2f}", f"{(px / entrada - 1) * 100:+.2f}%")
            m3.metric("Take-profit", f"{tp:.2f}")
            m4.metric("Stop-loss", f"{sl:.2f}")
            st.caption(f"Lleva {mins} min abierta (horizonte {horizonte} min).")

    st.subheader("Operaciones cerradas")
    cerr = broker.operaciones_cerradas(ords)
    if not cerr:
        st.caption("Todavía no hay compras y ventas completas.")
    else:
        d = pd.DataFrame(cerr)
        tot = d["ganancia_usd"].sum()
        win = (d["ganancia_usd"] > 0).mean() * 100
        st.write(f"**{len(d)} operaciones · ganadoras {win:.0f}% · total ${tot:+.2f}**")
        st.dataframe(d.rename(columns={"compra": "Compra (NY)", "venta": "Venta (NY)", "acciones": "Acciones",
                                       "precio_compra": "Precio compra", "precio_venta": "Precio venta",
                                       "ganancia_usd": "$", "ret_pct": "%"}), hide_index=True, width="stretch")


# ---------------- página ----------------
st.title(f"{symbol} · dónde comprar y vender")
st.caption("Probabilidades, no certezas. Esto no es asesoría financiera.")
decision()

t1, t2, t3, t4, t5 = st.tabs(["Modelo 1 · GBM", "Modelo 2 · Montecarlo", "Simulador", "Mi cuenta de papel", "Exportar datos"])
with t1:
    tab_gbm()
with t2:
    tab_mc()
with t3:
    tab_simulador()
with t4:
    tab_papel()
with t5:
    try:
        df_x, _, _ = datos()
        st.download_button("Descargar velas CSV", df_x.to_csv(index=False), file_name=f"{symbol}_1min.csv", mime="text/csv")
    except Exception as e:
        error_datos(e)
