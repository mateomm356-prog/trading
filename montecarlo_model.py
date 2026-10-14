"""
MODELO 2 — Simulación Montecarlo.

Genera miles de trayectorias de precio minuto a minuto y saca todo de la nube de caminos:
  - modo "bootstrap": BLOCK bootstrap — copia tramos contiguos de `bloque` minutos de los retornos REALES
    (dentro de una misma sesión). Al copiar tramos enteros conserva la autocorrelación de corto plazo
    (el rebote bid-ask, las rachas), que es justo lo que el remuestreo minuto a minuto destruye y lo que
    hacía que el modelo sobrestimara cuánto se mueve una hora.
  - modo "normal": incrementos gaussianos con la sigma calibrada (equivale a un GBM simulado; sirve para
    comparar contra el Modelo 1).
Lo que el GBM analítico no puede dar sin fórmulas complicadas y aquí sale gratis: probabilidad de tocar el
take-profit antes que el stop-loss DENTRO del horizonte (depende del camino, no solo del precio final).
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import (estimar_parametros, senal, explicar_senal, minutos_en_sesion_actual,
                    percentil_empirico, calibracion)

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
MIN_ANCLA = 5


# ---------------- generadores de incrementos ----------------
def _bloques_validos(rs, L):
    """Concatena los retornos por sesión y devuelve los índices de inicio de bloque que no cruzan sesión."""
    r1 = np.concatenate(rs)
    starts, off = [], 0
    for r in rs:
        if len(r) >= L:
            starts.append(np.arange(off, off + len(r) - L + 1))
        off += len(r)
    if not starts:                       # sesiones más cortas que el bloque: usar bloques de 1 min
        return r1, np.arange(len(r1)), 1
    return r1, np.concatenate(starts), L


def _incrementos(rng, p, shape, modo, bloque):
    n_sims, h = shape
    if modo != "bootstrap":
        return rng.normal(p["m"], p["sigma"], size=shape)
    r1, starts, L = _bloques_validos(p["retornos_sesion"], bloque)
    if not p["usar_drift"]:
        r1 = r1 - p["m_est"]             # quitar la tendencia de la muestra
    nb = int(math.ceil(h / L))
    st = rng.choice(starts, size=(n_sims, nb))
    idx = st[:, :, None] + np.arange(L)[None, None, :]
    return r1[idx].reshape(n_sims, nb * L)[:, :h]


def simular(S0, p, horizonte, n_sims=10000, modo="bootstrap", bloque=30, rng=None):
    """Matriz (n_sims x horizonte) de precios simulados."""
    rng = rng or np.random.default_rng()
    inc = _incrementos(rng, p, (n_sims, horizonte), modo, bloque)
    return S0 * np.exp(np.cumsum(inc, axis=1))


def bandas(paths, percentiles=PERCENTILES):
    return {q: np.percentile(paths, q, axis=0) for q in percentiles}


def distribucion_ancla(p, n, n_sims, modo, bloque, rng):
    """Distribución simulada del log-retorno de n minutos (ordenada)."""
    R = _incrementos(rng, p, (n_sims, n), modo, bloque).sum(axis=1)
    return np.sort(R)


def prob_barreras(paths, tp, sl):
    """Qué fracción de caminos toca TP antes que SL, SL antes que TP, o ninguno dentro del horizonte."""
    hit_tp, hit_sl = paths >= tp, paths <= sl
    t_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), np.inf)
    t_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), np.inf)
    return {
        "tp_primero": float(np.mean(t_tp < t_sl)),
        "sl_primero": float(np.mean(t_sl < t_tp)),
        "ninguno": float(np.mean(np.isinf(t_tp) & np.isinf(t_sl))),
    }


def analizar(closes, S_now, horizonte=60, ancla=60, umbral_compra=10, umbral_venta=90,
             n_sims=10000, modo="bootstrap", bloque=30, seed=None, tp=None, sl=None,
             sesiones=None, usar_drift=False):
    closes = np.asarray(closes, dtype=float)
    p = estimar_parametros(closes, sesiones, escala=ancla, usar_drift=usar_drift)
    rng = np.random.default_rng(seed)

    disp = minutos_en_sesion_actual(closes, sesiones)
    n = int(max(0, min(ancla, disp, len(closes) - 1)))
    nota = ""
    R = distribucion_ancla(p, max(n, MIN_ANCLA), n_sims, modo, bloque, rng)
    if n < MIN_ANCLA:
        pct, S_ancla, n = 50.0, float(closes[-1]), max(n, 1)
        nota = "Primeros minutos de la sesión: aún no hay ventana para medir."
    else:
        S_ancla = float(closes[-1 - n])
        pct = 100 * float(np.searchsorted(R, math.log(S_now / S_ancla)) / len(R))
        if n < ancla:
            nota = f"Sesión recién abierta: midiendo desde hace {n} min (no {ancla})."
    s = senal(pct, umbral_compra, umbral_venta)

    # calibración: percentil que este modelo le habría dado a cada ventana real del histórico
    if n == ancla:
        pct_hist = 100 * np.searchsorted(R, p["ventanas"]) / len(R)
        pct_emp = percentil_empirico(math.log(S_now / S_ancla), p["ventanas"])
    else:
        R_full = distribucion_ancla(p, ancla, n_sims, modo, bloque, rng)
        pct_hist = 100 * np.searchsorted(R_full, p["ventanas"]) / len(R_full)
        pct_emp = float("nan")
    cal = calibracion(pct_hist, umbral_compra, umbral_venta)

    paths = simular(S_now, p, horizonte, n_sims, modo, bloque, rng)
    b = bandas(paths)
    final = paths[:, -1]
    tp = float(tp) if tp else float(b[90][-1])
    sl = float(sl) if sl else float(b[10][-1])

    return {
        "modelo": f"Montecarlo ({modo}{f', bloques de {bloque} min' if modo == 'bootstrap' else ''}, {n_sims:,} caminos)",
        "precio": S_now,
        "precio_ancla": S_ancla,
        "ancla": n,
        "nota": nota,
        "horizonte": horizonte,
        "percentil": pct,
        "percentil_empirico": pct_emp,
        "calibracion": cal,
        "senal": s,
        "explicacion": explicar_senal(s, pct, n),
        "params": p,
        "t": np.arange(1, horizonte + 1),
        "bandas": b,
        "paths_muestra": paths[:40],
        "niveles_horizonte": {f"P{k}": float(v[-1]) for k, v in b.items()},
        "media_final": float(final.mean()),
        "take_profit": tp,
        "stop_loss": sl,
        "prob_subir": float((final > S_now).mean()),
        "barreras": prob_barreras(paths, tp, sl),
        "peor_5pct": float(np.percentile(final, 5) / S_now - 1),
        "mejor_5pct": float(np.percentile(final, 95) / S_now - 1),
    }


def figura(df, res, n_hist=120):
    hist = df.tail(n_hist)
    t0 = hist["t"].iloc[-1]
    fut = pd.date_range(t0, periods=res["horizonte"] + 1, freq="min")[1:]
    b = res["bandas"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["close"], name="Precio", line=dict(color="#1B2430", width=1.6)))
    for path in res["paths_muestra"]:
        fig.add_trace(go.Scatter(x=fut, y=path, line=dict(color="rgba(120,130,145,0.18)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    for lo, hi, a in ((5, 95, 0.10), (10, 90, 0.16), (25, 75, 0.26)):
        fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=list(b[hi]) + list(b[lo][::-1]),
                                 fill="toself", fillcolor=f"rgba(140,80,170,{a})", line=dict(width=0),
                                 name=f"P{lo}–P{hi}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=b[50], name="Mediana", line=dict(color="#8C50AA", dash="dash")))
    fig.add_hline(y=res["take_profit"], line=dict(color="#1F8A5B", dash="dot"), annotation_text="TP")
    fig.add_hline(y=res["stop_loss"], line=dict(color="#C8452B", dash="dot"), annotation_text="SL")
    fig.add_trace(go.Scatter(x=[t0], y=[res["precio"]], mode="markers", name="Ahora",
                             marker=dict(size=9, color="#1B2430")))
    fig.update_layout(title=f"Modelo 2 · Montecarlo · señal {res['senal']} (percentil {res['percentil']:.0f})",
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified", legend=dict(orientation="h"))
    return fig


def resumen(res):
    p, bz, c = res["params"], res["barreras"], res["calibracion"]
    return "\n".join([
        f"== {res['modelo']} ==",
        f"Precio actual: {res['precio']:.2f}   (hace {res['ancla']} min: {res['precio_ancla']:.2f})",
        f"sigma/min bruta {p['sigma1']*100:.3f}% -> calibrada a {p['escala']} min {p['sigma']*100:.3f}% (ratio {p['vr']:.2f})",
        f"Percentil del precio actual: {res['percentil']:.1f} | empírico {res['percentil_empirico']:.0f}",
        f"Calibración {c['n']} ventanas: zona COMPRA {c['compra']:.0f}% · zona VENTA {c['venta']:.0f}%",
        f"SEÑAL: {res['senal']}  -> {res['explicacion']}" + (f"  [{res['nota']}]" if res["nota"] else ""),
        f"Niveles a {res['horizonte']} min: " + "  ".join(f"{k}={v:.2f}" for k, v in res["niveles_horizonte"].items()),
        f"Take-profit {res['take_profit']:.2f} | Stop-loss {res['stop_loss']:.2f}",
        f"P(sube) {res['prob_subir']*100:.0f}% | toca TP primero {bz['tp_primero']*100:.0f}% | "
        f"toca SL primero {bz['sl_primero']*100:.0f}% | ninguno {bz['ninguno']*100:.0f}%",
        f"Peor 5% de escenarios: {res['peor_5pct']*100:+.2f}% | Mejor 5%: {res['mejor_5pct']*100:+.2f}%",
    ])


if __name__ == "__main__":
    import config
    import data_alpaca as da

    df, px, ts = da.cargar(config.SYMBOL, config.ALPACA_KEY, config.ALPACA_SECRET,
                           dias=config.DIAS_HISTORIA, feed=config.FEED, solo_regular=config.SOLO_HORARIO_REGULAR)
    res = analizar(df["close"].values, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA,
                   config.UMBRAL_VENTA, n_sims=config.N_SIMULACIONES, modo=config.MODO_MC, bloque=config.BLOQUE_MC,
                   sesiones=df["t"].dt.date.values, usar_drift=config.USAR_DRIFT)
    print(f"{config.SYMBOL} @ {ts}\n")
    print(resumen(res))
    fig = figura(df, res)
    out = f"montecarlo_{config.SYMBOL}.html"
    fig.write_html(out)
    print(f"\nGráfica guardada en {out}")
    fig.show()
