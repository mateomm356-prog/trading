"""
MODELO 1 — Movimiento Browniano Geométrico (solución analítica, sin simular).

    dS = mu*S dt + sigma*S dW      ->    ln(S_t/S_0) ~ Normal( m*t , sigma^2 * t ),   m = mu - sigma^2/2

Con eso se calcula en forma cerrada:
  - bandas de precio por percentil para cada minuto del horizonte
  - en qué percentil cae el precio actual respecto a lo esperado desde hace `ancla` minutos  -> señal
  - probabilidad de tocar el take-profit antes que el stop-loss (fórmula de barreras)

sigma se calibra a la escala del ancla (ver common.estimar_parametros) y el ancla nunca cruza al día anterior.
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from common import (estimar_parametros, norm_cdf, norm_cdf_vec, norm_ppf, senal, explicar_senal,
                    minutos_en_sesion_actual, percentil_empirico, calibracion)

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
MIN_ANCLA = 5


def proyeccion(S0, m, sigma, horizonte, percentiles=PERCENTILES):
    """Bandas analíticas: S0 * exp(m*t + z_p * sigma * sqrt(t))."""
    t = np.arange(1, horizonte + 1, dtype=float)
    bandas = {p: S0 * np.exp(m * t + norm_ppf(p / 100) * sigma * np.sqrt(t)) for p in percentiles}
    media = S0 * np.exp((m + sigma ** 2 / 2) * t)
    return t, bandas, media


def percentil_actual(S_now, S_ancla, m, sigma, n):
    """Percentil (0-100) del precio actual dentro de la lognormal proyectada desde S_ancla hace n minutos."""
    z = (math.log(S_now / S_ancla) - m * n) / (sigma * math.sqrt(n))
    return 100 * norm_cdf(z), z


def prob_terminar_arriba(S0, X, m, sigma, h):
    """P( S_h > X )."""
    return 1 - norm_cdf((math.log(X / S0) - m * h) / (sigma * math.sqrt(h)))


def prob_tp_antes_sl(S0, tp, sl, m, sigma):
    """P(tocar tp antes que sl) para un browniano con drift, sin límite de tiempo (fórmula de escala)."""
    u, l = math.log(tp / S0), math.log(sl / S0)
    k = 2 * m / sigma ** 2
    if abs(k * u) < 1e-9 and abs(k * l) < 1e-9:
        return -l / (u - l)
    return (1 - math.exp(-k * l)) / (math.exp(-k * u) - math.exp(-k * l))


def analizar(closes, S_now, horizonte=60, ancla=60, umbral_compra=10, umbral_venta=90,
             sesiones=None, usar_drift=False):
    """
    closes   = cierres de 1 min (histórico), S_now = precio actual
    sesiones = etiqueta de sesión (fecha) por cada cierre; evita cruzar la noche al estimar y al anclar
    """
    closes = np.asarray(closes, dtype=float)
    p = estimar_parametros(closes, sesiones, escala=ancla, usar_drift=usar_drift)

    # ancla: nunca más atrás que la apertura de la sesión actual
    disp = minutos_en_sesion_actual(closes, sesiones)
    n = int(max(0, min(ancla, disp, len(closes) - 1)))
    nota = ""
    if n < MIN_ANCLA:
        pct, z, S_ancla, n = 50.0, 0.0, float(closes[-1]), max(n, 1)
        nota = "Primeros minutos de la sesión: aún no hay ventana para medir."
    else:
        S_ancla = float(closes[-1 - n])
        pct, z = percentil_actual(S_now, S_ancla, p["m"], p["sigma"], n)
        if n < ancla:
            nota = f"Sesión recién abierta: midiendo desde hace {n} min (no {ancla})."
    s = senal(pct, umbral_compra, umbral_venta)

    # calibración: percentil que este mismo modelo le habría dado a cada ventana real del histórico
    pct_hist = 100 * norm_cdf_vec((p["ventanas"] - p["m"] * p["escala"]) / (p["sigma"] * math.sqrt(p["escala"])))
    cal = calibracion(pct_hist, umbral_compra, umbral_venta)
    pct_emp = percentil_empirico(math.log(S_now / S_ancla), p["ventanas"]) if n == ancla else float("nan")

    t, bandas, media = proyeccion(S_now, p["m"], p["sigma"], horizonte)
    tp, sl = float(bandas[90][-1]), float(bandas[10][-1])

    return {
        "modelo": "GBM analítico",
        "precio": S_now,
        "precio_ancla": S_ancla,
        "ancla": n,
        "nota": nota,
        "horizonte": horizonte,
        "percentil": pct,
        "percentil_empirico": pct_emp,
        "calibracion": cal,
        "z": z,
        "senal": s,
        "explicacion": explicar_senal(s, pct, n),
        "params": p,
        "t": t,
        "bandas": bandas,
        "media": media,
        "niveles_horizonte": {f"P{k}": float(v[-1]) for k, v in bandas.items()},
        "take_profit": tp,
        "stop_loss": sl,
        "prob_subir": prob_terminar_arriba(S_now, S_now, p["m"], p["sigma"], horizonte),
        "prob_tp_antes_sl": prob_tp_antes_sl(S_now, tp, sl, p["m"], p["sigma"]),
    }


def figura(df, res, n_hist=120):
    """Gráfica plotly: histórico + abanico de bandas del horizonte."""
    hist = df.tail(n_hist)
    t0 = hist["t"].iloc[-1]
    fut = pd.date_range(t0, periods=res["horizonte"] + 1, freq="min")[1:]
    b = res["bandas"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["t"], y=hist["close"], name="Precio", line=dict(color="#1B2430", width=1.6)))
    for lo, hi, a in ((5, 95, 0.10), (10, 90, 0.16), (25, 75, 0.26)):
        fig.add_trace(go.Scatter(x=list(fut) + list(fut[::-1]), y=list(b[hi]) + list(b[lo][::-1]),
                                 fill="toself", fillcolor=f"rgba(58,110,165,{a})", line=dict(width=0),
                                 name=f"P{lo}–P{hi}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=b[50], name="Mediana", line=dict(color="#3A6EA5", dash="dash")))
    fig.add_hline(y=res["take_profit"], line=dict(color="#1F8A5B", dash="dot"), annotation_text="TP (P90)")
    fig.add_hline(y=res["stop_loss"], line=dict(color="#C8452B", dash="dot"), annotation_text="SL (P10)")
    fig.add_trace(go.Scatter(x=[t0], y=[res["precio"]], mode="markers", name="Ahora",
                             marker=dict(size=9, color="#1B2430")))
    fig.update_layout(title=f"Modelo 1 · GBM · señal {res['senal']} (percentil {res['percentil']:.0f})",
                      margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified", legend=dict(orientation="h"))
    return fig


def resumen(res):
    p, c = res["params"], res["calibracion"]
    lines = [
        f"== {res['modelo']} ==",
        f"Precio actual: {res['precio']:.2f}   (hace {res['ancla']} min: {res['precio_ancla']:.2f})",
        f"sigma/min bruta {p['sigma1']*100:.3f}% -> calibrada a {p['escala']} min {p['sigma']*100:.3f}% "
        f"(ratio {p['vr']:.2f}) | sigma/{p['escala']}min {p['sigma_h']*100:.2f}% | "
        f"drift/hora {p['m_est']*60*100:+.3f}% ({'usado' if p['usar_drift'] else 'ignorado'})",
        f"Percentil del precio actual: {res['percentil']:.1f}  (z = {res['z']:+.2f}) | empírico {res['percentil_empirico']:.0f}",
        f"Calibración {c['n']} ventanas: zona COMPRA {c['compra']:.0f}% · zona VENTA {c['venta']:.0f}%",
        f"SEÑAL: {res['senal']}  -> {res['explicacion']}" + (f"  [{res['nota']}]" if res["nota"] else ""),
        f"Niveles a {res['horizonte']} min: " + "  ".join(f"{k}={v:.2f}" for k, v in res["niveles_horizonte"].items()),
        f"Take-profit {res['take_profit']:.2f} | Stop-loss {res['stop_loss']:.2f} | "
        f"P(sube) {res['prob_subir']*100:.0f}% | P(TP antes que SL) {res['prob_tp_antes_sl']*100:.0f}%",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import config
    import data_alpaca as da

    df, px, ts = da.cargar(config.SYMBOL, config.ALPACA_KEY, config.ALPACA_SECRET,
                           dias=config.DIAS_HISTORIA, feed=config.FEED, solo_regular=config.SOLO_HORARIO_REGULAR)
    res = analizar(df["close"].values, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA, config.UMBRAL_VENTA,
                   sesiones=df["t"].dt.date.values, usar_drift=config.USAR_DRIFT)
    print(f"{config.SYMBOL} @ {ts}\n")
    print(resumen(res))
    fig = figura(df, res)
    out = f"gbm_{config.SYMBOL}.html"
    fig.write_html(out)
    print(f"\nGráfica guardada en {out}")
    fig.show()
