"""
ESTRATEGIA — convierte el percentil del modelo en decisiones, con "ventana segura".

Problema que resuelve: el percentil bajo dice "cayó más de lo normal", pero no dice si YA dejó de caer.
Comprar en ese momento es "agarrar un cuchillo cayendo". Por eso la regla ahora tiene 3 piezas:

  1. ARMADO      — el percentil tocó la zona de compra (≤ umbral). La oportunidad queda "armada" N minutos.
  2. CONFIRMACIÓN — solo se compra cuando la pendiente corta (últimos 10 min) ya es positiva: el precio empezó a
                    rebotar. Mientras siga cayendo el estado es "ESPERA · sigue cayendo".
  3. FILTRO DE TENDENCIA (opcional) — si la tendencia larga (últimos 120 min) es muy bajista, no se compra.

Y para salir de una posición, además de take-profit / stop-loss / horizonte:
  4. SALIDA TEMPRANA — si después de comprar el precio está por debajo de la entrada y la pendiente corta
                       sigue fuertemente negativa, el estado de la posición es VENDER.

"Pendiente" aquí = cambio del precio en la ventana dividido por lo que normalmente se mueve en ese tiempo
(z = log(P_ahora / P_hace_h) / (σ·√h)). z = +1 significa "subió una desviación normal"; z = -2, "cayó el doble de lo normal".
"""
import math
import numpy as np

from common import ids_sesion, norm_cdf_vec, estimar_parametros

# ---------------- parámetros por defecto ----------------
DEFAULT = dict(
    ancla=60,            # min: ventana del percentil
    umbral_compra=10,
    umbral_venta=90,
    corto=10,            # min: ventana de la pendiente corta (confirmación de rebote)
    largo=120,           # min: ventana de la tendencia larga
    armado_min=30,       # min que la oportunidad queda armada después de tocar la zona
    margen=10,           # puntos de percentil extra permitidos al confirmar (10 -> se puede entrar hasta percentil 20)
    z_confirma=0.0,      # pendiente corta mínima para confirmar rebote
    usar_confirmacion=True,
    usar_tendencia=False,
    z_tendencia=-1.5,    # tendencia larga por debajo de esto bloquea la compra
    usar_salida_temprana=True,
    z_salida=-1.5,       # pendiente corta por debajo de esto (estando en pérdida) = vender ya
    horizonte=60,        # min máximos en una posición
    k_banda=1.2816,      # TP/SL a ±k·σ·√horizonte  (1.2816 = P90/P10)
)

COLORES = {
    "COMPRA": "#1F8A5B",
    "VENTA": "#C8452B",
    "NEUTRO": "#6B7480",
    "ESPERA · sigue cayendo": "#B7791F",
    "ESPERA · sigue subiendo": "#B7791F",
    "BLOQUEADA · tendencia bajista": "#4A5568",
}

EXPLICA = {
    "COMPRA": "Cayó más de lo normal y ya empezó a rebotar en los últimos minutos. Zona de compra confirmada.",
    "VENTA": "Subió más de lo normal y ya empezó a devolverse. Si tienes acciones, es zona de venta.",
    "NEUTRO": "Movimiento dentro de lo normal. Sin ventaja, mejor esperar.",
    "ESPERA · sigue cayendo": "Está barato frente a lo normal, pero todavía va cayendo. No compres aún: espera a que rebote.",
    "ESPERA · sigue subiendo": "Está caro frente a lo normal, pero todavía va subiendo. Si tienes acciones, déjalas correr un poco.",
    "BLOQUEADA · tendencia bajista": "Está barato, pero la tendencia de las últimas 2 horas es fuertemente bajista. No se compra contra la tendencia.",
}


def _z_ventana(lp, ids, h, sigma):
    """z de la ventana h min para cada punto, sin cruzar sesión (NaN si no hay datos suficientes)."""
    n = len(lp)
    z = np.full(n, np.nan)
    if h <= 0 or sigma <= 0:
        return z
    idx = np.arange(n)
    prev = idx - h
    ok = prev >= 0
    ok[ok] &= ids[prev[ok]] == ids[idx[ok]]
    z[ok] = (lp[idx[ok]] - lp[prev[ok]]) / (sigma * math.sqrt(h))
    return z


def series(closes, sesiones, sigma, prm=None):
    """Percentil, pendiente corta y tendencia larga para cada minuto. sigma = σ por minuto calibrada."""
    p = {**DEFAULT, **(prm or {})}
    closes = np.asarray(closes, dtype=float)
    ids = ids_sesion(closes, sesiones)
    lp = np.log(closes)
    z_anc = _z_ventana(lp, ids, p["ancla"], sigma)
    pct = 100 * norm_cdf_vec(np.nan_to_num(z_anc))
    pct[np.isnan(z_anc)] = np.nan
    return dict(pct=pct, z_corto=_z_ventana(lp, ids, p["corto"], sigma),
                z_largo=_z_ventana(lp, ids, p["largo"], sigma), ids=ids)


def estados(s, prm=None):
    """Aplica la regla minuto a minuto (con memoria del 'armado'). Devuelve lista de estados."""
    p = {**DEFAULT, **(prm or {})}
    pct, zc, zl, ids = s["pct"], s["z_corto"], s["z_largo"], s["ids"]
    n = len(pct)
    out = ["NEUTRO"] * n
    arm_c = arm_v = -10 ** 9          # último minuto en que tocó zona de compra / venta
    ses_prev = None
    for i in range(n):
        if ids[i] != ses_prev:
            arm_c = arm_v = -10 ** 9; ses_prev = ids[i]
        if np.isnan(pct[i]):
            continue
        if pct[i] <= p["umbral_compra"]:
            arm_c = i
        if pct[i] >= p["umbral_venta"]:
            arm_v = i
        z = zc[i] if not np.isnan(zc[i]) else 0.0

        # ---- lado compra ----
        if not p["usar_confirmacion"]:
            if pct[i] <= p["umbral_compra"]:
                out[i] = "COMPRA"
        elif i - arm_c <= p["armado_min"] and pct[i] <= p["umbral_compra"] + p["margen"]:
            out[i] = "COMPRA" if z > p["z_confirma"] else "ESPERA · sigue cayendo"
        if out[i] == "COMPRA" and p["usar_tendencia"] and not np.isnan(zl[i]) and zl[i] < p["z_tendencia"]:
            out[i] = "BLOQUEADA · tendencia bajista"

        # ---- lado venta ----
        if out[i] == "NEUTRO":
            if not p["usar_confirmacion"]:
                if pct[i] >= p["umbral_venta"]:
                    out[i] = "VENTA"
            elif i - arm_v <= p["armado_min"] and pct[i] >= p["umbral_venta"] - p["margen"]:
                out[i] = "VENTA" if z < -p["z_confirma"] else "ESPERA · sigue subiendo"
    return out


def desde_config(config):
    """Parámetros de la estrategia a partir de config.py."""
    g = lambda n, d: getattr(config, n, d)
    return dict(ancla=config.ANCLA, umbral_compra=config.UMBRAL_COMPRA, umbral_venta=config.UMBRAL_VENTA,
                horizonte=config.HORIZONTE, armado_min=g("ARMADO_MIN", 30), z_salida=g("Z_SALIDA", -1.5),
                usar_confirmacion=g("USAR_CONFIRMACION", True), usar_tendencia=g("USAR_TENDENCIA", False),
                usar_salida_temprana=g("USAR_SALIDA_TEMPRANA", True))


def ahora(closes, sesiones, precio, sigma, prm=None):
    """Estado de la estrategia en este momento (agrega el último precio como punto final)."""
    closes = np.append(np.asarray(closes, dtype=float), precio)
    ses = None if sesiones is None else np.append(np.asarray(sesiones), np.asarray(sesiones)[-1])
    s = series(closes, ses, sigma, prm)
    e = estados(s, prm)
    f = lambda x: None if np.isnan(x) else float(x)
    return dict(estado=e[-1], explicacion=EXPLICA[e[-1]], pct=f(s["pct"][-1]), z_corto=f(s["z_corto"][-1]),
                z_largo=f(s["z_largo"][-1]), sigma=sigma, historial=e)


def niveles(entrada, sigma, prm=None):
    """Take-profit y stop-loss alrededor del precio de entrada."""
    p = {**DEFAULT, **(prm or {})}
    d = p["k_banda"] * sigma * math.sqrt(p["horizonte"])
    return entrada * math.exp(d), entrada * math.exp(-d)


def salida(precio, entrada, minutos_dentro, z_corto, sigma, hora_min, prm=None):
    """¿Vender ya una posición abierta? Devuelve (motivo o None)."""
    p = {**DEFAULT, **(prm or {})}
    tp, sl = niveles(entrada, sigma, p)
    if precio >= tp:
        return "take-profit"
    if precio <= sl:
        return "stop-loss"
    if p["usar_salida_temprana"] and precio < entrada and z_corto is not None and not np.isnan(z_corto) \
            and z_corto < p["z_salida"]:
        return "sigue cayendo"
    if minutos_dentro >= p["horizonte"]:
        return "horizonte"
    if hora_min >= 15 * 60 + 55:
        return "cierre del día"
    return None


# ---------------- backtest ----------------
VARIANTES = {
    "Solo percentil (regla original)": dict(usar_confirmacion=False, usar_tendencia=False, usar_salida_temprana=False),
    "+ esperar rebote": dict(usar_confirmacion=True, usar_tendencia=False, usar_salida_temprana=False),
    "+ esperar rebote + salida temprana": dict(usar_confirmacion=True, usar_tendencia=False, usar_salida_temprana=True),
    "+ todo + filtro de tendencia": dict(usar_confirmacion=True, usar_tendencia=True, usar_salida_temprana=True),
}


def backtest(df, prm=None, acciones=1, costo_pct=0.05, espera_min=30):
    """
    Simula la estrategia sobre las velas de df (t, close). Solo compras, una posición a la vez, todo cerrado al final
    del día. σ de cada día se estima SOLO con los días anteriores (no hace trampa mirando el futuro), así que el primer
    día del histórico no opera.
    costo_pct = costo por lado (spread + deslizamiento), en %. Devuelve (operaciones, curva).
    """
    p = {**DEFAULT, **(prm or {})}
    df = df.reset_index(drop=True)
    closes = df["close"].values.astype(float)
    fechas = df["t"].dt.date.values
    hm = (df["t"].dt.hour * 60 + df["t"].dt.minute).values
    dias = list(dict.fromkeys(fechas))
    ops = []

    for d_i, dia in enumerate(dias[1:], start=1):
        previo = np.isin(fechas, dias[:d_i])
        try:
            sigma = estimar_parametros(closes[previo], fechas[previo], escala=p["ancla"])["sigma"]
        except ValueError:
            continue
        sel = np.where(fechas == dia)[0]
        c = closes[sel]
        s = series(c, np.zeros(len(c), dtype=int), sigma, p)
        est = estados(s, p)
        pos = None
        ultima_salida = -10 ** 9
        for j in range(len(c)):
            h = hm[sel[j]]
            if pos is not None:
                motivo = salida(c[j], pos["entrada"], j - pos["j"], s["z_corto"][j], sigma, h, p)
                if motivo is None and j == len(c) - 1:
                    motivo = "cierre del día"
                if motivo:
                    bruto = c[j] / pos["entrada"] - 1
                    neto = bruto - 2 * costo_pct / 100
                    ops.append(dict(dia=str(dia), entrada_hora=df["t"].iloc[sel[pos["j"]]].strftime("%H:%M"),
                                    salida_hora=df["t"].iloc[sel[j]].strftime("%H:%M"),
                                    entrada=round(pos["entrada"], 4), salida=round(float(c[j]), 4), motivo=motivo,
                                    minutos=j - pos["j"], ret_pct=round(neto * 100, 3),
                                    ganancia_usd=round(neto * pos["entrada"] * acciones, 2)))
                    pos = None; ultima_salida = j
            elif est[j] == "COMPRA" and h < 15 * 60 + 45 and j - ultima_salida >= espera_min:
                pos = dict(entrada=float(c[j]), j=j)
    return ops


def resumen(ops):
    if not ops:
        return dict(operaciones=0, ganadoras_pct=float("nan"), ret_prom_pct=float("nan"),
                    ret_total_pct=0.0, ganancia_usd=0.0, peor_racha_usd=0.0)
    r = np.array([o["ret_pct"] for o in ops])
    usd = np.array([o["ganancia_usd"] for o in ops])
    curva = np.cumsum(usd)
    dd = float((curva - np.maximum.accumulate(np.concatenate([[0], curva]))[1:]).min())
    return dict(operaciones=len(ops), ganadoras_pct=100 * float((r > 0).mean()), ret_prom_pct=float(r.mean()),
                ret_total_pct=float(r.sum()), ganancia_usd=float(usd.sum()), peor_racha_usd=min(0.0, dd))
