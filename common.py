"""Funciones compartidas por los dos modelos (estadística básica, sin scipy)."""
import math
import numpy as np

MIN_POR_DIA = 390        # minutos de sesión regular
DIAS_POR_ANO = 252


# ---------------- retornos ----------------
def ids_sesion(closes, sesiones=None):
    """Convierte etiquetas de sesión (fechas) en enteros 0..K-1 en orden cronológico."""
    n = len(closes)
    if sesiones is None:
        return np.zeros(n, dtype=int)
    _, ids = np.unique(np.asarray(sesiones), return_inverse=True)
    return ids.reshape(-1)


def retornos_por_sesion(closes, sesiones=None):
    """Lista de arrays de log-retornos de 1 min, uno por sesión: NUNCA cruza la noche (los gaps de apertura no cuentan)."""
    closes = np.asarray(closes, dtype=float)
    ids = ids_sesion(closes, sesiones)
    out = []
    for k in np.unique(ids):
        c = closes[ids == k]
        if len(c) > 1:
            out.append(np.diff(np.log(c)))
    return out


def retornos_ventana(rs, h):
    """Log-retornos de ventanas de h minutos (solapadas), sin cruzar sesión. Es la 'población de horas' real."""
    out = []
    for r in rs:
        if len(r) >= h:
            cs = np.concatenate([[0.0], np.cumsum(r)])
            out.append(cs[h:] - cs[:-h])
    return np.concatenate(out) if out else np.array([])


def minutos_en_sesion_actual(closes, sesiones=None):
    """Cuántos retornos de 1 min hay dentro de la última sesión (para no anclar en el día anterior)."""
    ids = ids_sesion(closes, sesiones)
    return int((ids == ids[-1]).sum()) - 1


def estimar_parametros(closes, sesiones=None, escala=60, usar_drift=False):
    """
    Estima los parámetros del GBM a partir de cierres de 1 min.

    Problema clásico: la sigma de 1 min viene inflada por el rebote bid-ask (ruido de microestructura) y, si no
    se separan las sesiones, por los gaps de apertura. Escalarla con sqrt(60) sobrestima lo que se mueve una hora
    y el percentil queda comprimido (nunca llega a 10 ni a 90).

    Solución: calibrar sigma a la ESCALA del ancla (ventanas de `escala` min dentro de cada sesión) y convertirla
    a sigma/min efectiva = sigma_h / sqrt(h). Devuelve también el 'variance ratio' (sigma_cal^2 / sigma_1min^2):
    si es < 1 el ruido de 1 min estaba inflando la volatilidad.
    """
    rs = retornos_por_sesion(closes, sesiones)
    r1 = np.concatenate(rs) if rs else np.array([])
    if len(r1) < 30:
        raise ValueError(f"Muy pocos datos ({len(r1)} retornos). Necesito al menos 30 precios.")

    sigma1 = float(r1.std(ddof=1))
    m_est = float(r1.mean())
    m = m_est if usar_drift else 0.0

    escala = int(max(5, escala))
    rh = retornos_ventana(rs, escala)
    if len(rh) >= 20:
        sigma_h = float(rh.std(ddof=1))
        sigma = sigma_h / math.sqrt(escala)
    else:                                   # no hay ventanas suficientes: caer al escalado clásico
        sigma = sigma1
        sigma_h = sigma1 * math.sqrt(escala)

    return {
        "retornos": r1,
        "retornos_sesion": rs,
        "ventanas": rh,                     # retornos reales de `escala` min
        "escala": escala,
        "n": int(len(r1)),
        "m": m,
        "m_est": m_est,
        "usar_drift": usar_drift,
        "sigma1": sigma1,                   # sigma/min bruta (1 min)
        "sigma": sigma,                     # sigma/min calibrada a la escala del ancla
        "sigma_h": sigma_h,                 # sigma de una ventana de `escala` min
        "vr": (sigma / sigma1) ** 2 if sigma1 > 0 else 1.0,
        "mu": m + sigma ** 2 / 2,
        "sigma_dia": sigma * math.sqrt(MIN_POR_DIA),
        "sigma_anual": sigma * math.sqrt(MIN_POR_DIA * DIAS_POR_ANO),
        "mu_anual": (m + sigma ** 2 / 2) * MIN_POR_DIA * DIAS_POR_ANO,
    }


# ---------------- normal ----------------
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_cdf_vec(z):
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def norm_ppf(p):
    """Inversa de la normal estándar (algoritmo de Acklam, error ~1e-9)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p debe estar en (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---------------- señal ----------------
def senal(percentil, umbral_compra=10, umbral_venta=90):
    """Zona según el percentil en que cae el precio actual dentro de la distribución del modelo."""
    if percentil <= umbral_compra:
        return "COMPRA"
    if percentil >= umbral_venta:
        return "VENTA"
    return "NEUTRO"


def explicar_senal(s, percentil, ancla):
    if s == "COMPRA":
        return (f"El precio cayó al percentil {percentil:.0f} de lo que el modelo esperaba "
                f"desde hace {ancla} min: está estadísticamente barato frente a su propia dinámica.")
    if s == "VENTA":
        return (f"El precio subió al percentil {percentil:.0f} de lo que el modelo esperaba "
                f"desde hace {ancla} min: está estadísticamente caro frente a su propia dinámica.")
    return (f"El precio está en el percentil {percentil:.0f}: dentro del rango normal. "
            f"Sin ventaja estadística, mejor esperar.")


def percentil_empirico(obs, ventanas):
    """Percentil del retorno observado entre las ventanas reales del histórico (sin modelo)."""
    if len(ventanas) == 0:
        return float("nan")
    return 100.0 * float((ventanas < obs).mean())


def calibracion(percentiles_hist, umbral_compra, umbral_venta):
    """Qué fracción de las ventanas históricas habría caído en zona COMPRA / VENTA con este modelo.
    Si el modelo está bien calibrado, debe dar ≈ umbral_compra% y ≈ (100-umbral_venta)%."""
    p = np.asarray(percentiles_hist, dtype=float)
    if len(p) == 0:
        return {"compra": float("nan"), "venta": float("nan"), "n": 0}
    return {"compra": 100 * float((p <= umbral_compra).mean()),
            "venta": 100 * float((p >= umbral_venta).mean()),
            "n": int(len(p))}
