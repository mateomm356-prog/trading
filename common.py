"""Funciones compartidas por los dos modelos (estadística básica, sin scipy)."""
import math
import numpy as np

MIN_POR_DIA = 390        # minutos de sesión regular
DIAS_POR_ANO = 252


def log_returns(closes):
    c = np.asarray(closes, dtype=float)
    return np.diff(np.log(c))


def estimar_parametros(closes):
    """
    Estima los parámetros del GBM a partir de precios de cierre de 1 minuto.
      m     = drift del log-precio por minuto  (= mu - sigma^2/2)
      sigma = volatilidad por minuto
      mu    = drift del precio (GBM)           (= m + sigma^2/2)
    """
    r = log_returns(closes)
    if len(r) < 30:
        raise ValueError(f"Muy pocos datos ({len(r)} retornos). Necesito al menos 30 precios.")
    m = float(r.mean())
    sigma = float(r.std(ddof=1))
    return {
        "retornos": r,
        "n": int(len(r)),
        "m": m,
        "sigma": sigma,
        "mu": m + sigma ** 2 / 2,
        "sigma_dia": sigma * math.sqrt(MIN_POR_DIA),
        "sigma_anual": sigma * math.sqrt(MIN_POR_DIA * DIAS_POR_ANO),
        "mu_anual": (m + sigma ** 2 / 2) * MIN_POR_DIA * DIAS_POR_ANO,
    }


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
