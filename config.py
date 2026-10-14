"""
Configuración.

Llaves de Alpaca (sirven las de Paper Trading): NO las pegues aquí si vas a subir esto a GitHub.
Se leen en este orden:
  1. Variables de entorno APCA_API_KEY_ID / APCA_API_SECRET_KEY
  2. Archivo local .streamlit/secrets.toml  (está en .gitignore, nunca se sube)
  3. En Streamlit Cloud: pestaña "Secrets" de la app (app.py las lee con st.secrets)
"""
import os
import pathlib


def _desde_secrets_toml(nombre):
    for p in (pathlib.Path(".streamlit/secrets.toml"), pathlib.Path.home() / ".streamlit" / "secrets.toml"):
        if p.exists():
            try:
                import tomllib
                return tomllib.loads(p.read_text(encoding="utf-8")).get(nombre)
            except Exception:
                return None
    return None


def _llave(nombre):
    return os.environ.get(nombre) or _desde_secrets_toml(nombre) or "PEGA_TU_LLAVE"


ALPACA_KEY = _llave("APCA_API_KEY_ID")
ALPACA_SECRET = _llave("APCA_API_SECRET_KEY")

SYMBOL = "SMCI"
SIMBOLOS_ALERTA = ["SMCI"]   # alertas.py vigila estos (ej. ["SMCI", "NVDA", "QBTS"])

# Telegram (para alertas.py). Mismo sitio que las llaves de Alpaca: secrets.toml o variables de entorno.
TELEGRAM_TOKEN = _llave("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _llave("TELEGRAM_CHAT_ID")

# "iex"  = tiempo real en el plan gratis (solo trades que pasan por la bolsa IEX).
# "sip"  = mercado completo, pero en el plan gratis NO deja pedir los últimos 15 min.
FEED = "iex"

DIAS_HISTORIA = 5            # sesiones de velas de 1 min para estimar mu y sigma
SOLO_HORARIO_REGULAR = True  # 9:30-16:00 hora de Nueva York

# Parámetros del modelo (minutos)
HORIZONTE = 60      # cuántos minutos hacia adelante proyectar
ANCLA = 60          # desde cuántos minutos atrás se mide dónde está el precio hoy
UMBRAL_COMPRA = 10  # percentil <= esto -> zona de compra
UMBRAL_VENTA = 90   # percentil >= esto -> zona de venta
USAR_DRIFT = False  # False = tendencia intradía = 0 (en 5 días el drift es ruido y sesga la señal hacia la tendencia reciente)

# Solo Montecarlo
N_SIMULACIONES = 10000
MODO_MC = "bootstrap"   # "bootstrap" = copia tramos de retornos reales (colas gordas) | "normal" = gaussiano
BLOQUE_MC = 30          # minutos por tramo en el block bootstrap (conserva la autocorrelación de corto plazo)
