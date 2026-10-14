"""
Alertas a Telegram cuando la señal pasa a COMPRA o VENTA.

Corre en bucle (python alertas.py) mientras el mercado está abierto; revisa cada CADA_SEG segundos
y te escribe al celular solo cuando la señal CAMBIA (no spamea).

Setup Telegram (2 min):
  1. En Telegram busca @BotFather -> /newbot -> nombre -> te da un TOKEN.
  2. Abre el chat con tu bot nuevo y escríbele "hola".
  3. Pon el TOKEN en .streamlit/secrets.toml (o variable de entorno TELEGRAM_TOKEN).
     El chat_id lo saca solo el script la primera vez.
"""
import sys
import time
import datetime as dt
from zoneinfo import ZoneInfo

import requests

import config
import data_alpaca as da
import gbm_model as gbm
import montecarlo_model as mc

NY = ZoneInfo("America/New_York")
TG = "https://api.telegram.org/bot{token}/{method}"

# ---------- ajustes ----------
SIMBOLOS = getattr(config, "SIMBOLOS_ALERTA", [config.SYMBOL])
CADA_SEG = 60              # cada cuánto revisar
AVISAR_VUELTA_NEUTRO = True
EMOJI = {"COMPRA": "🟢", "VENTA": "🔴", "NEUTRO": "⚪"}


# ---------- telegram ----------
def tg_enviar(token, chat_id, texto):
    r = requests.post(TG.format(token=token, method="sendMessage"),
                      json={"chat_id": chat_id, "text": texto, "parse_mode": "HTML"}, timeout=15)
    if r.status_code != 200:
        print("Telegram no envió:", r.text[:200])
    return r.ok


def tg_detectar_chat_id(token):
    """Lee los últimos mensajes que le llegaron al bot y devuelve el chat_id del primero."""
    r = requests.get(TG.format(token=token, method="getUpdates"), timeout=15)
    r.raise_for_status()
    for u in reversed(r.json().get("result", [])):
        m = u.get("message") or u.get("edited_message")
        if m and "chat" in m:
            return str(m["chat"]["id"])
    return None


def cargar_telegram():
    token = config._llave("TELEGRAM_TOKEN")
    if "PEGA_TU" in token:
        sys.exit("Falta TELEGRAM_TOKEN. Créalo con @BotFather y ponlo en .streamlit/secrets.toml")
    chat_id = config._llave("TELEGRAM_CHAT_ID")
    if "PEGA_TU" in chat_id:
        chat_id = tg_detectar_chat_id(token)
        if not chat_id:
            sys.exit("No encuentro tu chat. Escríbele 'hola' a tu bot en Telegram y vuelve a correr.")
        print(f"chat_id detectado: {chat_id}  (ponlo en secrets.toml como TELEGRAM_CHAT_ID para no buscarlo cada vez)")
    return token, chat_id


# ---------- mercado ----------
def mercado_abierto(ahora=None):
    t = ahora or dt.datetime.now(NY)
    if t.weekday() >= 5:
        return False
    hm = t.hour * 60 + t.minute
    return 570 <= hm < 960  # 9:30 - 16:00 NY


def segundos_hasta_apertura(ahora=None):
    t = ahora or dt.datetime.now(NY)
    apertura = t.replace(hour=9, minute=30, second=0, microsecond=0)
    if t >= apertura.replace(hour=16, minute=0) or t.weekday() >= 5:
        apertura += dt.timedelta(days=1)
    while apertura.weekday() >= 5 or apertura <= t:
        apertura += dt.timedelta(days=1)
    return max(60, int((apertura - t).total_seconds()))


# ---------- señal ----------
def evaluar(symbol):
    df, px, ts = da.cargar(symbol, config.ALPACA_KEY, config.ALPACA_SECRET, dias=config.DIAS_HISTORIA,
                           feed=config.FEED, solo_regular=config.SOLO_HORARIO_REGULAR)
    closes = df["close"].values
    sesiones = df["t"].dt.date.values
    r1 = gbm.analizar(closes, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA, config.UMBRAL_VENTA,
                      sesiones=sesiones, usar_drift=config.USAR_DRIFT)
    r2 = mc.analizar(closes, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA, config.UMBRAL_VENTA,
                     n_sims=config.N_SIMULACIONES, modo=config.MODO_MC, bloque=config.BLOQUE_MC,
                     sesiones=sesiones, usar_drift=config.USAR_DRIFT)
    return r1, r2, ts


def mensaje(symbol, r1, r2, ts):
    s1, s2 = r1["senal"], r2["senal"]
    titulo = s1 if s1 != "NEUTRO" else s2
    acuerdo = "✅ los dos modelos coinciden" if s1 == s2 else "⚠️ los modelos no coinciden"
    return (
        f"{EMOJI[titulo]} <b>{symbol} · {titulo}</b>\n"
        f"Precio {r1['precio']:.2f} ({(r1['precio'] / r1['precio_ancla'] - 1) * 100:+.2f}% en {r1['ancla']} min)\n"
        f"GBM: {s1} · percentil {r1['percentil']:.0f}\n"
        f"Montecarlo: {s2} · percentil {r2['percentil']:.0f}\n"
        f"{acuerdo}\n"
        f"TP {r1['take_profit']:.2f} · SL {r1['stop_loss']:.2f} (a {r1['horizonte']} min)\n"
        f"<i>{ts:%H:%M} NY</i>"
    )


def estado_de(r1, r2):
    s1, s2 = r1["senal"], r2["senal"]
    if s1 != "NEUTRO":
        return s1
    if s2 != "NEUTRO":
        return s2
    return "NEUTRO"


def bucle(token, chat_id):
    ultimo = {s: "NEUTRO" for s in SIMBOLOS}
    tg_enviar(token, chat_id, f"🤖 Alertas activas para {', '.join(SIMBOLOS)}. Te aviso cuando haya COMPRA o VENTA.")
    print("Alertas activas. Ctrl+C para parar.")
    while True:
        if not mercado_abierto():
            espera = segundos_hasta_apertura()
            print(f"Mercado cerrado. Duermo {espera // 3600}h {espera % 3600 // 60}m hasta la apertura.")
            time.sleep(espera)
            continue
        for symbol in SIMBOLOS:
            try:
                r1, r2, ts = evaluar(symbol)
            except Exception as e:
                print(f"{dt.datetime.now(NY):%H:%M} {symbol}: error {e}")
                continue
            estado = estado_de(r1, r2)
            print(f"{dt.datetime.now(NY):%H:%M} {symbol} {r1['precio']:.2f} · GBM {r1['senal']} ({r1['percentil']:.0f}) · "
                  f"MC {r2['senal']} ({r2['percentil']:.0f})")
            if estado != ultimo[symbol]:
                if estado != "NEUTRO":
                    tg_enviar(token, chat_id, mensaje(symbol, r1, r2, ts))
                elif AVISAR_VUELTA_NEUTRO:
                    tg_enviar(token, chat_id, f"⚪ {symbol} volvió a NEUTRO · precio {r1['precio']:.2f}")
                ultimo[symbol] = estado
        time.sleep(CADA_SEG)


if __name__ == "__main__":
    token, chat_id = cargar_telegram()
    if "--prueba" in sys.argv:
        ok = tg_enviar(token, chat_id, "✅ Prueba: el bot de alertas SMCI funciona.")
        sys.exit(0 if ok else 1)
    try:
        bucle(token, chat_id)
    except KeyboardInterrupt:
        print("\nAlertas detenidas.")
