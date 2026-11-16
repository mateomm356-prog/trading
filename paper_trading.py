"""
PAPER TRADING — compra y vende solo en tu cuenta Alpaca de papel (dinero ficticio) siguiendo la señal del modelo.

Reglas:
  - Solo compras (nunca venta en corto). Señal COMPRA -> compra MONTO_USD a precio de mercado.
  - Una posición abierta a la vez por acción.
  - Se vende cuando pasa lo primero: toca take-profit (P90), toca stop-loss (P10), se cumple el horizonte (60 min),
    o son las HORA_CIERRE (15:55 NY). Nunca se queda nada abierto de un día para otro.
  - Cada entrada y salida se guarda en operaciones.csv.

Correr:  python paper_trading.py          (déjalo abierto en horario de mercado)
Probar:  python paper_trading.py --estado (muestra cuenta y posiciones, no opera)
"""
import csv
import sys
import time
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import config
import data_alpaca as da
import gbm_model as gbm
import montecarlo_model as mc
import estrategia as es

NY = ZoneInfo("America/New_York")
PAPER = "https://paper-api.alpaca.markets/v2"

# ---------- ajustes ----------
SIMBOLOS = getattr(config, "SIMBOLOS_PAPEL", [config.SYMBOL])
MONTO_USD = 120            # dólares ficticios por entrada
CADA_SEG = 60              # cada cuánto revisar
HORA_CIERRE = (15, 55)     # NY: a esta hora se cierra todo
PRM = es.desde_config(config)   # regla de entrada/salida con "ventana segura" (ver estrategia.py y config.py)
ESPERA_TRAS_SALIDA = 30    # minutos sin volver a entrar en la misma acción después de salir
ARCHIVO = Path("operaciones.csv")


# ---------- alpaca trading ----------
def _h():
    return {"APCA-API-KEY-ID": config.ALPACA_KEY, "APCA-API-SECRET-KEY": config.ALPACA_SECRET}


def cuenta():
    r = requests.get(f"{PAPER}/account", headers=_h(), timeout=15); r.raise_for_status(); return r.json()


def posiciones():
    r = requests.get(f"{PAPER}/positions", headers=_h(), timeout=15); r.raise_for_status()
    return {p["symbol"]: p for p in r.json()}


def comprar(symbol, usd):
    r = requests.post(f"{PAPER}/orders", headers=_h(), timeout=15,
                      json={"symbol": symbol, "notional": str(usd), "side": "buy",
                            "type": "market", "time_in_force": "day"})
    r.raise_for_status(); return r.json()


def cerrar(symbol):
    r = requests.delete(f"{PAPER}/positions/{symbol}", headers=_h(), timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status(); return r.json()


def precio_relleno(order_id, intentos=10):
    """Espera a que la orden se ejecute y devuelve el precio promedio real."""
    for _ in range(intentos):
        r = requests.get(f"{PAPER}/orders/{order_id}", headers=_h(), timeout=15); r.raise_for_status()
        o = r.json()
        if o.get("status") == "filled" and o.get("filled_avg_price"):
            return float(o["filled_avg_price"]), float(o["filled_qty"])
        time.sleep(1)
    return None, None


# ---------- registro ----------
def registrar(fila):
    nuevo = not ARCHIVO.exists()
    with ARCHIVO.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "entrada_hora", "entrada_precio", "qty", "tp", "sl",
                                          "salida_hora", "salida_precio", "motivo", "ganancia_pct", "ganancia_usd"])
        if nuevo:
            w.writeheader()
        w.writerow(fila)


# ---------- mercado ----------
def ahora():
    return dt.datetime.now(NY)


def mercado_abierto(t=None):
    t = t or ahora()
    return t.weekday() < 5 and 570 <= t.hour * 60 + t.minute < 960


def hora_de_cerrar(t=None):
    t = t or ahora()
    return t.hour * 60 + t.minute >= HORA_CIERRE[0] * 60 + HORA_CIERRE[1]


def segundos_hasta_apertura(t=None):
    t = t or ahora()
    ap = t.replace(hour=9, minute=30, second=0, microsecond=0)
    if t >= ap or t.weekday() >= 5:
        ap += dt.timedelta(days=1)
    while ap.weekday() >= 5:
        ap += dt.timedelta(days=1)
    return max(60, int((ap - t).total_seconds()))


# ---------- señal ----------
def evaluar(symbol):
    df, px, ts = da.cargar(symbol, config.ALPACA_KEY, config.ALPACA_SECRET, dias=config.DIAS_HISTORIA,
                           feed=config.FEED, solo_regular=config.SOLO_HORARIO_REGULAR)
    closes, ses = df["close"].values, df["t"].dt.date.values
    r1 = gbm.analizar(closes, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA, config.UMBRAL_VENTA,
                      sesiones=ses, usar_drift=getattr(config, "USAR_DRIFT", False))
    r2 = mc.analizar(closes, px, config.HORIZONTE, config.ANCLA, config.UMBRAL_COMPRA, config.UMBRAL_VENTA,
                     n_sims=config.N_SIMULACIONES, modo=config.MODO_MC, bloque=getattr(config, "BLOQUE_MC", 30),
                     sesiones=ses, usar_drift=getattr(config, "USAR_DRIFT", False))
    ah = es.ahora(closes, ses, px, r1["params"]["sigma"], PRM)
    return r1, r2, px, ah


# ---------- bucle ----------
def bucle():
    abiertas = {}   # symbol -> dict(entrada_hora, entrada_precio, qty, tp, sl, vence)
    ultima_salida = {}
    c = cuenta()
    msg = f"🤖 Paper trading activo · {', '.join(SIMBOLOS)} · ${MONTO_USD} por entrada · equity ${float(c['equity']):,.0f}"
    print(msg)

    while True:
        t = ahora()
        if not mercado_abierto(t):
            if abiertas:
                for s in list(abiertas):
                    salir(s, abiertas, "mercado cerrado")
            esp = segundos_hasta_apertura(t)
            print(f"{t:%H:%M} mercado cerrado, duermo {esp // 3600}h {esp % 3600 // 60}m"); time.sleep(esp); continue

        for symbol in SIMBOLOS:
            try:
                r1, r2, px, ah = evaluar(symbol)
            except Exception as e:
                print(f"{t:%H:%M} {symbol}: error datos {e}"); continue

            pos = abiertas.get(symbol)
            print(f"{t:%H:%M} {symbol} {px:.2f} · {ah['estado']} · percentil {r1['percentil']:.0f} · pendiente {ah['z_corto'] or 0:+.1f}"
                  + (f" · abierta @ {pos['entrada_precio']:.2f} TP {pos['tp']:.2f} SL {pos['sl']:.2f}" if pos else ""))

            if pos:
                mins = int((t - pos["entrada_hora"]).total_seconds() // 60)
                motivo = es.salida(px, pos["entrada_precio"], mins, ah["z_corto"], pos["sigma"],
                                   t.hour * 60 + t.minute, PRM)
                if motivo is None and hora_de_cerrar(t):
                    motivo = "cierre del día"
                if motivo:
                    salir(symbol, abiertas, motivo); ultima_salida[symbol] = t
            elif (ah["estado"] == "COMPRA" and not hora_de_cerrar(t) and r1["ancla"] >= config.ANCLA
                  and (symbol not in ultima_salida or t - ultima_salida[symbol] >= dt.timedelta(minutes=ESPERA_TRAS_SALIDA))):
                entrar(symbol, abiertas, r1, t, ah["sigma"])

        time.sleep(CADA_SEG)


def entrar(symbol, abiertas, r1, t, sigma):
    try:
        o = comprar(symbol, MONTO_USD)
        precio, qty = precio_relleno(o["id"])
        if precio is None:
            print("orden no se llenó, la cancelo"); requests.delete(f"{PAPER}/orders/{o['id']}", headers=_h(), timeout=15); return
    except Exception as e:
        print(f"error al comprar: {e}"); return
    tp, sl = es.niveles(precio, sigma, PRM)   # TP/SL alrededor del precio real de entrada
    abiertas[symbol] = dict(entrada_hora=t, entrada_precio=precio, qty=qty, tp=tp, sl=sl, sigma=sigma,
                            vence=t + dt.timedelta(minutes=config.HORIZONTE))
    m = (f"🟢 <b>COMPRÉ {symbol}</b> {qty:.4f} acc @ {precio:.2f}\nTP {tp:.2f} (+{(tp / precio - 1) * 100:.2f}%) · "
         f"SL {sl:.2f} ({(sl / precio - 1) * 100:.2f}%) · percentil {r1['percentil']:.0f}\n<i>{t:%H:%M} NY</i>")
    print(m.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))


def salir(symbol, abiertas, motivo):
    pos = abiertas.pop(symbol)
    try:
        o = cerrar(symbol)
        precio, _ = precio_relleno(o["id"]) if o else (None, None)
    except Exception as e:
        print(f"error al vender: {e}"); abiertas[symbol] = pos; return
    if precio is None:
        try:
            precio = float(posiciones().get(symbol, {}).get("current_price", pos["entrada_precio"]))
        except Exception:
            precio = pos["entrada_precio"]
    t = ahora()
    g_pct = (precio / pos["entrada_precio"] - 1) * 100
    g_usd = (precio - pos["entrada_precio"]) * pos["qty"]
    registrar(dict(symbol=symbol, entrada_hora=pos["entrada_hora"].strftime("%Y-%m-%d %H:%M"),
                   entrada_precio=round(pos["entrada_precio"], 4), qty=round(pos["qty"], 4), tp=round(pos["tp"], 4),
                   sl=round(pos["sl"], 4), salida_hora=t.strftime("%Y-%m-%d %H:%M"), salida_precio=round(precio, 4),
                   motivo=motivo, ganancia_pct=round(g_pct, 3), ganancia_usd=round(g_usd, 2)))
    e = "🟢" if g_usd >= 0 else "🔴"
    m = (f"{e} <b>VENDÍ {symbol}</b> @ {precio:.2f} · {motivo}\nResultado {g_pct:+.2f}% (${g_usd:+.2f})\n<i>{t:%H:%M} NY</i>")
    print(m.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))


def resumen():
    if not ARCHIVO.exists():
        print("Todavía no hay operaciones."); return
    filas = list(csv.DictReader(ARCHIVO.open(encoding="utf-8")))
    g = [float(f["ganancia_pct"]) for f in filas]
    usd = sum(float(f["ganancia_usd"]) for f in filas)
    win = sum(1 for x in g if x > 0)
    print(f"{len(g)} operaciones · ganadoras {win} ({100 * win / len(g):.0f}%) · promedio {sum(g) / len(g):+.2f}% · total ${usd:+.2f}")
    for m in ("take-profit", "stop-loss", "horizonte", "cierre del día"):
        k = [float(f["ganancia_pct"]) for f in filas if f["motivo"] == m]
        if k: print(f"  {m}: {len(k)} · promedio {sum(k) / len(k):+.2f}%")


if __name__ == "__main__":
    if "--estado" in sys.argv:
        c = cuenta(); print(f"Equity ${float(c['equity']):,.2f} · cash ${float(c['cash']):,.2f}")
        for s, p in posiciones().items():
            print(f"  {s}: {p['qty']} acc @ {float(p['avg_entry_price']):.2f} · ahora {float(p['current_price']):.2f} · {float(p['unrealized_plpc']) * 100:+.2f}%")
        resumen(); sys.exit()
    if "--resumen" in sys.argv:
        resumen(); sys.exit()
    try:
        bucle()
    except KeyboardInterrupt:
        print("\nDetenido. Si quedó algo abierto, ciérralo en el dashboard de Alpaca o corre de nuevo.")
