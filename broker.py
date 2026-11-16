"""
Cuenta de PAPEL de Alpaca (dinero ficticio). Las órdenes se ejecutan con precios reales del mercado.
Mercado cerrado: la orden queda en cola y se ejecuta en la apertura.
"""
import datetime as dt
from zoneinfo import ZoneInfo

import requests

NY = ZoneInfo("America/New_York")


def _ny(ts):
    """Hora UTC de Alpaca -> texto hora Nueva York."""
    import pandas as pd
    return pd.to_datetime(ts, utc=True).tz_convert(NY).strftime("%Y-%m-%d %H:%M")

PAPER = "https://paper-api.alpaca.markets/v2"


def _h(key, secret):
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _ok(r):
    if r.status_code >= 400:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise RuntimeError(f"Alpaca ({r.status_code}): {msg}")
    return r.json() if r.text else {}


def cuenta(key, secret):
    return _ok(requests.get(f"{PAPER}/account", headers=_h(key, secret), timeout=15))


def reloj(key, secret):
    return _ok(requests.get(f"{PAPER}/clock", headers=_h(key, secret), timeout=15))


def posicion(symbol, key, secret):
    r = requests.get(f"{PAPER}/positions/{symbol}", headers=_h(key, secret), timeout=15)
    if r.status_code == 404:
        return None
    return _ok(r)


def comprar(symbol, qty, key, secret):
    return _ok(requests.post(f"{PAPER}/orders", headers=_h(key, secret), timeout=15,
                             json={"symbol": symbol, "qty": str(qty), "side": "buy", "type": "market",
                                   "time_in_force": "day"}))


def vender_todo(symbol, key, secret):
    r = requests.delete(f"{PAPER}/positions/{symbol}", headers=_h(key, secret), timeout=15)
    if r.status_code == 404:
        return None
    return _ok(r)


def ordenes(symbol, key, secret, limite=100):
    return _ok(requests.get(f"{PAPER}/orders", headers=_h(key, secret), timeout=15,
                            params={"status": "all", "symbols": symbol, "limit": limite, "direction": "desc"}))


def operaciones_cerradas(ords):
    """Empareja compras y ventas ejecutadas (FIFO) -> lista de operaciones con ganancia real."""
    llenas = [o for o in ords if o.get("status") == "filled"]
    llenas.sort(key=lambda o: o["filled_at"])
    cola, out = [], []
    for o in llenas:
        q, px = float(o["filled_qty"]), float(o["filled_avg_price"])
        if o["side"] == "buy":
            cola.append([q, px, o["filled_at"]])
        else:
            while q > 1e-9 and cola:
                lote = cola[0]
                m = min(q, lote[0])
                out.append(dict(compra=_ny(lote[2]), venta=_ny(o["filled_at"]),
                                acciones=m, precio_compra=lote[1], precio_venta=px,
                                ganancia_usd=round((px - lote[1]) * m, 2),
                                ret_pct=round((px / lote[1] - 1) * 100, 2)))
                lote[0] -= m; q -= m
                if lote[0] <= 1e-9:
                    cola.pop(0)
    return out[::-1]
