"""
Conexión a Alpaca Market Data v2 (REST). Funciona con llaves de Paper Trading gratis.
Plan gratis: feed "iex" = tiempo real; feed "sip" = completo pero sin los últimos 15 min.
"""
import datetime as dt
import requests
import pandas as pd

BASE = "https://data.alpaca.markets/v2/stocks"


def _headers(key, secret):
    if not key or "PEGA_TU" in key:
        raise RuntimeError("Falta la llave de Alpaca. Ponla en .streamlit/secrets.toml, en Secrets de Streamlit Cloud, o en la barra lateral.")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"}


def _check(resp):
    if resp.status_code == 401 or resp.status_code == 403:
        try:
            msg = resp.json().get("message", "")
        except Exception:
            msg = resp.text
        if "subscription" in msg.lower() or "sip" in msg.lower():
            raise RuntimeError("Tu plan gratis no permite ese feed/rango. Usa FEED = 'iex'. Alpaca dice: " + msg)
        raise RuntimeError("Alpaca rechazó las llaves (401/403). Revisa key y secret. Alpaca dice: " + msg)
    resp.raise_for_status()
    return resp.json()


def get_bars(symbol, key, secret, dias=5, feed="iex", timeframe="1Min", solo_regular=True):
    """Velas de 1 min de las últimas `dias` sesiones. Devuelve DataFrame t/open/high/low/close/volume (hora NY)."""
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=dias * 2 + 4)  # margen por fines de semana
    params = {
        "timeframe": timeframe,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "feed": feed,
        "adjustment": "raw",
        "sort": "asc",
    }
    rows, token = [], None
    while True:
        if token:
            params["page_token"] = token
        j = _check(requests.get(f"{BASE}/{symbol}/bars", headers=_headers(key, secret), params=params, timeout=20))
        rows += j.get("bars") or []
        token = j.get("next_page_token")
        if not token:
            break

    if not rows:
        raise RuntimeError(f"Alpaca no devolvió velas para {symbol}. ¿Símbolo correcto? ¿Mercado abrió en ese rango?")

    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["t", "open", "high", "low", "close", "volume"]]

    if solo_regular:
        hm = df["t"].dt.hour * 60 + df["t"].dt.minute
        df = df[(hm >= 570) & (hm < 960)]  # 9:30 - 16:00 NY

    sesiones = df["t"].dt.date.unique()[-dias:]
    df = df[df["t"].dt.date.isin(sesiones)].reset_index(drop=True)
    return df


def get_ultimo_precio(symbol, key, secret, feed="iex"):
    """Último trade (precio, hora NY)."""
    j = _check(requests.get(f"{BASE}/{symbol}/trades/latest", headers=_headers(key, secret),
                            params={"feed": feed}, timeout=10))
    t = j["trade"]
    return float(t["p"]), pd.to_datetime(t["t"], utc=True).tz_convert("America/New_York")


def cargar(symbol, key, secret, dias=5, feed="iex", solo_regular=True):
    """Velas + último precio en una sola llamada. Devuelve (df, precio_actual, hora_precio)."""
    df = get_bars(symbol, key, secret, dias=dias, feed=feed, solo_regular=solo_regular)
    try:
        px, ts = get_ultimo_precio(symbol, key, secret, feed=feed)
    except Exception:
        px, ts = float(df["close"].iloc[-1]), df["t"].iloc[-1]
    return df, px, ts


if __name__ == "__main__":
    import config
    df, px, ts = cargar(config.SYMBOL, config.ALPACA_KEY, config.ALPACA_SECRET,
                        dias=config.DIAS_HISTORIA, feed=config.FEED, solo_regular=config.SOLO_HORARIO_REGULAR)
    print(df.tail())
    print(f"\n{len(df)} velas | último precio {config.SYMBOL}: {px:.2f} @ {ts}")
