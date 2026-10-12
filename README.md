# SMCI · GBM + Montecarlo (Alpaca en vivo)

## 1. Llaves de Alpaca (gratis)
1. Cuenta en https://alpaca.markets (no hay que fondear nada).
2. Dashboard → **Paper Trading** → **API Keys** → *Generate*. Guarda key y secret.

Plan gratis = tiempo real solo del feed **iex** (fracción del volumen, suficiente para SMCI).
`sip` es el mercado completo pero en plan gratis no entrega los últimos 15 min. Déjalo en `iex`.

## 2. Publicar en Streamlit Cloud (link con datos en vivo)
1. Crea un repo en https://github.com/new (privado sirve). Sube TODOS los archivos de esta carpeta
   ("uploading an existing file", arrastra y suelta). NO subas ningún archivo con llaves.
2. Entra a https://share.streamlit.io con tu cuenta de GitHub → **Create app** / **Deploy**.
3. Repo = el que creaste · Branch = main · Main file = `app.py`.
4. **Advanced settings → Secrets**: pega esto con tus llaves reales:
   ```
   APCA_API_KEY_ID = "PK..."
   APCA_API_SECRET_KEY = "..."
   ```
5. **Deploy**. En 1-2 min tienes el link `https://<nombre>.streamlit.app`. Ábrelo desde el celular.

Para cambiar llaves después: App settings → Secrets. Si nadie abre la app en varios días se duerme; se despierta sola al abrir el link.

## 3. Correr local (Windows, PowerShell)
```
cd carpeta_del_proyecto
pip install -r requirements.txt
```
Crea la carpeta `.streamlit` y dentro `secrets.toml` con el mismo contenido del paso 2.4 (usa `secrets.ejemplo.toml` de guía). Luego:
```
streamlit run app.py          # app en vivo, dos pestañas
python gbm_model.py           # Modelo 1 suelto
python montecarlo_model.py    # Modelo 2 suelto
python data_alpaca.py         # solo probar la conexión
```
Si no creas el secrets.toml, la app te pide las llaves en la barra lateral.

## Cómo leer la señal
- Se estima σ y drift con velas de 1 min de los últimos N días.
- Se mira dónde está el precio actual dentro de la distribución que el modelo proyectaba desde hace `ANCLA` min.
- Percentil ≤ 10 → **COMPRA**. Percentil ≥ 90 → **VENTA**. Entre medio → **NEUTRO**.
- Take-profit = P90 y stop-loss = P10 a `HORIZONTE` min. Montecarlo además dice qué tan probable es tocar TP antes que SL dentro de ese tiempo.

Modelo 1 (GBM) = fórmula cerrada, asume retornos normales.
Modelo 2 (Montecarlo bootstrap) = remuestrea los retornos reales, respeta colas gordas. Si difieren mucho, ese día la distribución no es normal.

Son probabilidades, no certezas. No es asesoría financiera.
