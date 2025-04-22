import ccxt
import pandas as pd
import ta
from xgboost import XGBClassifier

def write_output_to_file(output, filename="output.txt"):
    with open(filename, "a") as file:
        file.write(output + "\n")

# Load trained model
model = XGBClassifier()
model.load_model("btc_model.json")

# Get the latest candle data
exchange = ccxt.binance()
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Add indicators to latest data
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
macd = ta.trend.MACD(df['close'])
df['macd_diff'] = macd.macd_diff()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
df.dropna(inplace=True)

# Predict signal for latest candle
latest_features = df.iloc[-1:][['rsi', 'ema20', 'ema50', 'macd_diff', 'atr']]
signal = model.predict(latest_features)[0]
latest_row = df.iloc[-1]
entry_price = latest_row['close']
atr = latest_row['atr']

# Calculate SL and TP
if signal == 1:  # BUY
    stop_loss = entry_price - 1.5 * atr
    take_profit = entry_price + 2.5 * atr
    signal_text = f"📢 Signal: BUY | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}"
else:  # SELL
    stop_loss = entry_price + 1.5 * atr
    take_profit = entry_price - 2.5 * atr
    signal_text = f"📢 Signal: SELL | Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}"

print(signal_text)
write_output_to_file(signal_text)
