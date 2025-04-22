import ccxt
import pandas as pd
import ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from datetime import datetime



# try:

#     with open("output.txt", "a") as file:
#         file.write("Some output\n")
#         print("Saving to:", os.path.abspath("output.txt"))
# except Exception as e:
#     print("Error writing to file:", e)



def write_output_to_file(output, filename="output.txt"):
    with open(filename, "a") as file:  # 'a' mode appends to the file
        file.write(output + "\n")


# Load BTCUSDT OHLCV historical data using ccxt (Binance)
exchange = ccxt.binance()
bars = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=1000)
df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Add indicators
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
macd = ta.trend.MACD(df['close'])
df['macd_diff'] = macd.macd_diff()
df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

# Target: 1 if price goes up next candle, else 0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# Features and training
features = ['rsi', 'ema20', 'ema50', 'macd_diff', 'atr']
X = df[features]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model to file
model.save_model("btc_model.json")
now = datetime.now()


# Optional: Accuracy
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
output = str(now)+" Accuracy:"+str(accuracy_score(y_test, model.predict(X_test)))
write_output_to_file(output)

