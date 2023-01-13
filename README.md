# Lightning-Fast Indicators

## Accesses candles through the Alpaca Market Data API and calculates technical indicators.

You can retrieve years of historical data for over 6000 symbols with one click of a button without fear of overwhelming your RAM. These modules were written to create neatly formatted data files to use when backtesting day trading strategies. 
- dataprocessor.py: Imports data and calculates indicators before saving to two .Feather files. 
- One file contains data for all symbols for one month. The other file contains the start and end indices of each symbol in the first file.
- Each month has a separate set of .Feather files. 
- candlesticks.py: Produces candlestick data including functionality for adding technical indicators.

![Opener](img/Banner2.JPG)

## Getting Started

1) Clone this repository.
2) Download all required libraries using pip install.
3) Make an account with Alpaca (go to Alpaca.markets)

**Data retrieval and indicator calculation:**

1) Specify user parameters in dataprocessor.py:
    - API key: *line 28*
    - Secret key: *line 29*
    - Place to save files: *line 30*
2) Specify data parameters in dataprocessor.py:
    - start_date: *line 1282* (YYYY, MM, dd) 
    - end_date: *line 1283*
    - timeframe: *line 1284* (e.g. '5min')
    - symbols: *line 1286* - ['AAPL', 'SPY'] (set to import 1602 symbols found in CSV file by default).
3) Click run.

**Accessing data in your script:**

1) Read the two .Feather files using:

```
data_file = pd.read_feather("""<your filepath>\\YYYY\\MM\\5min.feather""")
index_file = pd.read_feather("""<your filepath>\\YYYY\\MM\\5min_indices.feather""")
```
2) Get the start and end indices of your desired symbol:
```
aapl_index = np.where(index_file['symbol'] == 'AAPL')[0][0]
start_index = index_file['start'][aapl_index]
end_index = index_file['end'][aapl_index]
```
 3) Put symbol data in a dictionary:
```
aapl_dict = {header: data_file[header][start_index:end_index].to_numpy() for header in data_file.columns}
```
4) Use dictionary key from tables below to access Numpy array for each query:
```
close_price = aapl_dict['close']
open_price = aapl_dict['open']
ema_20 = aapl_dict['20 EMA']
...
```
**Market data with associated dictionary key:**

|Name| Dictionary Key|
|---|---|
|Open| 'open'|
|High| 'high'|
|Low| 'low'|
|Close| 'close'|
|Volume| 'volume'|
|Timestamp 1*| 'time'|
|Timestamp 2**| 'integer time'|

\* Timestamp as a datetime.datetime object \
\** Integer representation of timestamp - YYYYMMddHHmm (no hyphens or colons)

**Indicators calculated with associated dictionary key:**

| Indicator | Dictionary Key |
|-----------|----|
|9 EMA | '9 EMA'|
|20 EMA | '20 EMA'|
|50 EMA | '50 EMA'|
|200 EMA| '200 EMA'|
|MACD | 'MACD'|
|Signal| 'signal'|
|RSI| 'RSI'|
|VWAP| 'VWAP'|
|Bollinger Band (upper)| 'Bollinger (Upper)'|
|Bollinger Band (lower)| 'Bollinger (Lower)'|
|Consecutives*| 'Consecutives'|
|Hammer**| 'Hammer'|
|Shooting Star| 'Shooting Star'|
|Spinning Top| 'Spinning Top'|
|Green Marubozu| 'Green Marubozu'|
|Red Marubozu| 'Red Marubozu'|
|Bullish Engulfing| 'Bullish Engulfing'|
|Bearish Engulfing| 'Bearish Engulfing'|
|Evening Star Reversal| 'Evening Star'|
|Morning Star Reversal| 'Morning Star'|
|Tweezer Top| 'Tweezer Top'|
|Tweezer Bottom| 'Tweezer Bottom'|

\* Number of consecutive candles of the same color (positive and negative values for bullish and bearish candles respectively)
\** All indicators after this point represent candlestick patterns. Represented using a boolean array. For multi-candle patterns, set to true for candles where they complete.

**Plotting candlestick charts with candlesticks.py:**

Plots one day of data. Candlesticks containing prices and volume bars are automatically plotted.

1) Import candlsticks.py
2) Pass the parameters to create instance of CandleGraph:
    - Symbol name.
    - Data for one symbol: aapl_dict variable from above.
    - Timeframe.
    - Any index found in the day of interest.
    - Trade information: to show entry, exit, stoploss values/bars (optional)
3) Call the following methods to plot indicators:

|| Instance Methods | |
|------|-----|--------|
|plot_EMA(n)| plot_RSI()| plot_MACD()|
|plot_VWAP()| plot_bollinger()| plot_pattern()|
