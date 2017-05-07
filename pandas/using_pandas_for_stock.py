import pandas as pd
from pandas_datareader import data, wb
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from matplotlib.finance import candlestick_ohlc, quotes_historical_yahoo_ohlc
import numpy as np

# 开始时间
start = datetime.datetime(2010, 1, 1)
# 结束时间
end = datetime.date.today()

# 读取苹果公司股票数据
apple = data.DataReader("AAPL", "yahoo", start, end)
# 查看类型，pandas.core.frame.DataFrame
type(apple) 
# 读取前几行数据, 每列包括开盘价、收盘价等
print(apple.head())
# 包括Open, High, Low, Close, Volume, Adj Close（调整后的价格）
# 绘制Adj Close的折线图
# apple["Adj Close"].plot(grid=True)
# plt.show()

# 在蜡烛图中，黑色蜡烛表示交易日当天收盘价高于开盘价（盈利）
# 而红色蜡烛表示交易日当天开盘价高于收盘价（亏损）
# 烛芯表示最高价与最低价，蜡烛体则表示开盘价与收盘价
def pandas_candlestick_ohlc(dat, stick="day", otherseries=None):
    # 定位在星期一的Locator，用于将大刻度major ticks设在星期一
    mondays = WeekdayLocator(MONDAY)
    # 定位在每日的Locator，用于将小刻度minor ticks设为每日
    alldays = DayLocator()
    # 设置大刻度的格式如Jan 1
    weekFormatter = DateFormatter('%b %d')
    # 设置小刻度的格式如1
    dayFormatter = DateFormatter('%d')

    # 通过DataFrame.loc按照行列索引抽取数据
    transdat = dat.loc[:, ["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                # 构造week属性，用isocalendar返回结果是三元组(年号,第几周,第几天)，再取周数
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1])
            elif stick == "month":
                # 构造month属性
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month)
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0])
            # 分组，用set去除相同的，再转回list类型
            grouped = transdat.groupby(list(set(["year", stick])))
            # 用于存放按年、月或星期整理出来的新数据
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []})
            for name, group in grouped:
                # name形如(2017, 1)，group是该分组下的具体数据
                # iloc是按索引抽取int型数据，group.iloc[-1,3]表示最后一行第4个int，group.Low表示Low属性
                # index[0]是该组第一行的index，这样每个组就只设一个
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 1], 
                    "High": max(group.High), 
                    "Low": min(group.Low), 
                    "Close": group.iloc[-1, 3]}, 
                    index=[group.index[0]]))
            
            if stick == "week":
                stick =5
            elif stick == "month":
                stick = 30
            elif stick == "year":
                stick = 365
    elif (type(stick) == int and stick >= 1):
        # 自定义天数的分组
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []})
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')
        # 设置大刻度Locator，将major ticks设在星期一
        ax.xaxis.set_major_locator(mondays)
        # 设置小刻度Locator，将minor ticks设在每日
        ax.xaxis.set_minor_locator(alldays)
    else:
        # 大于两年的，格式中增加年份
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
    # 显示网格
    ax.grid(True)

    # 绘制蜡烛图，quotes数据格式为(time, open, high, low, close, ...) 
    # quotes还可以直接从quotes_historical_yahoo_ohlc('AAPL', start, end)获得
    # zip接受任意多个序列作为参数，返回一个行为tuple的列表，列为每个序列。
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)

    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax=ax, lw=1.3, grid=True)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()

# 调用绘制蜡烛图的函数
# pandas_candlestick_ohlc(apple)

def pandas_simple_candlestick_ohlc():
    # 开始时间
    start = datetime.datetime(2017, 1, 1)
    # 结束时间
    end = datetime.date.today()

    # 沪市
    ticker_ss = '600000.ss' 
    # 深市
    ticker_sz = '000001.sz'
    # 港股 
    ticker_hk = '0700.hk'
    # 获取数据
    quotes = quotes_historical_yahoo_ohlc(ticker_sz, start, end)
    fig, ax= plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    # 设置主要刻度的显示格式
    weekFormatter = DateFormatter('%m-%d-%Y')
    ax.xaxis.set_major_formatter(weekFormatter)
    # 设置主要刻度Locator，将major ticks设在星期一
    mondays = WeekdayLocator(MONDAY)
    ax.xaxis.set_major_locator(mondays)
    # 设置次要刻度Locator，将minor ticks设在每日
    alldays = DayLocator()
    ax.xaxis.set_minor_locator(alldays)

    # 注意，ohlc代表o开盘价、h最高价、l最低价、c收盘价
    candlestick_ohlc(ax, quotes, width=0.6, colorup='r', colordown='g')

    ax.grid(True)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

# 调用简单的绘制蜡烛图的函数
# pandas_simple_candlestick_ohlc()

# 读取多家公司股票数据
microsoft = data.DataReader("MSFT", "yahoo", start, end)
google = data.DataReader("GOOG", "yahoo", start, end)
# 用df来操作
stocks = pd.DataFrame({"AAPL": apple["Adj Close"],
                      "MSFT": microsoft["Adj Close"],
                      "GOOG": google["Adj Close"]})
stocks.head()

stocks.plot(grid = True)
# 在绘制图表时使用两种不同的y轴尺度，因为不同股之间价格差异太大
stocks.plot(secondary_y = ["AAPL", "MSFT"], grid = True)

# 计算回报率 price[t] / price[0]
stock_return = stocks.apply(lambda x: x / x[0])
# axhline中h是指水平方向，绘制一条y=1的辅助线
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)

# 计算取对数后的差，apply作用于每列，shift表示整体往后移一位，最前面用nan补全
stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1)))
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)

# 计算移动均线，在rolling中设置滑动窗口，用round保留2位小数
apple["20d"] = np.round(apple["Close"].rolling(window = 20, center = False).mean(), 2)
apple["50d"] = np.round(apple["Close"].rolling(window = 50, center = False).mean(), 2)
apple["200d"] = np.round(apple["Close"].rolling(window = 200, center = False).mean(), 2)
pandas_candlestick_ohlc(apple.loc['2016-01-04':'2016-08-07',:], otherseries = ["20d", "50d", "200d"])

plt.show()





