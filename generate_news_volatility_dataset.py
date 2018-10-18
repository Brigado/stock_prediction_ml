import pandas as pd
import os
from dateutil import parser
from datetime import timezone, datetime, time
import datetime

volatility_path_dir = "Stock_is_volatile/"

df_news_date = pd.read_csv("news2.csv")
df_news_symbol = pd.read_csv("symbol-news-pairs.csv")

#get dataframes for all companys volatility
volatility_dfs = {}
for company_file in os.listdir(volatility_path_dir):
    company_symbol = company_file[12:-4]
    volatility_dfs.update({company_symbol:pd.read_csv(volatility_path_dir + company_file)})

df_dates = pd.read_csv(volatility_path_dir + "is_volatile_A.csv")['Date']
market_open_dates = df_dates.values
early_closing_dates = [datetime.date(2010, 11, 26), datetime.date(2011, 11, 25), datetime.date(2012, 11, 23), datetime.date(2013, 11, 29), datetime.date(2014, 11, 28), datetime.date(2015, 11, 27), datetime.date(2016, 11, 25), datetime.date(2017, 11, 24)]
for year in range(2010, 2019):
    early_closing_dates.append(datetime.date(year, 12, 24))
    early_closing_dates.append(datetime.date(year, 7, 3))

print("finished reading data")

def get_next_open(aim_date):
    #filter out weekends and holidays - no stock data given
    while str(aim_date) not in market_open_dates:
        try:
            aim_date += datetime.timedelta(days=1)
            if aim_date > datetime.date(2018, 8, 30):
                return -1

        except Exception:
            print(aim_date)

    return aim_date

#to which market day refers given news date
#Monday - Thursday: next day
#Friday-Sonday: all map to Monday
#holidays: map to next work day
#market time: 9:30 am (Market Open) - 4 pm (Market Close)
#relevant for day: last close to this days open
def calc_market_day(news_date_time):
    market_open_utc = time(9+4, 30, 0)
    market_close_utc = time(16+4, 0, 0)
    news_date_utc = parser.parse(news_date_time)
    news_date = news_date_utc.date()
    news_time = news_date_utc.time()

    if (news_date > datetime.date(2018, 8, 30)): #we only have stock data till them
        return -1

    if news_date in early_closing_dates:
        market_close_utc = time(13+4, 0, 0)

    next_open = 0
    if str(news_date) not in market_open_dates:
        next_open = get_next_open(news_date)

    elif market_open_utc <= news_time and news_time <= market_close_utc:
        return -1
    elif news_time < market_open_utc:
        #for this day
        next_open = get_next_open(news_date)
    elif(market_close_utc < news_time):
        #for next day
        news_date += datetime.timedelta(days=1)
        next_open = get_next_open(news_date)

    return next_open

lst = []
#iterate through all news data we have for our companies
for index, row in df_news_symbol.iterrows():
    if (index % 10000 == 0):
        print(index)
        if(index%20000 == 0):
            df = pd.DataFrame(lst, columns=['market_date', 'company_symbol', 'news_id', 'headline', 'is_volatile'])
            df.to_csv("news_to_volatility_dataset_{}.csv".format(index))
    #get Date
    news_id = row["news_id"]
    news_date = df_news_date.loc[df_news_date['id'] == news_id, 'timestamp'].values[0]
    market_day = calc_market_day(news_date)

    #only append if in relevant time
    if market_day != -1:

        #get volatility
        company_symbol = row['company_symbol']
        df_volatility = volatility_dfs[company_symbol]
        try:
            volatility = df_volatility.loc[df_volatility['Date'] == str(market_day), 'volatile'].values[0]
        except Exception:
            print("exception:{}".format(str(market_day)))
            continue


        lst.append([market_day, company_symbol, news_id, row['headline'], volatility])

df = pd.DataFrame(lst, columns=['market_date', 'company_symbol', 'news_id', 'headline', 'is_volatile'])
df.to_csv("news_to_volatility_dataset.csv")