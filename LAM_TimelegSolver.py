### version.23061521

class Stock:
    def __init__(self, ticker, market):
        self.ticker = ticker
        self.market = market
        self.df = self.df()
    
    def df(self):
        file_ticker = self.ticker.replace('/', '_') if '/' in self.ticker else self.ticker 
        folder_path = f'./datasets-{self.market}/'
        self.df = pd.read_csv(folder_path+ f'dataset-{self.market}-{file_ticker}.csv') 
        return self.df

class K200stock(Stock):
    def __init__(self, ticker):
        super().__init__(ticker, market='KOSPI200')

class R3000stock(Stock):
    def __init__(self, ticker):
        super().__init__(ticker, market='RUSSELL3000')


import pandas as pd
import numpy as np
from scipy.stats import pearsonr 
import matplotlib.pyplot as plt

class Pair:
    def __init__(self, Stock_target, Stock_source):
        self.target = Stock_target
        self.source = Stock_source 
        self.df = self.df()
        self.period = 75 # default: period = 75
        self.lag = 0
        self.lead = 0
        self.shift = 0
        self.date = None
        self.returns_t = None
        self.returns_s = None
        self.date_target = self.df.iloc[-self.period]['date']
        self.date_source = self.date_target
        self.df_result = pd.DataFrame()
        self.df_rollover = pd.DataFrame()
        self.params = self.record_params()
        
    def df(self, option='return'):
        r_t = self.target.df[['date', 'CHG_PCT_1D']]
        r_s = self.source.df[['date', 'CHG_PCT_1D']]
        df_merge = r_t.merge(r_s, on='date', how='outer')
        df_merge = df_merge.sort_values('date').fillna(0)
        df_merge = df_merge.reset_index(drop=True)
        self.df = df_merge.rename(columns={'CHG_PCT_1D_x': 'CHG_PCT_1D_t', 'CHG_PCT_1D_y': 'CHG_PCT_1D_s'})
        return self.df

    def set_date(self, date=None):
        self.date = date
        self.record_params()                                        
        return self.apply()
    
    def set_period(self, period=75):
        self.period = period
        self.record_params()                                
        return self.apply()
    
    def set_lag(self, lag=0):
        self.lag = lag
        self.record_params()                        
        return self.apply()

    def set_lead(self, lead=0):
        self.lead = lead
        self.record_params()                
        return self.apply()
    
    def set_shift(self, shift):
        self.shift = shift
        self.record_params()        
        return self.apply()
    
    def set_params(self, date, period, lag, lead, shift):
        self.date = date
        self.period = period
        self.lag = lag
        self.lead = lead
        self.shift = shift
        self.record_params()
        return self.apply()
    
    def reset(self):
        self.period = 75 # default: period = 75
        self.lag = 0
        self.lead = 0
        self.shift = 0
        self.date = None
        self.returns_t = None
        self.returns_s = None
        self.date_target = self.df.iloc[-self.period]['date']
        self.date_source = self.date_target
        self.df_result = pd.DataFrame()
        self.params = self.record_params()
        return self.apply()

    def record_params(self):
        self.params = {'date': self.date, 'period': self.period, 'lag': self.lag, 'lead': self.lead, 'shift': self.shift, 'date_target': self.date_target, 'date_source': self.date_source }
        return self.params
    
    def apply(self):    
        if self.date == None:
            df_ref = self.df         
            index_ref_f = df_ref.index[-1]
            index_ref_i = index_ref_f - self.period
        else:
            df_ref = self.df[self.df['date'] >= self.date]
            index_ref_i = df_ref.index[0]
            index_ref_f = index_ref_i + self.period    
            if len(df_ref) < self.period:
                raise ValueError("(parameter error) date is too close.")
        
        index_target_i = index_ref_i + self.shift + self.lead 
        index_source_i = index_ref_i + self.shift - self.lag
        index_target_f = index_target_i + self.period
        index_source_f = index_source_i + self.period

        if index_target_f > self.df.index[-1]:
            index_target_f = -1
            
        self.returns_target = self.df[['date', 'CHG_PCT_1D_t']].iloc[index_target_i: index_target_f]
        if index_target_f == -1:
            index_source_f = index_source_i + len(self.returns_target)
        self.returns_source = self.df[['date', 'CHG_PCT_1D_s']].iloc[index_source_i: index_source_f]        
        self.date_target = self.returns_target.iloc[0]['date']
        self.date_source = self.returns_source.iloc[0]['date']
        self.record_params()
        return self
    
    def corr(self, correction=False, limit=False):
        if correction == False:
            upper_bound = 10 if limit == True else None
            lower_bound = -10 if limit == True else None
            self._corr, self.p_value = pearsonr(self.returns_target['CHG_PCT_1D_t'].clip(lower=lower_bound, upper=upper_bound), self.returns_source['CHG_PCT_1D_s'].clip(lower=lower_bound, upper=upper_bound))
        else:
            list_rs_t = self.returns_target['CHG_PCT_1D_t'].tolist()
            list_rs_s = self.returns_source['CHG_PCT_1D_s'].tolist()

            merged_list_rs = pd.DataFrame({'rs_t': list_rs_t, 'rs_s': list_rs_s})
            merged_list_rs['delta_t_times_delta_s'] = merged_list_rs['rs_t'] * merged_list_rs['rs_s']
            max_influence_idx = merged_list_rs['delta_t_times_delta_s'].idxmax()
            merged_list_excl = merged_list_rs[merged_list_rs.index != max_influence_idx]

            corr_tot, p_value_tot = pearsonr(merged_list_rs['rs_t'], merged_list_rs['rs_s'])
            corr_excl, p_value_excl = pearsonr(merged_list_excl['rs_t'], merged_list_excl['rs_s'])

            self._corr = np.sqrt(abs(corr_tot * corr_excl)) * np.sign(max(corr_tot, corr_excl)) 
            self.p_value = np.sqrt(p_value_tot * p_value_excl)
        return {'target': self.target.ticker, 'source': self.source.ticker, 'lag': self.lag, 'shift': self.shift, 'corr': self._corr, 'p_value': self.p_value, 'date_target': self.date_target, 'date_source': self.date_source, 'period': self.period}

    def result(self, correction=False, limit=False):
        bucket = []
        for i in range(self.period):
            dictCorr = self.set_lag(i).corr(correction, limit)
            bucket.append(dictCorr)
        self.df_result = pd.DataFrame(bucket)
        self.set_lag(0)
        return self.df_result
    
    def solution(self, rank=1, key=abs, correction=False, limit=False):
        self.result(correction, limit)
        sorted_df = self.df_result.sort_values(by='corr', key=key, ascending=False)
        self.rank_info = sorted_df.iloc[rank-1].to_dict()        
        self.lag = 0
        return self.rank_info

    def rollover(self, roll_range, date=None, period=75, bidirect=True):
        pair = self.set_date(date).set_period(period)   
        j = -roll_range
        bucket = []
        self.lag = self.solution()['lag']
        for i in range(roll_range+1):
            bucket.append({'shift': i+j, 'corr': pair.set_shift(i+j).corr()['corr']})
        
        if bidirect == True:
            bucket_future = []
            for k in range(roll_range):
                bucket_future.append({'shift': k+1, 'corr': pair.set_shift(k+1).corr()['corr']})
            bucket = bucket + bucket_future

        self.df_rollover = pd.DataFrame(bucket)
        self.reset()
        
        return self.df_rollover
    
    def rollover_plot(self, roll_range, date=None, period=75, bidirect=True):

        df_rollover = self.rollover(roll_range, date, period, bidirect)
        shift_values = df_rollover['shift'].tolist()
        corr_values = df_rollover['corr'].tolist()
        best_lag = self.solution()['lag']
        self.reset()
        
        plt.figure(figsize=(12, 6))
        plt.plot(shift_values, corr_values, marker='o')
        
        plt.axhline(y=corr_values[shift_values.index(0)], color='r', linestyle='--')
        plt.xlabel('shift (day)')
        plt.ylabel('corr')
        plt.title(f'Rollover: lag {best_lag}, backward {roll_range}, period {period}, source({self.source.ticker}) -> target({self.target.ticker})')
        
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_position(('data', 0))
        plt.grid(axis='x')
        plt.show()
 
    def price_plot(self, option="price"):
        if option == "price":
            df_merge = self.target.df[['date', 'PX_LAST']].merge(self.source.df[['date', 'PX_LAST']], on='date', how='outer')
            title = 'Price'
        if option == "ma":
            df_merge = self.target.df[['date', 'MOV_AVG_5D']].merge(self.source.df[['date', 'MOV_AVG_5D']], on='date', how='outer')
            title = 'MA(5)'
        df_merge.sort_values(by='date', inplace=True)
        df_merge.fillna(method='ffill', inplace=True)
        df_merge.set_index('date', inplace=True)

        fig, ax1 = plt.subplots(figsize=(10,6))      
        ax1.set_ylabel('price_target (WON)')
        ax1.plot(df_merge.index, df_merge.iloc[:, 0], color='C0')
        ax2 = ax1.twinx()
        ax2.set_ylabel('price_source ($)')
        ax2.plot(df_merge.index, df_merge.iloc[:, 1], color='C1')
        ax1.legend(['target stock'], loc='upper left')
        ax2.legend(['source stock'], loc='upper right')

        num_ticks = 10
        x_ticks = np.linspace(0, len(df_merge)-1, num_ticks, dtype=np.int64)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(df_merge.index[x_ticks], rotation=45, ha='right')

        plt.title(f'{title}: target: {self.target.ticker}, source: {self.source.ticker}')
        plt.show()
    
    def compare_plot(self, best=False, info=False):

        if best == True:
            solution = self.solution()            
            best_corr = round(solution['corr'], 3)
            best_lag = solution['lag']
            self.set_lag(best_lag)
            title_header = f'Corr = {best_corr}'
        elif info == True:
            title_header = f'Corr = {corr}, shfit = {self.shift}'
        else:
            corr = round(self.corr()['corr'], 3)
            title_header = f'Corr = {corr}'
            
        df_compare = pd.DataFrame()
        df_compare['target'] = [0] + self.returns_target['CHG_PCT_1D_t'].tolist()
        df_compare['source'] = [0] + self.returns_source['CHG_PCT_1D_s'].tolist()
        df_compare.reset_index(drop=True)
        df_compare['normed_price_target'] = (1 + df_compare['target'] / 100).cumprod()
        df_compare['normed_price_source'] = (1 + df_compare['source'] / 100).cumprod()
        self.df_compare = df_compare
        
        plt.plot(df_compare.index, df_compare['normed_price_target'], label='target')
        plt.plot(df_compare.index, df_compare['normed_price_source'], label=f'lagged {self.lag} source')
        plt.xlabel(f'day (period: {self.period})')
        plt.ylabel('normed price')
              
        plt.title(f'{title_header}: {self.target.ticker} vs. lagged {self.lag} {self.source.ticker}')
        plt.xticks(np.arange(0, len(df_compare.index), 5))
        plt.legend()
        plt.grid(axis='x')
        plt.show()
   
    def stat_plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        # 첫 번째 플롯 그리기
        axes[0].plot(self.df_result['lag'], self.df_result['corr'], color='red')
        axes[0].set_xlabel('time lag')
        axes[0].set_ylabel('correlation')
        axes[0].set_title(f'Corr by lag: source({self.source.ticker}) -> target({self.target.ticker})')

        max_corr = self.df_result['corr'].max()
        min_corr = self.df_result['corr'].min()
        axes[0].annotate(f'Max: {max_corr:.2f}', xy=(self.df_result['lag'].iloc[self.df_result['corr'].idxmax()], max_corr),
                         xytext=(5, 0), textcoords='offset points', color='red')
        axes[0].annotate(f'Min: {min_corr:.2f}', xy=(self.df_result['lag'].iloc[self.df_result['corr'].idxmin()], min_corr),
                         xytext=(5, -10), textcoords='offset points', color='blue')

        mean_corr = self.df_result['corr'].mean()
        axes[0].axhline(mean_corr, color='red', linestyle='dashed', linewidth=0.5)
        axes[0].annotate(f'Mean: {mean_corr:.2f}', xy=(0, mean_corr),
                     xytext=(5, 0), textcoords='offset points', color='gray')
        
        # 두 번째 플롯 그리기
        axes[1].plot(self.df_result['lag'], self.df_result['p_value'], color='gray')
        axes[1].set_xlabel('time lag')
        axes[1].set_ylabel('p-value')
        axes[1].set_title(f'p-value by lag: source({self.source.ticker}) -> target({self.target.ticker})')

        mean_corr = self.df_result['p_value'].mean()
        axes[1].axhline(mean_corr, color='gray', linestyle='dashed', linewidth=0.5)
        axes[1].annotate(f'Mean: {mean_corr:.2f}', xy=(0, mean_corr),
                     xytext=(5, 0), textcoords='offset points', color='gray')
        
        max_p = self.df_result['p_value'].max()
        min_p = self.df_result['p_value'].min()
        axes[1].annotate(f'Min: {min_p:.2f}', xy=(self.df_result['lag'].iloc[self.df_result['p_value'].idxmin()], min_p),
                         xytext=(5, -10), textcoords='offset points', color='blue')

        plt.tight_layout()
        plt.show()


import random

class Market:
    def __init__(self, marketName):
        self.marketName = marketName
        self.df = self.df()
        self.tickers = self.df['value'].tolist()

    def df(self):
        folder_path = f'./dataset-members/'
        df = pd.read_csv(folder_path+ f'dataset-{self.marketName}-members.csv', ) 
        return df
    
    def member(self, index=None):
        n = len(self.tickers)
        if index == None:
            index = random.randint(0, n-1)
        return self.tickers[index]


import time

def intermarket_timelag(list_tickers_target, list_tickers_source, correction=False, limit=False):
    start_time = time.time()
    print(f"start: time lag solutions ...")
    
    n = len(list_tickers_target)*len(list_tickers_source)
    i = 1
    bucket = []
    for ticker_target in list_tickers_target:
        for ticker_source in list_tickers_source:
            print(f"-- ({i}/{n}) pair: (target: {ticker_target}, source: {ticker_source}) ...")
            target = K200stock(ticker_target)
            source = R3000stock(ticker_source)
            pair = Pair(target, source)
            solution = pair.solution(correction, limit)
            bucket.append(solution)
            i += 1

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"end: {execution_time} seconds elapsed")

    result = pd.DataFrame(bucket)
    return result


def getRankedPair(df, rank=1):
    i = rank - 1 
    t = K200stock(df.loc[i, 'target'])
    s = R3000stock(df.loc[i, 'source'])
    pair = Pair(t, s)
    return pair


def saveResult(df_result, subname):
    df_result.to_csv(f'total-result-batch-{subname}.csv', index=False)
    print(f'saved batch: {subname}')
    print('')    


def openRankedResult(fileName):
    df_result = pd.read_csv(fileName)
    df_result = df_result.sort_values(by='corr', key=abs, ascending=False)
    df_result = df_result.reset_index(drop=True)
    return df_result


import time 

class InterMarket:
    def __init__(self, target_market, source_market):
        self.targets = target_market
        self.sources = source_market 
    
    def solve_timelag(self, batch_unit = 200):
        start_time = time.time()
        print(f"start: time lag solutions ...")

        n = len(self.sources.tickers)
        i = 1
        bucket = []
        for ticker_source in self.sources.tickers:
            ticker_target = self.sources.tickers[i-1]
            print(f"-- ({i}/{n}) pair: (target: {tickers}, source: {ticker_source}) ...")
            target = K200stock(ticker_target)
            source = R3000stock(ticker_source)
            pair = Pair(target, source)
            solution = pair.solution(correction, limit)
            bucket.append(solution)
            i += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"end: {round(execution_time, 2)} seconds elapsed")

        result = pd.DataFrame(bucket)
        return result


import os
import pandas as pd

def mergeResultFiles(folder_path, output_file):
    file_list = os.listdir(folder_path)
    
    dataframes = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        _, file_ext = os.path.splitext(file_name)
        if file_ext.lower() == '.csv':
            df = pd.read_csv(file_path)            
            dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)    
    merged_df.to_csv(output_file, index=False)    
    return merged_df

    