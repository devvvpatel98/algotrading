import operator
from math import ceil,floor
import pandas as pd
import scipy as scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from hmmlearn import hmm
from nltk.sentiment import SentimentAnalyzer
from QuantConnect.Data.Custom.Tiingo import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
from AlgorithmImports import *
from numpy import unravel_index
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from functools import reduce
from scipy import stats
 
class FinalStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetSecurityInitializer(lambda x: x.SetMarketPrice(self.GetLastKnownPrice(x)))
        self.UniverseSettings.Resolution = Resolution.Hour
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)
        spy = self.AddEquity("SPY", Resolution.Hour)
 
        self.start_date = datetime(2023, 1, 1,0,0)
        self.end_date = datetime(2023, 4, 1 ,0,0)
        self.SetStartDate(self.start_date.year, self.start_date.month, self.start_date.day )  
        self.SetEndDate(self.end_date.year, self.end_date.month, self.end_date.day)
        self.SetCash(10_000_000)
        self.SetTimeZone(TimeZones.NewYork)
        self.numberOfSymbols = 365
 
        ################ HMM Specific ##########################    
        self.switch = 'neutral'
        self.SetWarmup(10)
 
        #have we just switched?
        self.curr = 'g'
        self.changed = True
 
        #sentiment analysis lookback days
        self.days = 4
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose("SPY"), self.MarketClose)
       
        self.daily_return = 0
        self.prev_value = self.Portfolio.TotalPortfolioValue
        # Fama French Model
        self.symbols = [ spy.Symbol ]
        self.winsorize = 10
        self.num_fine = 50
 
        #rebalance
        # HMM stocks dictionary
        self.hmm_dict = {}
        self.recent_prices = {}
 
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketOpen("SPY"), Action(self.rebalance))
       
        # Growth Multifactor Model
 
        self.num_portfolios = 6
        #################### Pair trading ####################################
 
   
        # run the clustering thing first time at start date
        self.Schedule.On(self.DateRules.On(self.start_date.year, self.start_date.month, self.start_date.day ),
                         self.TimeRules.At(9, 45), Action(self.GetCluster))
        self.Schedule.On(self.DateRules.On(self.start_date.year, self.start_date.month, self.start_date.day ),
                         self.TimeRules.At(9, 45), Action(self.DefineWieghts))
        # run the clustering thing every week and rebalance every friday @4 pm
        self.Schedule.On(self.DateRules.WeekEnd(),self.TimeRules.At(16, 0), self.RebalancePair)
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(1, 0), Action(self.GetCluster))
        self.Schedule.On(self.DateRules.WeekStart(), self.TimeRules.At(1, 0), Action(self.DefineWieghts))
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(10, 0), Action(self.TradePairs))
 
        #Parameters for pairs
        #Slope length is the number of days to look back to get the slope
        self.slope_length = 60
        self.min_sample = 4
        self.lookback = 365
        self.entry = 1.5
        self.exit = 0
       
        self.stop = 2.5
        self.PairInvested = None
        self.PairShortSpread = None
        self.PairLongSpread = None
        self.StockInvestedQty= None
        self.pair_info = None
        self.test_flag= False
        ##############################################################
 
    def CoarseSelectionFunction(self, coarse):
        CoarseWithFundamental = [x for x in coarse if x.HasFundamentalData and (float(x.Price)>10)]
        sortedByDollarVolume = sorted(CoarseWithFundamental, key=lambda x: x.DollarVolume, reverse=True)
        top = sortedByDollarVolume[:self.numberOfSymbols]
        return [i.Symbol for i in top]
       
    def FineSelectionFunction(self, fine):
        self.Debug('Universe selection time:' + str(self.Time))
 
        MarketCapFilter = [x for x in fine if float(x.MarketCap)> 250_000_000]
        sortedByMarketCap = sorted(MarketCapFilter, key=lambda c: c.MarketCap, reverse=True)
        filteredFine = [i.Symbol for i in sortedByMarketCap]
 
        self.filter_fine = filteredFine[:self.numberOfSymbols]    
 

        filtered_fine = [x for x in fine if x.OperationRatios.OperationMargin.Value
                                        and x.ValuationRatios.PriceChange1M
                                        and x.ValuationRatios.BookValuePerShare]
     
        # rank stocks by three factor.
        sortedByfactor1 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.PriceChange1M, reverse=True)
        sortedByfactor2 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValueYield, reverse=True)
        sortedByfactor3 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValuePerShare, reverse=False)
        stock_dict = {}
        # assign a score to each stock (ranking process)
        for i,ele in enumerate(sortedByfactor1):
            rank1 = i
            rank2 = sortedByfactor2.index(ele)
            rank3 = sortedByfactor3.index(ele)
            score = sum([rank1*0.2,rank2*0.4,rank3*0.4])
            stock_dict[ele] = score
        # sort the stocks by their scores
        self.sorted_stock = sorted(stock_dict.items(), key=lambda d:d[1],reverse=False)
        sorted_symbol = [x[0] for x in self.sorted_stock]
        # sort the top stocks into the long_list and the bottom ones into the short_list
        self.french_long = [x.Symbol for x in sorted_symbol[:self.num_fine]]
        self.french_short = [x.Symbol for x in sorted_symbol[-self.num_fine:]]
 
        for i in self.french_long:
            if i in self.french_short:
                self.french_long.remove(i)
                self.french_short.remove(i)
 
        #FINE FILTERING FOR GROWTH STOCKS
        filtered_fine = [x for x in fine if x.EarningReports.TotalDividendPerShare.ThreeMonths
                                        and x.ValuationRatios.PriceChange1M
                                        and x.ValuationRatios.BookValuePerShare
                                        and x.ValuationRatios.FCFYield]
        sortedByfactor1 = sorted(filtered_fine, key=lambda x: x.EarningReports.TotalDividendPerShare.ThreeMonths, reverse=True)
        sortedByfactor2 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.PriceChange1M, reverse=False)
        sortedByfactor3 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.BookValuePerShare, reverse=True)
        sortedByfactor4 = sorted(filtered_fine, key=lambda x: x.ValuationRatios.FCFYield, reverse=True)
        num_stocks = floor(len(filtered_fine)/self.num_portfolios)
        stock_dict = {}
        for i,ele in enumerate(sortedByfactor1):
            rank1 = i
            rank2 = sortedByfactor2.index(ele)
            rank3 = sortedByfactor3.index(ele)
            rank4 = sortedByfactor4.index(ele)
            score = [ceil(rank1/num_stocks),
                     ceil(rank2/num_stocks),
                     ceil(rank3/num_stocks),
                     ceil(rank4/num_stocks)]
            score = sum(score)
            stock_dict[ele] = score
        self.sorted_stock = sorted(stock_dict.items(), key=lambda d:d[1],reverse=True)
        sorted_symbol = [self.sorted_stock[i][0] for i in range(len(self.sorted_stock))]
        topFine = sorted_symbol[:self.num_fine]
        self.growth_long = [i.Symbol for i in topFine]
        for i in self.french_long:
            score = self.senti_stock(i)
            if score < 0.9:
                self.french_long.remove(i)
 
        for i in self.french_short:
            score = self.senti_stock(i)
            if (score > 0.998) and (score <1.9):
                self.french_short.remove(i)
   
        for i in self.growth_long:
            score = self.senti_stock(i)
            if score < 0.9:
                self.growth_long.remove(i)
        return self.growth_long+ self.filter_fine + self.french_long + self.french_short
 
    ####################################### pair trading strategy start ####################################
   
   
    def GetCluster(self):
       
        self.Debug('clustering:{}'.format(self.Time))
        #self.Debug(self.filter_fine)
        qb = self
        # looking back for 1 year  
        history = qb.History(self.filter_fine, self.lookback, Resolution.Daily)
        marketcap = []
        sector_code = []
        if history.empty:
            return
        self.df = history['close'].unstack(level =0)
        self.Debug(self.df.shape)
        df = self.df
        dg = df.apply(lambda x: x.pct_change(), axis=0)
        self.Debug('dg len')
        self.Debug(len(dg))
        #self.Debug('allcolumns:{}'.format(dg.shape))
        dg = dg.dropna()
        self.Debug(len(dg))
 
        N_PRIN_COMPONENTS = 5
 
        ### get the fundamentals based on the tickers we're taking
        for i in dg.columns:
            marketcap.append(self.Securities[i].Fundamentals.MarketCap)
            sector_code.append(self.Securities[i].Fundamentals.AssetClassification.MorningstarSectorCode)
 
        ### if we have more components we want, then we do PCA, else don't do
        if dg.shape[0] > N_PRIN_COMPONENTS:
            pca = PCA(n_components=N_PRIN_COMPONENTS)
            pca.fit(dg)
           # self.Log(pca.components_.T.shape)
           # self.Log(np.array(marketcap).reshape(len(marketcap), 1).shape)
       
            X = np.hstack(
                (pca.components_.T,
                np.array(marketcap).reshape(len(marketcap), 1),
                np.array(sector_code).reshape(len(sector_code), 1)
                ))
        else:
            X = np.hstack(
                (dg.T,
                np.array(marketcap).reshape(len(marketcap), 1),
                np.array(sector_code).reshape(len(sector_code), 1)
                ))
       
        ### divide by each sector match stock ticker to sector
        sector_dict = dict.fromkeys(set(sector_code), None)
        for uni_sector in set(sector_code):
            #print(uni_sector)
            ind = [i for i, n in enumerate(sector_code) if n == uni_sector]
            #print(len(ind))
            v =[dg.columns[i] for i in ind]
            #print(len(v))
            sector_dict[uni_sector] = v
       
        ### cluster within each sector
        sector_cluster_assignment = {}
        for sector in set(sector_code):
            #print(sector)
            a =  X[X[:, -1] == sector]
            a = preprocessing.StandardScaler().fit_transform(a)
            clustering = OPTICS(min_samples=self.min_sample).fit(a)
            labels = clustering.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            clustered_series_all = pd.Series(index=sector_dict[sector], data=labels.flatten())
            #clustered_series = pd.Series(index=dg.columns, data=labels.flatten())
            sector_cluster_assignment[sector] = clustered_series_all
            #print(n_clusters_)
            #self.Debug('sector:'+ str(sector) + 'cluster_number :' + str(n_clusters_))
       
        ## for each sector, each cluster, pick the one pair that meets both criteria
        sector_cluster_matrix = dict.fromkeys(set(sector_code), None)
        for sector,cluster_res in sector_cluster_assignment.items():
            #print(sector)
            #print(cluster_res.value_counts())
            cluster_dict = {}
            clustered_series = cluster_res[cluster_res != -1]
            #print(clustered_series)
            for i, which_clust in enumerate(clustered_series.value_counts().index):
                tickers = clustered_series[clustered_series == which_clust].index
                #print(tickers)
                score_matrix, pvalue_matrix, pairs, adf_matrix, adf_p_matrix, adf_pair = self.find_cointegrated_pairs(
                    df[tickers].dropna()
                )
                cluster_dict[which_clust] = {}
                sat_pair = list(set(pairs).intersection(adf_pair)) # pairs that satisifes both conditions
                if len(sat_pair)>1:
                    cluster_dict[which_clust]['final_pair'] = self.compare_pairs(sat_pair, tickers, pvalue_matrix,  adf_p_matrix)
                elif len(sat_pair) == 1:
                    cluster_dict[which_clust]['final_pair'] = sat_pair[0]
                else:
                    cluster_dict[which_clust]['final_pair'] = None
               
            sector_cluster_matrix[sector] = cluster_dict
       
        total_pairs = []
        for sector, sector_info in sector_cluster_matrix.items():
            for cluster, matrix in sector_info.items():
                #self.Debug(matrix['final_pair'])
                if matrix['final_pair'] is not None:
                    total_pairs.append(matrix['final_pair'] )
        self.final_pairs = total_pairs
        self.Debug(self.final_pairs)
        #pass
               
    def find_cointegrated_pairs(self, data, significance=0.03):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        adf_matrix = np.ones((n, n))
        adf_p_matrix = np.ones((n, n))
        keys = data.keys()
        #print(keys)
        pairs=[]
        adf_pairs= []
       
        for i in range(n):
            for j in range(i+1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                #self.Debug(len(S1))
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                model = sm.OLS(S1,S2)
                results = model.fit()
                res = results.resid
                #self.Debug(len(res))
                if len(res)>3:  
                    adf = adfuller(res)
                    adf_matrix[i, j] = adf[0]
                    adf_p_matrix[i, j] = adf[1]
                    if adf[1] < significance:
                        adf_pairs.append((keys[i], keys[j]))
                # if the cointegration test is significant, append the pair
                if pvalue < significance:
                    pairs.append((keys[i], keys[j]))
                # if the adf test is significant, append the pair
               
        return score_matrix, pvalue_matrix, pairs, adf_matrix, adf_p_matrix, adf_pairs
     
    def compare_pairs(self, pair_list, tick_order, p_val, adf_p_val):
        """
        If we have more than 1 pair that satisfy the condition, we compare them
        """
        p_val_total = []
        for i in pair_list:
            #print(i)
            ticker_ord = tick_order.tolist()
            row = ticker_ord.index(i[0])
            col = ticker_ord.index(i[1])
            #print(row, col)
            coint_p = p_val[row, col]
            adf_p =  adf_p_val[row, col]
            p_val_total.append(coint_p+adf_p)
        min_ind = np.array(p_val_total).argmin()
        return pair_list[min_ind]
 
    def DefineWieghts(self):
        self.Debug('weights:{}'.format(self.Time))
 
        lst_df= []
 
        vol_arr = np.empty((0,))
        pair_info =  {key: {} for key in self.final_pairs}
        for symbols in self.final_pairs:
            ticker0= str(symbols[0])
            ticker1= str(symbols[1])
            self.Debug('DF in we')
            df = self.df[[ticker0, ticker1]]
            self.Debug(df.shape)
            df = df.dropna()
            self.Debug(df.shape)
            Y = df.iloc[:self.slope_length][ticker0].apply(lambda x: math.log(x))
            X = df.iloc[:self.slope_length][ticker1].apply(lambda x: math.log(x))
            self.Debug(len(Y))
            X = sm.add_constant(X)
            model = sm.OLS(Y,X)
            results = model.fit()
            sigma = math.sqrt(results.mse_resid) # standard deviation of the residual
            #deviation of the residual
            slope = results.params[1]
            pair_info[symbols]['slope'] = slope
            res = results.resid #regression residual mean of res =0 by definition
            adf = adfuller (res)
            z_series = res/sigma
            dfpair = z_series.to_frame().rename(columns={0: "z-score"})
 
            #start calc for vol pair
            stddev = dfpair.std()
            wtp = 1/(1+slope)
            volatility  = stddev*(wtp**2)
            vol_arr = np.append(vol_arr, volatility)
            dfpair['return'] =0
       
            for i in range(1,dfpair.shape[0]):
                if (np.sign(dfpair['z-score'].iloc[i])) != (np.sign(dfpair['z-score'].iloc[i-1])):
                    dfpair['return'].iloc[i] = wtp*abs(dfpair['z-score'].iloc[i-1])
                else:
                    dfpair['return'].iloc[i] = wtp*(abs(dfpair['z-score'].iloc[i-1]) - abs(dfpair['z-score'].iloc[i]))
            lst_df.append(dfpair)
 
        risk_contribution = [1 / vol for vol in vol_arr]
        # calculate the sum of the inverse volatilities for all assets
        sum_risk_contribution = np.sum(risk_contribution)
        # calculate the weight of each asset
        weights = [rc / sum_risk_contribution for rc in risk_contribution]
        w_pair = dict(zip(self.final_pairs, weights))
        for (k,v), (k2,v2) in zip(pair_info.items(), w_pair.items()):
            pair_info[k]['pair_w'] = v2
        #self.Log(pair_info)
        #return pair_info
        self.pair_info = pair_info
        self.PairInvested = {key: False for key in self.final_pairs} # initiate all not invested at first
        self.PairShortSpread = {key: False for key in self.final_pairs}
        self.PairLongSpread = {key: False for key in self.final_pairs}
        self.StockInvestedQty = {key: 0 for key in list(reduce(operator.concat,self.final_pairs))}
       
 
    def RebalancePair(self):
        self.Debug('rebalancing:{}'.format(self.Time))
        if self.pair_info is not None:
            for pair, pair_stat in self.pair_info.items():
                if self.PairInvested[pair]:
                    self.MarketOrder(pair[0], -self.StockInvestedQty[pair[0]])
                    self.MarketOrder(pair[1], -self.StockInvestedQty[pair[1]])
        # reset everything
            self.PairInvested = None
            self.PairShortSpread = None
            self.PairLongSpread = None
            self.StockInvestedQty= None
   
    def GetStats(self, ticker):
        symbols = [self.Symbol(ticker[0]), self.Symbol(ticker[1])]
        df = self.History(symbols, self.lookback, Resolution.Daily)
        dg = df["close"].unstack(level=0)
        dg= dg.dropna()
        #self.Debug(self.dg)
       
        ticker0= str(symbols[0])
        ticker1= str(symbols[1])
        Y = dg[ticker0].apply(lambda x: math.log(x))
        X = dg[ticker1].apply(lambda x: math.log(x))
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        sigma = math.sqrt(results.mse_resid) # standard deviation of the residual
        res = results.resid #regression residual mean of res =0 by definition
        z_series = res/sigma
        latest_z = z_series[-1]
        return latest_z
   
    def OnData(self, data):
        if self.IsWarmingUp: return
        pass
 
       
    def TradePairs(self):
        if self.IsWarmingUp: return
        # if it is not weekend, we run this
        if self.Time.isoweekday() in range(1, 6):
            self.Debug('ondata{}'.format(self.Time))
            for pair, pair_stat in self.pair_info.items():
                price_p = self.Securities[pair[0]].Price
                price_q = self.Securities[pair[1]].Price
                z_score = self.GetStats(pair)
                self.Log('pair:{}, z_score:{}'.format(pair,z_score))
                self.Log('pair:{}, long:{}, short: {}, invested: {}'.format(pair, self.PairLongSpread[pair], self.PairShortSpread[pair] ,self.PairInvested[pair]))
               
                if price_p!= 0 :
                    qty_p = 0.5 * pair_stat['pair_w'] * (1/ (1+ pair_stat['slope'])) * self.Portfolio.TotalPortfolioValue / price_p
                else:
                    qty_p = 0
                    qty_q = 0
                if price_q!= 0:
                    qty_q = 0.5 * pair_stat['pair_w'] * (pair_stat['slope'] / (1+ pair_stat['slope'])) * self.Portfolio.TotalPortfolioValue / price_q
                else:
                    qty_q = 0
                    qty_p = 0
 
                if self.PairInvested[pair]:
                    if (self.PairShortSpread[pair]) and ((z_score <= self.exit) or (abs(z_score) > self.stop)):
                        self.MarketOrder(pair[0], -self.StockInvestedQty[pair[0]])
                        self.MarketOrder(pair[1], -self.StockInvestedQty[pair[1]])
                        self.Log("exit trade: sym1 " + str(pair[0]) + " /w "+ str(-self.StockInvestedQty[pair[0]]) + " sym2 " +str(pair[1]) + " /w "+str(-self.StockInvestedQty[pair[1]]))
                        self.PairInvested[pair] = False # reset
                        self.PairShortSpread[pair] = False # reset
                        self.StockInvestedQty[pair[0]]= 0 #reset
                        self.StockInvestedQty[pair[1]]= 0 # reset
                    if (self.PairLongSpread[pair]) and ((z_score >= self.exit) or (abs(z_score) > self.stop)):
                        self.MarketOrder(pair[0], -self.StockInvestedQty[pair[0]])
                        self.MarketOrder(pair[1],-self.StockInvestedQty[pair[1]])
                        self.Log("exit trade: sym1 " + str(pair[0]) + " /w "+ str(-self.StockInvestedQty[pair[0]]) + " sym2 " +str(pair[1]) + " /w "+str(-self.StockInvestedQty[pair[1]]))
                        self.PairInvested[pair] = False # reset
                        self.PairLongSpread[pair] = False # reset
                        self.StockInvestedQty[pair[0]]= 0  # reset
                        self.StockInvestedQty[pair[1]]= 0  # reset
 
                else:
                    if (abs(z_score) > self.entry) and (qty_p!=0 ) and( abs(z_score) < self.stop):
                        if z_score > 0:
                            self.PairInvested[pair] = True
                            self.PairShortSpread[pair] = True
                       
                            self.MarketOrder(pair[0], -qty_p)
                            self.MarketOrder(pair[1], qty_q)
                            self.StockInvestedQty[pair[0]] = -qty_p
                            self.StockInvestedQty[pair[1]] = qty_q
                        elif z_score <0:
                            self.PairInvested[pair] = True
                            self.PairLongSpread[pair] = True
                           
                            self.MarketOrder(pair[0], qty_p)
                            self.MarketOrder(pair[1], -qty_q)
                           
                            self.StockInvestedQty[pair[0]] = qty_p
                            self.StockInvestedQty[pair[1]] = -qty_q
                            self.Log("enter trade: sym1 " + str(pair[0]) + " /w "+ str(-np.sign(z_score) *qty_p) + " sym2 " +str(pair[1]) + " /w "+str(np.sign(z_score) * qty_q))
                     
    ####################################### pair trading strategy end ######################################
 
    ######################################## HMM strategy start #############################################
 
    def analyze_sentiment(self, data, ticker):
        # Initialize sentiment analyzer
        sid = SentimentIntensityAnalyzer()
 
        news_text = ' '.join([row['title'] + ' ' + row['description'] for _, row in data.iterrows()])
 
        # Convert to lowercase and split into words
        news_text = news_text.lower().split(" ")
 
        # Convert text to Unicode
        news_text = ' '.join(news_text)
        news_text = str(news_text.encode("utf-8"), "utf-8")
 
        # Perform sentiment analysis
        scores = sid.polarity_scores(news_text)
        #self.Debug(str(scores))
 
        # Extract compound sentiment score
        sentiment_score = scores['compound']
 
        #self.Debug(str(scores))
 
        #print(f"Sentiment analysis for {ticker}: {sentiment_score}")
 
        return sentiment_score
 
    def fetch_tiingo_news_data(self, ticker):
        self.tiingo_symbol = self.AddData(TiingoNews, ticker).Symbol
        #self.Debug(str(self.tiingo_symbol))
 
        # Historical data
        news_data = self.History(self.tiingo_symbol, self.days, Resolution.Daily)
        #self.Debug(f"We got {len(news_data)} items from our history request")
 
        return news_data
 
    #sentiment for each stock, called in individual factor model functions
    def senti_stock(self, ticker):
 
        data = self.fetch_tiingo_news_data(ticker)
        if ('title' not in data.iterrows()) or ('description' not in data.iterrows()) :
            return 2
        sentiment_score = self.analyze_sentiment(data, ticker)
 
        return sentiment_score
   
    def rebalance(self):
        self.Debug('HMM rebalancing {}'.format(self.Time))
 
        next = self.next = self.train()
 
        if self.Portfolio.TotalHoldingsValue == 0:
            self.hmm_dict = {}
            self.recent_prices = {}
            self.Debug("Total holdings value" + str(self.Portfolio.TotalPortfolioValue))
            self.switch = next
            if self.switch == 'bear':
                self.curr = 'f'
                self.FamaFrench()
            else:
                self.curr = 'g'
                self.GrowthModel()
            return
       
        self.switch = next
        if self.curr == 'g':
            if self.switch == 'bear':
 
                self.changed = True
 
                self.FamaFrench()
 
                self.curr = 'f'
                return
            else:
 
                self.changed = False
                self.GrowthModel()
 
                return
        else:
            #self.Debug('286 curr is f')
            if self.switch == 'bull':
 
                self.changed = True
                self.GrowthModel()
 
                self.curr = 'g'
                return
            else:
 
                self.changed = False
                self.FamaFrench()
 
                return
 
    def FamaFrench(self):
        l1 = [x for x in self.french_long if self.Securities.ContainsKey(x)]
        l2 = [x for x in self.french_short if self.Securities.ContainsKey(x)]
        q10 = .5*self.Portfolio.TotalPortfolioValue/len(l1) #positive value for new fama long
        q11 = -.5*self.Portfolio.TotalPortfolioValue/len(l2) #negative value for new fama short
 
 
        if self.changed == True:
            if len(self.hmm_dict.keys()) != 0:
                for i in self.hmm_dict.keys():
                    self.MarketOrder(i, -1 * self.hmm_dict[i] / self.recent_prices[i])
 
            self.hmm_dict = {}
            self.recent_prices = {}
            self.Debug("first fama run")
 
        else:
 
            hmm_dic_vals = self.hmm_dict.values() #dict values as a list
            hdv = [*set(hmm_dic_vals)] #dict values no duplicates (should have one >0 entry and one <0 entry only)
            q = sorted(hdv) #sorted so q=[<0, >0]
            q00 = q[1] #positive value (from old fama long)
            q01 = q[0] #negative value (from old fama short)
 
            dic_keys = list(self.hmm_dict.keys())
 
            for i in dic_keys: #for i in dict !!!!!!!
 
                price = self.Portfolio[i].Price
                last_price = self.recent_prices[i]
 
                if ((i not in l1) and (i not in l2)) or (price == 0): #if not in french long or short
 
                    if self.hmm_dict[i] > 0: #if quantity > 0
                        self.MarketOrder(i, math.floor(-q00 / last_price)) #get rid of it
 
                    else: #if quantity < 0
                        self.MarketOrder(i, math.floor(-q01 / last_price)) #get rid of it
 
                    del self.hmm_dict[i]
                    del self.recent_prices[i]
 
                elif (i not in l1) and (i in l2): # if not in long but is in short
 
                    if self.hmm_dict[i] > 0: #if originally in long list and now shorting
                        s = math.floor((q11/price) - (q00/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract long amount, add (negative) short amount
 
                        self.hmm_dict[i] = math.floor(q11)
                    else: #if originally in short list and still shorting
                        s = math.floor((q11/price) - (q01/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #get rid of old amount, add new (negativee) amount
 
                        self.hmm_dict[i] = math.floor(q11)
 
                    self.recent_prices[i] = price
 
                else: #if in long and not in short
 
                    if self.hmm_dict[i] > 0: #if was in long and still is long
                        s = math.floor((q10/price) - (q00/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract old long amount, add new long amount
 
                        self.hmm_dict[i] = math.floor(q10)
                    elif self.hmm_dict[i] < 0: #if was in short and now is long
                        s = math.floor((q10/price) - (q01/last_price))
                        if s != 0:
                            self.MarketOrder(i, s) #subtract old (negative) short amount, add new long amount
   
                        self.hmm_dict[i] = math.floor(q10)
               
                    self.recent_prices[i] = price
 
 
        for i in l1: #for i in long list
            price = self.Portfolio[i].Price
            if price == 0:
                continue
 
            elif i not in list(self.hmm_dict.keys()): #if not yet covered (not in dict)
                self.MarketOrder(i, math.floor(q10 / price)) #add new long amount
                self.hmm_dict[i] = math.floor(q10)
 
                self.recent_prices[i] = price
 
        for i in l2: #for i in short list
            price = self.Portfolio[i].Price
            if price == 0:
                continue
 
            elif i not in list(self.hmm_dict.keys()): #if not yet covered (not in dict)
                self.MarketOrder(i, math.floor(q11 / price)) #add new (negative) short amount
                self.hmm_dict[i] = math.floor(q11)
                #add price to recent_prices
                self.recent_prices[i] = price
 
    def GrowthModel(self):
        #self.Log("Growth Model")
        self.Log("Growth Model")
        self.Debug("before growth " + str(self.Portfolio.TotalPortfolioValue))
 
        #new MO values
        q1 = .5*self.Portfolio.TotalPortfolioValue/len(self.growth_long)
 
        l1 = [x for x in self.growth_long if self.Securities.ContainsKey(x)]
 
 
        if self.changed == False:
 
            q0 = .5*self.Portfolio.TotalPortfolioValue/len(self.hmm_dict)
 
            dic_keys = list(self.hmm_dict.keys())
 
            for i in dic_keys: #for i in dict
                price = self.Portfolio[i].Price
                last_price = self.recent_prices[i]
 
                if (i not in l1) or (price == 0): #if not in growth_long or price 0 (ie, we no longer want it in hmm strat)
                    self.MarketOrder(i, math.floor(-q0 / last_price)) #get rid of it
                    del self.hmm_dict[i]
                    del self.recent_prices[i]
                else: #if in growth long (ie, still want it)
                    s = math.floor((q1/price) - (q0/last_price))
                    if s != 0:
                        self.MarketOrder(i, s) #adjust quantity accordingly
                    self.hmm_dict[i] = math.floor(q1)
 
                    #after changing market orders and hmm_dict, put prices in recent_prices dict
                    self.recent_prices[i] = price
       
        else:
            if len(self.hmm_dict.keys()) != 0:
                for i in self.hmm_dict.keys():
                    self.MarketOrder(i, -1 * self.hmm_dict[i] / self.recent_prices[i])
            self.Debug("new regime reset port: " + str(self.Portfolio.TotalPortfolioValue))
            self.hmm_dict = {}
            self.recent_prices = {}
 
        for i in l1: #for i in growth_long
            price = self.Portfolio[i].Price
            if price == 0:
                return
            elif i not in list(self.hmm_dict.keys()): #if not in dict (ie, if new stock we want to include)
                self.MarketOrder(i, math.floor(q1 / price)) #add it in
                self.hmm_dict[i] = math.floor(q1)
                #add price to recent_prices
                self.recent_prices[i] = price
        self.Debug("after growth: " + str(self.Portfolio.TotalPortfolioValue))
 

    def MarketClose(self):
        self.daily_return = 100*((self.Portfolio.TotalPortfolioValue - self.prev_value)/self.prev_value)
        self.prev_value = self.Portfolio.TotalPortfolioValue
        self.Log(self.daily_return)
        self.Log("Switch: {}".format(self.switch))
        return
    def train(self):
       # Hidden Markov Model Modifiable Parameters
        hidden_states = 3
        em_iterations = 75
 
        history = self.History(self.symbols, 2600, Resolution.Daily)
        for symbol in self.symbols:
            if not history.empty:
                # get historical open price
                prices = list(history.loc[symbol.Value]['close'])
 
        Volatility = []
        # MA is the 10 day SMA
        MA = []
        # Return is the single-day percentage return
        Return = []
        ma_sum = 0;
        # Warming up data for moving average and volatility calculations
        for i in range (0, 10):
            Volatility.append(0);
            MA.append(0);
            Return.append(0);
            ma_sum += prices[i];
        # Filling in data for return, moving average, and volatility
        for i in range(0, len(prices)):
            if i >= 10:
                tail_close = prices[i-10]
                prev_close = prices[i-1]
                head_close = prices[i]
                ma_sum = (ma_sum - tail_close + head_close)
                ma_curr = ma_sum/10
                MA.append(ma_curr)
                Return.append(((head_close-prev_close)/prev_close)*100)
                #Computing Volatility
                vol_sum = 0
                for j in range (0, 10):
                    curr_vol = abs(ma_curr - prices[i-j]);
                    vol_sum += (curr_vol ** 2);
                Volatility.append(vol_sum/10);
        prices = prices[10:]
        Volatility = Volatility[10:]
        Return = Return[10:]
        # Creating the Hidden Markov Model
        model = hmm.GaussianHMM(n_components = hidden_states,
                                covariance_type="full", n_iter = em_iterations);
        obs = [];
        for i in range(0, len(Volatility)):
            arr = [];
            arr.append(Volatility[i]);
            arr.append(Return[i]);
            obs.append(arr);
        # Fitting the model and obtaining predictions
        model.fit(obs)
        predictions = model.predict(obs)
        # Regime Classification
        regime_vol = {};
        regime_ret = {};
        for i in range(0, hidden_states):
            regime_vol[i] = [];
            regime_ret[i] = [];
        for i in range(0, len(predictions)):
            regime_vol[predictions[i]].append(Volatility[i]);
            regime_ret[predictions[i]].append(Return[i]);
        vols = []
        rets = []
        today_regime = predictions[-1]
        for i in range(0, hidden_states):
            vol_dist = Distribution()
            vol_dist.Fit(regime_vol[i])
            vols.append(vol_dist.PDF(Volatility[-1]))
            ret_dist = Distribution()
            ret_dist.Fit(regime_ret[i])
            rets.append(ret_dist.PDF(Return[-1]))
        # > 0.5 Low-Pass Filter
        bear = -1
        bull = -1
        neg_return = 1
        pos_return = -1
        low_vol = 100
        for i in range(0, hidden_states):
            if sum(regime_ret[i]) / len(regime_ret[i]) < neg_return:
                neg_return = sum(regime_ret[i]) / len(regime_ret[i])
                bear = i
            if sum(regime_ret[i]) / len(regime_ret[i]) > pos_return:
                pos_return = sum(regime_ret[i]) / len(regime_ret[i])
                bull = i
        if vols[today_regime] / sum(vols) >= 0.3 and rets[today_regime] / sum(rets) >= 0.5:
            if bear == today_regime:
                return 'bear'
            else:
                return 'bull'
        else:
            return 'neutral'
 
    ######################################## HMM strategy end #############################################
 
# Kolmogorov-Smirnov Test to find best distribution
class Distribution(object):
    def __init__(self, dist_names_list = []):
        self.dist_names = ['norm','lognorm','expon', 'gamma',
                           'beta', 'rayleigh', 'norm', 'pareto']
        self.dist_results = []
        self.params = {}
        self.DistributionName = ""
        self.PValue = 0
        self.Param = None
        self.isFitted = False
 
    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)
            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name,p))
        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p
        self.isFitted = True
        return self.DistributionName, self.PValue
    def PDF(self, x):
        dist = getattr(scipy.stats, self.DistributionName)
        n = dist.pdf(x, *self.params[self.DistributionName])
        return n
