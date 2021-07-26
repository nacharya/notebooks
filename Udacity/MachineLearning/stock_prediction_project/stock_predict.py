#!/Users/nabin.acharya/anaconda/bin/python

from string import Template
import urllib
import zipfile
import StringIO
import os
import urllib2
import getopt
import sys

from pandas_datareader import data as pd_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dtt

from sklearn import svm, metrics, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn import cross_validation as CV

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# to turn off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DataGather:
    def __init__(self, ticker, start_date, end_date, dataDir="data"):
        self.ticker = ticker
        self.start_date = dtt.datetime.strptime(start_date, "%m-%d-%Y")
        self.end_date = dtt.datetime.strptime(end_date, "%m-%d-%Y")
        self.data_setup(dataDir)
        self.dataDir = dataDir + "/" + ticker
    def data_setup(self, dataDir):
        if not os.path.isdir(dataDir):
            os.mkdir(dataDir)
        if not os.path.isdir(dataDir + "/" + self.ticker):
            os.mkdir(dataDir + "/" + self.ticker)
        self.dataDir = dataDir + "/" + self.ticker
    def file_exists(self, url):
        try:
            ret = urllib2.urlopen(url)
            return ret.code == 200
        except:
            return False
    def get_retry(self, retry, url, dest):
        if not os.path.exists(dest):
            for i in range(0,retry):
                while True:
                    try:
                        urllib.urlretrieve(url, dest)
                    except IOError:
                        continue
                    break
            print("Received: " + dest + " from : " + url)
        else:
            print("Already have " + dest)
    def GetShortedData(self):
        s = Template('http://www.batstrading.com/market_data/shortsales/$year/$month/$fName-dl?mkt=bzx')
        delta = dtt.timedelta(days=1)
        d = self.start_date
        diff = 0
        count = 0
        weekend = set([5,6])
        while d <= self.end_date:
            file_name = 'BATSshvol%s.txt.zip' % d.strftime('%Y%m%d')
            dest = os.path.join(self.dataDir, file_name) 
            url = s.substitute(fName=file_name, year=d.year, month='%02d' % d.month)
            if self.file_exists(url):
                self.get_retry(5, url, dest)
            else:
                count+=1
            if d.weekday() not in weekend:
                diff += 1
            d += delta
    def ProcessShortDate(self, query_date):
        self.query_date = dtt.datetime.strptime(query_date, "%m-%d-%Y")
        file_name = 'BATSshvol%s.txt.zip' % self.query_date.strftime('%Y%m%d')
        dest = os.path.join(self.dataDir, file_name) 
        return self.readShortRatio(dest)[self.ticker]
    def readShortRatio(self,fName):
        zipped = zipfile.ZipFile(fName)
        lines = zipped.read(zipped.namelist()[0])
        buf = StringIO.StringIO(lines)
        df = pd.read_csv(buf,sep='|',index_col=1,parse_dates=False,dtype={'Date':object,'Short Volume':np.float32,'Total Volume':np.float32})
        ratio = df['Short Volume']/df['Total Volume']
        ratio.name = dtt.datetime.strptime(df['Date'][-1],'%Y%m%d')
        return ratio
    def ProcessShortedData(self, symbol):
        sr = []
        delta = dtt.timedelta(days=1)
        d = self.start_date
        diff = 0
        count = 0
        weekend = set([5,6])
        while d <= self.end_date:
            file_name = 'BATSshvol%s.txt.zip' % d.strftime('%Y%m%d')
            dest = os.path.join(self.dataDir, file_name) 
            lastValue = 0
            if os.path.exists(dest):
                try:
                    lastValue = self.readShortRatio(dest)[symbol]
                    sr.append(lastValue)
                except:
                    sr.append(lastValue)
            if d.weekday() not in weekend:
                diff += 1
            d += delta
        return sr
    def GetStockDataSheet(self, dataDir="data"):
        self.data_setup(dataDir)
        data_source = 'yahoo'
        fileName = "data/" + self.ticker + "/" + self.ticker + "-stock-data.csv"
        tkr = [ self.ticker ]
        if not os.path.exists(fileName):
            panel_data = pd_data.DataReader(tkr, data_source, self.start_date, self.end_date)
            adj_close = panel_data.ix['Adj Close']
            adj_close['Volumes'] = panel_data.ix['Volume']
            adj_close['Open'] = panel_data.ix['Open']
            adj_close['Close'] = panel_data.ix['Close']
            adj_close['Short_Ratio'] = self.ProcessShortedData(self.ticker)
            adj_close['Mov_avg'] = panel_data.ix['Close'].rolling(100, min_periods=1).mean()
            adj_close.to_csv(fileName)
            print("Finished Writing " + fileName)
        else:
            print("Already exists: " + fileName)

class DataProcessing:
    def __init__(self, dataGathered, dataDir="processed"):
        self.dataGathered = dataGathered
        self.data_setup(dataDir)
        self.dataDir = dataDir + "/" + dataGathered.ticker
    def data_setup(self, dataDir):
        if not os.path.isdir(dataDir):
            os.mkdir(dataDir)
        if not os.path.isdir(dataDir + "/" + self.dataGathered.ticker):
            os.mkdir(dataDir + "/" + self.dataGathered.ticker)
        self.dataDir = dataDir + "/" + self.dataGathered.ticker
    def data_read(self, percent=100.00):
        fileName = self.dataGathered.dataDir + "/" + self.dataGathered.ticker + "-stock-data.csv"
        print fileName
        try:
            self.dframe = pd.read_csv(fileName)
            if (percent < 100.0):
                rows = int(self.dframe.shape[0] * percent)
                self.dframe = pd.read_csv(fileName, nrows= rows)
        except IOError:
            print "Unable to read: ", fileName
            return None
        self.dframe.fillna(method='ffill', inplace=True)
        #print "Received ", self.dframe.shape[0], " samples, for ", self.dframe.shape[1], " features "
        #print "Features: ", list(self.dframe.columns)
        return self.dframe, self.dframe.shape[0]
    def show(self):
        x = pd.to_datetime(self.dframe['Date'])
        #y = self.dframe['Close']
        #y = self.dframe['Short_Ratio']
        y = self.dframe['Mov_avg']
        fig = plt.figure(figsize=(18,6))
        gr = fig.add_subplot(111)
        gr.plot(x,y,'b-o')
        #ticklabels= [num2date(i).strftime("%Y-%m-%d") for i in graph.get_xticks().tolist()]
        #graph.set_xticklabels(ticklabels)
        plt.show()
        plt.close()
    def data_process(self, query_date, bufferLength=6, metric="Close"):
        self.query_date = dtt.datetime.strptime(query_date, "%m-%d-%Y")
        self.endquerydelta = abs(np.busday_count(self.query_date, self.dataGathered.end_date))
        #print "Delta Num days: ", self.endquerydelta
        self.dframe['Date'] = pd.to_datetime(self.dframe['Date']) # convert date string to datetime type
        self.dframe['time_diff'] = (self.dframe['Date'] - self.dframe['Date'].min()) / np.timedelta64(1,'D')
        dataf = pd.DataFrame(index=self.dframe.index)
        for i in xrange(0, bufferLength + self.endquerydelta):
            dataf["Behind%s" % str(i + 1)] = self.dframe[metric]
        dataf['time_diff'] = self.dframe['time_diff']
        train_cols = ['time_diff']
        for i in xrange(0, bufferLength):
            train_cols.append("Behind%s" % str(i + 1))
        label_col = 'Behind' + str(bufferLength + self.endquerydelta)
        #print label_col
        dataf.dropna(inplace=True)
        row_data = dataf[train_cols]
        label_data = dataf[label_col]
        self.scaler = preprocessing.StandardScaler().fit(row_data)
        self.scaled_data = pd.DataFrame(self.scaler.transform(row_data))
        self.label_data = label_data
        self.final_row_unscaled = row_data.tail(1)
        #
        fileName = self.dataDir + "/" + self.dataGathered.ticker + "-stock-processed-dframe.csv"
        self.dframe.to_csv(fileName) 
    def debug(self, query_date, bufferLength=5):
        self.query_date = dtt.datetime.strptime(query_date, "%m-%d-%Y")
        self.dframe['Date'] = pd.to_datetime(self.dframe['Date'])
        self.dframe['time_diff'] = (self.dframe['Date'] - self.dframe['Date'].min()) / np.timedelta64(1,'D')
        self.endquerydelta = abs(np.busday_count(self.query_date, self.dataGathered.end_date))
        dataf = pd.DataFrame(index=self.dframe.index)
        for i in xrange(0, bufferLength + self.endquerydelta):
            dataf["Behind%s" % str(i + 1)] = self.dframe['Close']
        dataf['time_diff'] = self.dframe['time_diff']
        train_cols = ['time_diff']
        for i in xrange(0, bufferLength):
            train_cols.append("Behind%s" % str(i + 1))
        print train_cols
        label_col = 'Behind' + str(bufferLength + self.endquerydelta)
        print label_col + " " + str(self.endquerydelta) + " " + str(bufferLength)
    def bmark(self, predicted_val, test_val):
        #print "R^2 test score:", r2_score(predicted_val, test_val)
        RMSE = mean_squared_error(test_val, predicted_val)**0.5/100
        print "RMSE test score:", RMSE
        MAE = mean_absolute_error(test_val, predicted_val)/100
        print "MAE test score:", MAE
    def stock_linear_regression(self):
        lreg = LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label_data, test_size=0.25, random_state=42)
        parameters = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
        grid_obj = GridSearchCV(lreg, parameters, cv=None)
        grid_obj.fit(X_train, y_train)
        predict_train = grid_obj.predict(X_train)
        #print "train score:", r2_score(predict_train, y_train)
        predict_test = grid_obj.predict(X_test)
        #self.bmark(predict_test, y_test)
        self.grid_obj = grid_obj
    def stock_svm(self):
        clf = svm.SVR()
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data, self.label_data, test_size=0.30, random_state=42)
        #clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        parameters = {'C': [1, 10], 'epsilon': [0.1, 1e-2, 1e-3]}
        r2_scorer = metrics.make_scorer(metrics.r2_score)
        grid_obj = GridSearchCV(clf, param_grid=parameters, n_jobs=5, scoring=r2_scorer)
        grid_obj.fit(X_train, y_train)
        predict_train = grid_obj.predict(X_train)
        #print "best svr params", grid_obj.best_params_
        #print "train score:", r2_score(predict_train, y_train)
        predict_test = grid_obj.predict(X_test)
        #self.bmark(predict_test, y_test)
        self.grid_obj = grid_obj
    def stock_nn(self):
        X_train, X_test, y_train, y_test = train_test_split(self.scaled_data.as_matrix(), self.label_data.as_matrix(), test_size=0.25, random_state=42)
        model = Sequential()
        model.add(Dense(220, activation="relu", kernel_initializer="normal", input_dim=X_train.shape[1]))
        model.add(Dropout(0.15))
        model.add(Dense(1, activation="linear", kernel_initializer="normal"))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X_train, y_train, epochs=150, batch_size=25, verbose=0)
        loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
        #self.bmark(predict_test, y_test)
        #print "loss and metrics: ", loss_and_metrics
        self.grid_obj = model
    def train_test(self, ttype):
        if (ttype == "linear"):
            self.stock_linear_regression()
            inputSeq = self.scaler.transform(self.final_row_unscaled)
            inputSeq = pd.DataFrame(inputSeq)
            self.pvalue = self.grid_obj.predict(inputSeq)[0]
        elif (ttype == "svm"):
            self.stock_svm()
            inputSeq = self.scaler.transform(self.final_row_unscaled)
            inputSeq = pd.DataFrame(inputSeq)
            self.pvalue = self.grid_obj.predict(inputSeq)[0]
        elif (ttype == "nn"):
            self.stock_nn()
            inputSeq = self.scaler.transform(self.final_row_unscaled)
            self.pvalue = self.grid_obj.predict(inputSeq)[0][0]
    def predict(self):
        return self.pvalue


class BenchmarkTest:
    def __init__(self, dataProcessed, query_date, dframe, rownum):
        self.dataProcessed = dataProcessed
        self.query_date = dtt.datetime.strptime(query_date, "%m-%d-%Y")
        self.df = dframe
        self.total_rows = dframe.shape[0]
        self.rownum = rownum
    def process(self, reg_type):
        self.predict_val = []
        self.test_val = []
        self.sr =[]
        print "Row: ", self.rownum , " total: ", self.total_rows
        for i in range(self.rownum, self.total_rows):
            self.test_val.append(self.df.iloc[i]['Close'])
            qdate = dtt.datetime.strptime(self.df.iloc[i]['Date'],"%Y-%m-%d").strftime('%m-%d-%Y')
            self.dataProcessed.data_process(qdate)
            self.dataProcessed.train_test(reg_type)
            self.predict_val.append(self.dataProcessed.predict())
            self.sr.append(self.dataProcessed.dataGathered.ProcessShortDate(qdate))
    def show(self):
        # show the percent errors
        print "MAE : ", mean_absolute_error(self.test_val, self.predict_val, multioutput='uniform_average')/100
        print "RMSE : ", mean_squared_error(self.test_val, self.predict_val)**0.5/100

def usage():
    print sys.argv[0], " --help "
    print "\t" + "--gather/-g"
    print "\t" + "--summary/-s "
    print "\t" + "--benchmark/-b <svm/linear> <percent in float>"

def main():
    gather_mode = False
    summary_mode = False
    benchmark_mode = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:sgb", ["help", "summary", "gather", "benchmark" ])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    if len(opts) == 0:
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-s", "--summary"):
            summary_mode = True
            break
        if o in ("--gather", "-g"):
            gather_mode = True
            break
        if o in ("--benchmark", "-b"):
            benchmark_mode = True
            if len(sys.argv) != 4:
                usage()
                sys.exit(2)
            if sys.argv[2] not in ("svm", "linear"):
                usage()
                sys.exit(2)
            try:
                split_percent = float(sys.argv[3])
            except:
                usage()
                sys.exit(2)
            break
    q0 = { 'ticker' : 'GOOG', 'begin' : '1-1-2017', 'end' : '5-12-2017', 'query' : '6-12-2017' , 'related' : [ 'BIDU', 'MSFT', 'AAPL', 'FB' ] }
    q1 = { 'ticker' : 'AAPL', 'begin' : '1-1-2017', 'end' : '5-12-2017', 'query' : '6-12-2017' , 'related' : [ 'MSFT', 'GOOG', 'NVDA', 'ADBE' ] }
    q2 = { 'ticker' : 'AMZN', 'begin' : '1-1-2017', 'end' : '5-12-2017', 'query' : '6-12-2017' , 'related' : [ 'GOOGL', 'TWX', 'CBS', 'NFLX' ] }
    q3 = { 'ticker' : 'NFLX', 'begin' : '1-1-2017', 'end' : '5-12-2017', 'query' : '6-12-2017' , 'related' : [ 'AMZN', 'VIAB', 'FOXA', 'CBS' ] }

    queries = [ q0, q1, q2, q3 ]
    if gather_mode == True:
        for q in queries:
            dg = DataGather(q['ticker'], q['begin'], q['end'])
            dg.GetShortedData()
            dg.GetStockDataSheet()
        sys.exit(0)

    if benchmark_mode == True:
        for q in queries:
            dg = DataGather(q['ticker'], q['begin'], q['end'])
            dg.GetStockDataSheet()
            dp = DataProcessing(dg)
            df_full, rowcount = dp.data_read()
            if (df_full is not None):
                df_part, rownum = dp.data_read(percent = split_percent)
                bm = BenchmarkTest(dp, q['query'], df_full, rownum)
                bm.process(sys.argv[2])
                bm.show()
        sys.exit(0)
    if summary_mode == True:
        for q in queries:
            dg = DataGather(q['ticker'], q['begin'], q['end'])
            dg.GetStockDataSheet()
            dp = DataProcessing(dg)
            df, rownum= dp.data_read()
            if (df is not None):
                dp.data_process(q['query'])
                dp.train_test("linear")
                # dp.train_test("svm")
                print "stock ", q['ticker'], " predicted for ", q['query'], " is : ", dp.predict()
        sys.exit(0)

    usage()
if __name__ == '__main__':
    main()
