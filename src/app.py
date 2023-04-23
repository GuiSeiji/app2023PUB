import dash
from dash import Dash, html, dcc, Input, Output,dash_table,State
import dash_bootstrap_components as dbc
import pandas_ta as ta
import dash_daq as daq
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import pandas_ta as ta
import warnings
import datetime as dt
import requests
import mplfinance
from stockstats import StockDataFrame as sdf


listofstocks = pd.read_csv('listaClear.csv')

listofstocks = listofstocks.drop(['Unnamed: 0','Name','Country','Sector'],axis=1)
acoes = ['TSLA','BAC','SCHW','SOFI','AMD','NIO','AMZN','CS','CCL','META']

def get_RETURN(df):                        
    retorno = df['Close'].pct_change() * 100

    trend = 0 * retorno
    trend[ retorno > 0.2 ] = 1
    trend[ retorno < -0.2] = -1

    return pd.DataFrame({'return': retorno.round(2), 'return_t': trend})

def get_SMA(df, window):                        
    sma = df['Close'].rolling(window).mean()

    trend = 0 * sma
    trend[ df['Close'] > sma] = 1
    trend[ df['Close'] < sma] = -1
    
    return pd.DataFrame({f'sma_{window}': sma.round(2), f'sma_{window}_t': trend})

def get_EMA(df, window):
    close = df['Close']
    ema = df['Close'].ewm(span=window, adjust=False).mean()

    trend = 0 * ema
    trend[ close > ema] = 1
    trend[ close < ema] = -1
    
    return pd.DataFrame({f'ema_{window}': ema.round(2), f'ema_{window}_t': trend})

def get_RSI(df, window = 14):
    sma = df['Close'].rolling(window).mean()
    rsi = df.ta.rsi(length=window)

    trend = 0 * sma
    trend[rsi < 30] = 1
    trend[rsi > 70] = -1


    return pd.DataFrame({'rsi': rsi.round(2), 'rsi_t': trend})

def get_BOL(df, window = 20, n_std = 2):
    # Cálculo das bandas de Bollinger
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * n_std)
    lower_band = rolling_mean - (rolling_std * n_std)

    # Identificação da tendência
    """
    trend = []
    for i in range(len(df)):
        if df['Close'][i] > upper_band[i]:
            trend.append(1)  # Tendência de alta
        elif df['Close'][i] < lower_band[i]:
            trend.append(-1)  # Tendência de baixa
        else:
            trend.append(0)  # Sem tendência definida
    """

    trend = rolling_mean * 0
    trend[df['Close'] > upper_band] = 1
    trend[df['Close'] > upper_band] = -1
    trend[df['Close'] == upper_band] = 0

    # Criação de um novo dataframe com a coluna "tendencia"
    df_trend = pd.DataFrame({ 'bol_t': trend})
    return df_trend

def get_MACD(df, fast=12, slow=26, signal=9):
    exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    trend = 0 * macd
    trend[macd > signal_line] = 1
    trend[macd < signal_line] = -1

    return pd.DataFrame({'macd': macd.round(2), 'macd_t': trend})

def get_STOC(df, janela=14, suavizacao=3, sobrevenda=20, sobrecompra=80):
    # Cálculo das linhas %K e %D
    high_n = df['High'].rolling(window=janela).max()
    low_n = df['Low'].rolling(window=janela).min()
    k_percent = 100 * ((df['Close'] - low_n) / (high_n - low_n))
    d_percent = k_percent.rolling(window=suavizacao).mean()
    
    # Identificação da tendência
    tendencia = []
    for i in range(len(df)):
        if k_percent.iloc[i] < sobrevenda and d_percent.iloc[i] < sobrevenda and k_percent.iloc[i] > d_percent.iloc[i]:
            tendencia.append(1)  # Tendência de alta
        elif k_percent.iloc[i] > sobrecompra and d_percent.iloc[i] > sobrecompra and k_percent.iloc[i] < d_percent.iloc[i]:
            tendencia.append(-1)  # Tendência de baixa
        else:
            tendencia.append(0)  # Tendência neutra
    
    return pd.DataFrame(tendencia, index=df.index, columns=['stoc_t'])


def get_CHAI(df):
    money_flow_multiplier = 2 * ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    money_flow_volume = money_flow_multiplier * df['Volume']
    adl = money_flow_volume.cumsum()
    chaikin = pd.DataFrame({'chai': adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()})
    chaikin['chai_t'] = chaikin['chai'].apply(lambda x: 1 if x > 0 else -1)

    return chaikin

def get_ADX(data, adx_period=14, adx_threshold=25):
    
    df = data.copy()
    stock_df = sdf.retype(df)
    df['pdi'] = stock_df['pdi']
    df['adx'] = stock_df['adx']
    df['mdi'] = stock_df['mdi']
    
    adx = df['adx']
    pdi = df['pdi']
    mdi = df['mdi']


    df.loc[(adx > adx_threshold) & (pdi > mdi), 'adx_t'] = 1
    df.loc[(adx > adx_threshold) & (pdi < mdi), 'adx_t'] = -1
    df.loc[adx <= adx_threshold, 'adx_t'] = 0
    
    return df[['adx',  'adx_t']]

def create_stock_data_dict(acoes,periodo):
    
    df_dict = {}
    
    for symbol in acoes:
        data = yf.Ticker(symbol)
        data = data.history(period = periodo)
        ema10_df = get_EMA(data, 10)
        sma10_df = get_SMA(data, 10)
        ema5_df = get_EMA(data, 5)
        sma5_df = get_SMA(data, 5)
        stoc_df = get_STOC(data)
        bol_df = get_BOL(data)
        adx_df = get_ADX(data)
        chai_df = get_CHAI(data)
        rsi_df = get_RSI(data, 14)
        macd_df = get_MACD(data)
        return_df = get_RETURN(data)
        data = data.join([rsi_df, sma10_df, ema10_df, sma5_df, ema5_df, macd_df, return_df, stoc_df, bol_df, adx_df, chai_df])
        data.dropna(inplace=True)
        data['sma_5_t'] = data['sma_5_t'].astype(int)
        data['ema_5_t'] = data['ema_5_t'].astype(int)
        data['sma_10_t'] = data['sma_10_t'].astype(int)
        data['ema_10_t'] = data['ema_10_t'].astype(int)
        data['chai_t'] = data['chai_t'].astype(int)
        data['stoc_t'] = data['stoc_t'].astype(int)
        data['bol_t'] = data['bol_t'].astype(int)
        data['adx_t'] = data['adx_t'].astype(int)
        data['macd_t'] = data['macd_t'].astype(int)
        data['rsi_t'] = data['rsi_t'].astype(int)
        data['return_t'] = data['return_t'].astype(int)
        table = (data.iloc[-1]).copy()
        table['symbol']= symbol
        df_dict[symbol] = table
        print(symbol)
    return df_dict


df_dict = create_stock_data_dict(acoes,periodo='1y')

df = pd.concat(df_dict.values(), axis=1).T.reset_index(drop=True)
trend = ['rsi_t','sma_10_t','ema_10_t','sma_5_t','ema_5_t','macd_t','return_t','adx_t','chai_t','stoc_t','bol_t']
value = df.drop([coluna for coluna in df.columns if coluna not in trend], axis=1)
value['Soma'] = value.sum(axis=1)
df = df.join(value["Soma"])

lista_1 = ['rsi','sma_10','ema_10','sma_5','ema_5','macd','return','adx','chai']
lista_2 = ['rsi_t','sma_10_t','ema_10_t','sma_5_t','ema_5_t','macd_t','return_t','adx_t','chai_t']

df['Soma'] = pd.to_numeric(df['Soma'], errors='coerce')

id_max = df.loc[df['Soma'].idxmax(),'symbol']
id_min = df.loc[df['Soma'].idxmin(),'symbol']

colunas_manter = ['Close','Volume','rsi','sma_10','ema_10','sma_5','ema_5','macd','return','adx','chai','symbol']

options =[{'label':'1 ano','value':'1y'},
         {'label':'2 anos','value':'2y'},
         {'label':'5 anos','value':'5y'},
         {'label':'10 anos','value':'10y'}]


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.MORPH])

server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.H1("Portifolio",
                style={"textAlign": "center",'padding-top':'10px'}), width=12)),
    html.Hr(),
    dbc.Row([dbc.Col(dcc.Dropdown(
        id = 'dropdown_update', options=options, value='1y'),width = 10),
           dbc.Col(html.Button('Atualizar',id='update_button'),width=2)]),
    html.Hr(),
    dbc.Row((dbc.Col(
        daq.Gauge(
            color={"gradient":True,"ranges":{"red":[-11,2],"yellow":[2,8],"green":[8,11]}},
            max = 11,
            min = -11,
            value = df['Soma'].max(),
            label = 'Maior tendência - '+id_max),

      width = 6),
            dbc.Col(
            daq.Gauge(
            color={"gradient":True,"ranges":{"red":[-11,2],"yellow":[2,8],"green":[8,11]}},
            max = 11,
            min = -11,
            value = df['Soma'].min(),
            label = 'Menor Tendência - '+id_min),
            width = 6),
            )),
    html.Hr(),
    dbc.Row(dbc.Col(
          dash_table.DataTable(id='dale',
            columns=[
            {"name": i, "id": i, "deletable": False, "selectable": True, "hideable": False}
            for i in colunas_manter],
            data=df.to_dict('records'),
            style_cell={'textAlign':'center','padding':'5px','backgoundColor':'white','width':'180px'},
            style_table={'overflowX':'auto'},
            style_data={'color': 'black','backgroundColor': 'white'},
            style_data_conditional=(
                [
                {'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(197, 213, 234)'},]
                +
                    
                [{'if':
                     {'filter_query':'{'+lista_2[i]+'}'+'= -1',
                      'column_id':lista_1[i]},
                 'color':'red'
                 } for i in range(9)]
                +
                [{'if':
                     {'filter_query':'{'+lista_2[i]+'}'+'= 1',
                      'column_id':lista_1[i]},
                 'color':'green'
                }for i in range(9)]
                
                 
            ),
            style_header={'backgroundColor': 'white','fontWeight': 'bold','color':'black'},

            ))),
    html.Hr(),
    dbc.Row([dbc.Col(
        dcc.Dropdown(
        id='drop',
        options= [{'label': valor, 'value': valor} for valor in listofstocks['Symbol']]),
        
    width=10),
            dbc.Col(
    html.Button('adicionar',id='bo'),width=2)
            ]),
    html.Hr(),
 
],style = {'background-color':'white'})


def update_stock_table(periodo):
    df = create_stock_data_dict(acoes,periodo=periodo)
    df = pd.concat(df.values(), axis=1).T.reset_index(drop=True)
    trend = ['rsi_t','sma_10_t','ema_10_t','sma_5_t','ema_5_t','macd_t','return_t','adx_t','chai_t','stoc_t','bol_t']
    value = df.drop([coluna for coluna in df.columns if coluna not in trend], axis=1)
    value['Soma'] = value.sum(axis=1)
    df = df.join(value["Soma"])

    lista_1 = ['rsi','sma_10','ema_10','sma_5','ema_5','macd','return','adx','chai']
    lista_2 = ['rsi_t','sma_10_t','ema_10_t','sma_5_t','ema_5_t','macd_t','return_t','adx_t','chai_t']

    id_max = df.loc[df['Soma'].idxmax(),'symbol']
    id_min = df.loc[df['Soma'].idxmin(),'symbol']
    
    return df.to_dict('records')


@app.callback(Output('dale','data'),Input('update_button','n_clicks'),State('dropdown_update','value'))
def update_table(n_clicks, periodo):
    if n_clicks is not None:
        df = update_stock_table(periodo)
        return df



@app.callback(Output('dale', 'data'), [Input('bo', 'n_clicks')], [State('drop', 'value'), State('dale', 'data'),State('acoes', 'data')])

def add_row(n_clicks, value, data):
    
    if value is not None and value not in acoes:
        acoes.append(value)
        
    df = create_stock_data_dict(acoes,periodo='10y')
    
    return df.to_dict('records')

    
if __name__=='__main__':
    app.run_server(debug=False)