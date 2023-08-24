import streamlit as st
import yfinance as yf
from datetime import date
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

data_inicio = '2017-01-01'
data_final = date.today()  # .strftime('%y-%m-%d')

st.title('Analise de Ações')

# criando a sidebar

st.sidebar.header('Escolha o ativo')

n_dias = st.slider('Quantidade de dias de previsão', 30, 360)


def pegar_dados_acoes():
    arquivo = 'C:/Users/Micro/PycharmProjects/streamli_prophet/acoes.csv'
    return pd.read_csv(arquivo, delimiter=';')


df = pegar_dados_acoes()

acao = df['snome']

nome_acao_escolhida = st.sidebar.selectbox('Escolha uma ação: ', acao)

def_acao = df[df['snome'] == nome_acao_escolhida]
acao_escolhida = def_acao.iloc[0]['sigla_acao']

acao_escolhida = acao_escolhida + '.SA'


def pegar_valores_online(sigla_acao):
    df = yf.download(sigla_acao, data_inicio, data_final)
    df.reset_index(inplace=True)
    return df


df_valores = pegar_valores_online(acao_escolhida)

st.subheader('Tabela de valores- ' + nome_acao_escolhida)
st.write(df_valores.tail(10))

# cria graficos
st.subheader('Gráfico de preços')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_valores['Date'], y=df_valores['Close'], name='Preço Fechamento', line_color='yellow'))
fig.add_trace(go.Scatter(x=df_valores['Date'], y=df_valores['Open'], name='Preço Abertura', line_color='blue'))

st.plotly_chart(fig)

# previsão
df_treino = df_valores[['Date', 'Close']]

# renomear as colunas para o prophet
df_treino = df_treino.rename(columns={'Date': 'ds', 'Close': 'y'})

modelo = Prophet()
modelo.fit(df_treino)

futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
previsao = modelo.predict(futuro)

st.subheader('Previsão')
st.write(previsao[['ds', 'yhat_lower', 'yhat_upper']].tail(n_dias))

#grafico
grafico1 = plot_plotly(modelo, previsao)
st.plotly_chart(grafico1)

#grafico 2

grafico2 = plot_components_plotly(modelo,previsao)
st.plotly_chart(grafico2)



























