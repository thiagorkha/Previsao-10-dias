# Bibliotecas para manipulação de dados
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Bibliotecas para visualização de dados
import matplotlib.pyplot as plt
import streamlit as st

# Bibliotecas para pré-processamento e machine learning
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# Bibliotecas para deep learning
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l1_l2
from kerastuner.tuners import RandomSearch
import tensorflow as tf

# Bibliotecas para indicadores técnicos e seleção de features
import ta
from boruta import BorutaPy

# Configurações
warnings.filterwarnings("ignore")
def remove_outliers(df, column='Close'):
    """Remove outliers usando o método IQR."""
    df = df.droplevel(axis=1, level=1)
    df = df.dropna(subset=[column])
    if df[column].empty:
        raise ValueError(f"Erro: Não há dados suficientes na coluna '{column}' após remover valores NaN.")

    quantiles = df[column].quantile([0.25, 0.75]).dropna()
    if len(quantiles) < 2:
        raise ValueError("Erro: Dados insuficientes para calcular os quantis. Verifique se há valores NaN ou uma amostra muito pequena.")

    Q1, Q3 = quantiles.iloc[0], quantiles.iloc[1]
    IQR = Q3 - Q1
    return df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

def add_technical_indicators(data):
    """Adiciona indicadores técnicos ao DataFrame."""
    df = data.copy()
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = ta.trend.ema_indicator(df['MACD'], window=9)
    df['Close_diff'] = df['Close'].diff()
    df['Bollinger_Upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['Volume_Mean'] = df['Volume'].rolling(window=20).mean()
    df['ROC'] = ta.momentum.roc(df['Close'])
    df.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def build_lstm_model(hp, input_shape, forecast_horizon):
    """Constrói um modelo LSTM com Keras Tuner."""
    input_layer = Input(shape=input_shape)
    lstm_layer = Bidirectional(LSTM(units=hp.Int('lstm_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Dropout(hp.Float('dropout_lstm', 0.2, 0.5, step=0.1))(lstm_layer)
    gru_layer = Bidirectional(GRU(units=hp.Int('gru_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)
    gru_layer = BatchNormalization()(gru_layer)
    gru_layer = Dropout(hp.Float('dropout_gru', 0.2, 0.5, step=0.1))(gru_layer)
    dense_layer = Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(gru_layer)
    output_layer = Dense(forecast_horizon)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3])), loss=Huber(), metrics=['mae'])
    return model

def build_gru_model(hp, input_shape, forecast_horizon):
    """Constrói um modelo GRU com Keras Tuner."""
    input_layer = Input(shape=input_shape)
    gru_layer = Bidirectional(GRU(units=hp.Int('gru_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)
    gru_layer = BatchNormalization()(gru_layer)
    gru_layer = Dropout(hp.Float('dropout_gru', 0.2, 0.5, step=0.1))(gru_layer)
    dense_layer = Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(gru_layer)
    output_layer = Dense(forecast_horizon)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3])), loss=Huber(), metrics=['mae'])
    return model

def create_and_compile_model(hp, input_shape, forecast_horizon, model_type='lstm'):  # Função para criar e compilar o modelo
    if model_type == 'lstm':
        model = build_lstm_model(hp, input_shape, forecast_horizon)
    elif model_type == 'gru':
        model = build_gru_model(hp, input_shape, forecast_horizon)
    else:
        raise ValueError("Invalid model type. Choose 'lstm' or 'gru'.")
    return model

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula o MAPE (Mean Absolute Percentage Error)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape_per_step = []
    for i in range(y_pred.shape[1]):
        y_true_aligned = y_true[:, i]
        numerator = np.abs(y_true_aligned - y_pred[:, i])
        denominator = np.abs(y_true_aligned)
        non_zero_mask = denominator != 0
        safe_division = np.divide(numerator[non_zero_mask], denominator[non_zero_mask], out=np.zeros_like(numerator[non_zero_mask]), where=non_zero_mask)
        mape_per_step.append(np.mean(safe_division) * 100)
    return np.mean(mape_per_step)

def analyze_stock_with_lstm_and_strategy(stock_ticker, forecast_horizon=10, test_size=0.2, interval="1d"):
    """Função principal para análise de ações com LSTM e estratégia."""
    # Baixa os dados
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
    data = yf.download(stock_ticker, start=start_date, end=end_date, interval=interval)

    if data.empty:
        st.error(f"Erro: Nenhum dado encontrado para {stock_ticker}.")
        return

    # Remove outliers e adiciona indicadores técnicos
    data = remove_outliers(data)
    processed_data = add_technical_indicators(data)
    current_price = processed_data['Close'].iloc[-1]

    # Pré-processamento dos dados
    feature_columns = [col for col in processed_data.columns if col != 'Close']
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(processed_data[feature_columns])
    target_scaler = RobustScaler()
    scaled_target = target_scaler.fit_transform(processed_data[['Close']])

    # Seleção de features com Boruta
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
    boruta_selector.fit(scaled_features, scaled_target.ravel())
    selected_features_boruta = scaled_features[:, boruta_selector.support_]

    # Preparação dos dados para o modelo
    X, y = [], []
    lookback = 30
    for i in range(lookback, len(processed_data) - forecast_horizon + 1):
        X.append(selected_features_boruta[i - lookback:i])
        y.append(scaled_target[i:i + forecast_horizon].flatten())
    X, y = np.array(X), np.array(y)

    # Pipeline de pré-processamento
    pipeline = Pipeline([('scaler', RobustScaler())])
    X = pipeline.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape[0], lookback, -1)

    # Treinamento e avaliação do modelo
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {'lstm': build_lstm_model, 'gru': build_gru_model, 'rf': RandomForestRegressor(random_state=42)}
    best_models = {}
    best_losses = {}

    for model_name, model_builder in models.items():
        best_loss = float('inf')
        best_model_cv = None
        best_hp_cv = None

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            if model_name in ('lstm', 'gru'):
                def create_model(hp):
                    return create_and_compile_model(hp, (X_train_fold.shape[1], X_train_fold.shape[2]), forecast_horizon, model_type=model_name)

                tuner = RandomSearch(
                    create_model,
                    objective='val_loss',
                    max_trials=3,
                    executions_per_trial=1,
                    directory=f'stock_tuning_{model_name}_fold_{fold}',
                    project_name=f'stock_forecast_v2_{model_name}_fold_{fold}'
                )
                tuner.search(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), callbacks=[EarlyStopping(patience=3)], verbose=0)
                best_model_fold = tuner.get_best_models(num_models=1)[0]
                loss = best_model_fold.evaluate(X_val_fold, y_val_fold, verbose=0)[0]

                if loss < best_loss:
                    best_loss = loss
                    best_model_cv = best_model_fold

            else:  # Random Forest
                best_model_fold = model_builder()
                X_train_fold_reshaped = X_train_fold.reshape(X_train_fold.shape[0], -1)
                best_model_fold.fit(X_train_fold_reshaped, y_train_fold)
                X_val_fold_reshaped = X_val_fold.reshape(X_val_fold.shape[0], -1)
                loss = mean_squared_error(y_val_fold, best_model_fold.predict(X_val_fold_reshaped))

                if loss < best_loss:
                    best_loss = loss
                    best_model_cv = best_model_fold

        best_models[model_name] = best_model_cv
        best_losses[model_name] = best_loss

    # Previsões
    last_data = selected_features_boruta[-lookback:].reshape(1, lookback, -1)
    if isinstance(best_model_cv, RandomForestRegressor):
        last_data_reshaped = last_data.reshape(1, -1)
        forecast = best_model_cv.predict(last_data_reshaped).flatten()
    else:
        forecast = best_model_cv.predict(last_data).flatten()

    forecast = target_scaler.inverse_transform(forecast.reshape(1, -1)).flatten()

    # Exibe os resultados
    st.write(f"Preço atual do ativo: R$ {current_price:.2f}")
    st.write("Previsões para os próximos dias:")
    for i, price in enumerate(forecast, start=1):
        pct_change = ((price / current_price) - 1) * 100
        if pct_change > 2:
            signal = "[COMPRA]"
        elif pct_change < -2:
            signal = "[VENDA]"
        else:
            signal = "[NEUTRO]"
        st.write(f"{signal} Dia {i}: R$ {price:.2f} ({pct_change:+.2f}%)")

    # Gráfico de retorno acumulado
    returns = np.diff(processed_data['Close']) / processed_data['Close'][:-1]
    cumulative_returns = np.cumsum(returns)
    buy_and_hold = processed_data['Close'] / processed_data['Close'].iloc[0] - 1

    plt.figure(figsize=(12, 6))
    plt.plot(processed_data.index[1:], cumulative_returns, label="Retorno Acumulado", color='b')
    plt.plot(processed_data.index, buy_and_hold, label="Buy & Hold", linestyle='dashed', color='g')
    plt.title(f'Rendimento Acumulado vs Buy & Hold - {stock_ticker} ({interval})')
    plt.xlabel('Data')
    plt.ylabel('Retorno (%)')
    plt.legend()
    st.pyplot(plt)

# Interface Streamlit
st.title("Análise de Ações com LSTM e Estratégia")

# Entrada do usuário
stock_ticker = st.text_input("Digite o ticker da ação (ex: CSNA3.SA):", "CSNA3.SA")
forecast_horizon = st.slider("Horizonte de previsão (dias):", 1, 30, 10)
interval = st.selectbox("Intervalo dos dados:", ["1d", "1wk", "1mo"])

# Botão para executar a análise
if st.button("Analisar"):
    analyze_stock_with_lstm_and_strategy(stock_ticker, forecast_horizon, interval=interval)
