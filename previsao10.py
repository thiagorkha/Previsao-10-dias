import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l1_l2
from kerastuner.tuners import RandomSearch
import warnings
import ta
from boruta import BorutaPy
from sklearn.pipeline import Pipeline
import logging
import tensorflow as tf
from tensorflow.keras.models import clone_model

warnings.filterwarnings("ignore")

def remove_outliers(df, column='Close'):
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
    df = data.copy()

    # Indicadores técnicos da biblioteca TA
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
    df['Volume_Mean'] = df['Volume'].rolling(window=20).mean() # Volume médio
    df['ROC'] = ta.momentum.roc(df['Close']) # Taxa de variação (ROC)

    df.drop(['EMA_12', 'EMA_26'], axis=1, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    return df

def build_lstm_model(hp, input_shape, forecast_horizon):
    # Regularização L1/L2 adicionada nas camadas LSTM, GRU e Dense
    input_layer = Input(shape=input_shape)
    lstm_layer = Bidirectional(LSTM(units=hp.Int('lstm_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)  # return_sequences=False (removido)
    lstm_layer = BatchNormalization()(lstm_layer)
    lstm_layer = Dropout(hp.Float('dropout_lstm', 0.2, 0.5, step=0.1))(lstm_layer)
    gru_layer = Bidirectional(GRU(units=hp.Int('gru_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)  # return_sequences=False (removido)
    gru_layer = BatchNormalization()(gru_layer)
    gru_layer = Dropout(hp.Float('dropout_gru', 0.2, 0.5, step=0.1))(gru_layer)
    dense_layer = Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(gru_layer)
    output_layer = Dense(forecast_horizon)(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-4, 1e-3])), loss=Huber(), metrics=['mae'])
    return model

def build_gru_model(hp, input_shape, forecast_horizon):
    input_layer = Input(shape=input_shape)
    gru_layer = Bidirectional(GRU(units=hp.Int('gru_units', 32, 128, step=32), kernel_regularizer=l1_l2(1e-5, 1e-4)))(input_layer)  # return_sequences=False (removido)
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
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mape_per_step = []
    for i in range(y_pred.shape[1]):  # Itera sobre os passos do horizonte
        # ***CORREÇÃO CRUCIAL: Alinhar y_true com y_pred para o passo atual***
        y_true_aligned = y_true[:, i]  # Seleciona os valores de y_true para o passo atual
        numerator = np.abs(y_true_aligned - y_pred[:, i])
        denominator = np.abs(y_true_aligned)
        non_zero_mask = denominator != 0
        safe_division = np.divide(numerator[non_zero_mask], denominator[non_zero_mask], out=np.zeros_like(numerator[non_zero_mask]), where=non_zero_mask)
        mape_per_step.append(np.mean(safe_division) * 100)

    return np.mean(mape_per_step)

def analyze_stock_with_lstm_and_strategy(stock_ticker, forecast_horizon=10, test_size=0.2, interval="1d"):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
    data = yf.download(stock_ticker, start=start_date, end=end_date, interval=interval)

    if data.empty:
        print(f"Erro: Nenhum dado encontrado para {stock_ticker}.")
        return

    data = remove_outliers(data)

    processed_data = add_technical_indicators(data)
    current_price = processed_data['Close'].iloc[-1]
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
    selected_feature_names = np.array(feature_columns)[boruta_selector.support_]

    # ***CRIAR DATAFRAME COM FEATURES SELECIONADAS NA ORDEM CORRETA***
    X_selected = pd.DataFrame(scaled_features[:, boruta_selector.support_], columns=np.array(feature_columns)[boruta_selector.support_])


    # Dados externos (IBOV e SP500)
    try:
        ibov = yf.download('^BVSP', start=start_date, end=end_date, interval=interval)['Close']
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval=interval)['Close']

        # Adiciona os dados externos como features
        processed_data['IBOV'] = ibov.reindex(processed_data.index)  # Alinha os índices
        processed_data['SP500'] = sp500.reindex(processed_data.index)  # Alinha os índices

        processed_data.dropna(inplace=True)  # Remove linhas com NaN após adicionar dados externos
        feature_columns = [col for col in processed_data.columns if col not in ['Close', 'IBOV', 'SP500']] # Atualiza as colunas de features
        scaled_features = scaler.fit_transform(processed_data[feature_columns]) # Re-escalar os dados com as novas features
    except Exception as e:
        print(f"Erro ao baixar dados externos: {e}")
        return

    X, y = [], []
    lookback = 30
    for i in range(lookback, len(processed_data) - forecast_horizon +1 ): # Correção: Adicionado +1 para alinhar os dados
        X.append(selected_features_boruta[i - lookback:i])
        y.append(scaled_target[i:i + forecast_horizon].flatten())
    X, y = np.array(X), np.array(y)

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
    ])

    X = pipeline.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape[0], lookback, -1)

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

            num_sequences_train = X_train_fold.shape[0]
            y_train_fold_reshaped = y_train_fold.reshape(num_sequences_train, forecast_horizon)

            num_sequences_val = X_val_fold.shape[0]
            y_val_fold_reshaped = y_val_fold.reshape(num_sequences_val, forecast_horizon)





            if model_name in ('lstm', 'gru'):
                def create_model(hp):
                    model = create_and_compile_model(hp, (X_train_fold.shape[1], X_train_fold.shape[2]), forecast_horizon, model_type=model_name)
                    return model

                # ***CALLBACK ModelCheckpoint***
                checkpoint_filepath = f'best_{model_name}_model_fold_{fold}.h5' # Caminho para salvar o melhor modelo
                checkpoint_callback = ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    monitor='val_loss',  # Monitora a perda de validação
                    save_best_only=True,  # Salva apenas o melhor modelo
                    save_weights_only=False, # Salva o modelo completo
                    mode='min',  # Busca a menor perda
                    verbose=0
                )

                tuner = RandomSearch(
                    create_model,
                    objective='val_loss',
                    max_trials=3,
                    executions_per_trial=1,
                    directory=f'stock_tuning_{model_name}_fold_{fold}',
                    project_name=f'stock_forecast_v2_{model_name}_fold_{fold}'
                )

                tuner.search(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), callbacks=[EarlyStopping(patience=3), checkpoint_callback], verbose=0)

                # ***CARREGAR O MELHOR MODELO SALVO***
                best_model_fold = tf.keras.models.load_model(checkpoint_filepath)



                # ***OBTER OS MELHORES HIPERPARÂMETROS***
                best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

                print("X_train_fold shape:", X_train_fold.shape)
                print("y_train_fold_reshaped shape:", y_train_fold_reshaped.shape)
                print("X_val_fold shape:", X_val_fold.shape)
                print("y_val_fold_reshaped shape:", y_val_fold_reshaped.shape)



                # ***TREINAR O MODELO FINAL COM OS MELHORES HIPERPARÂMETROS***
                best_model_fold = create_model(best_hp)  # Cria o modelo com os melhores HP
                best_model_fold.fit(X_train_fold, y_train_fold, epochs=10, validation_data=(X_val_fold, y_val_fold), callbacks=[EarlyStopping(patience=3)], verbose=0)

                loss = best_model_fold.evaluate(X_val_fold, y_val_fold, verbose=0)[0]

                if loss < best_loss:
                    best_loss = loss
                    best_model_cv = best_model_fold
                    best_hp_cv = best_hp  # Guarda os melhores hiperparâmetros

                tf.keras.backend.clear_session()

            else:  # Random Forest
                best_model_fold = model_builder
                X_train_fold_reshaped = X_train_fold.reshape(X_train_fold.shape[0], -1)
                best_model_fold.fit(X_train_fold_reshaped, y_train_fold)

                X_val_fold_reshaped = X_val_fold.reshape(X_val_fold.shape[0], -1)
                loss = mean_squared_error(y_val_fold, best_model_fold.predict(X_val_fold_reshaped))

                if loss < best_loss:
                    best_loss = loss
                    best_model_cv = best_model_fold  # Guarda o melhor modelo para esse fold

        best_models[model_name] = best_model_cv  # O melhor modelo para este *ticker*
        best_losses[model_name] = best_loss

        if model_name in ('lstm', 'gru'):
            # ***CORREÇÃO CRUCIAL: Usar X (dados originais após pipeline) para o reshape***
            X_reshaped = X.reshape(-1, lookback, X.shape[2])  # Usar X e não X_selected
            y_reshaped = y.reshape(-1, forecast_horizon)
            best_model_final = create_model(best_hp_cv)  # Cria o modelo com os melhores HP
            best_model_final.fit(X_reshaped, y_reshaped, epochs=10, callbacks=[EarlyStopping(patience=3)], verbose=0)
            best_models[model_name] = best_model_final  # O modelo final treinado no dataset completo


    # Previsões - reshape para Random Forest
    predictions = {}
    for model_name, model in best_models.items():
        if model_name in ('lstm', 'gru'):
            predictions[model_name] = model.predict(X).reshape(X.shape[0], forecast_horizon)
        else: # Random Forest
            X_reshaped = X.reshape(X.shape[0], -1) # Reshape dados para 2D
            predictions[model_name] = model.predict(X_reshaped).reshape(X.shape[0], forecast_horizon) # Previsões com dados remodelados


    last_data = selected_features_boruta[-lookback:].reshape(1, lookback, -1)  # Usando selected_features_boruta AQUI
    # Reshape para o Random Forest, se necessário
    if isinstance(best_model_cv, RandomForestRegressor):  # Verifica o tipo do modelo
        last_data_reshaped = last_data.reshape(1, -1)
        forecast = best_model_cv.predict(last_data_reshaped).flatten()
    else:  # LSTM ou GRU
        forecast = best_model_cv.predict(last_data).flatten()

    forecast = target_scaler.inverse_transform(forecast.reshape(1, -1)).flatten()

    # Ensemble (média ponderada)
    predictions = {}
    for model_name, model in best_models.items():
        if model_name in ('lstm', 'gru'):
            predictions[model_name] = model.predict(X).reshape(X.shape[0], forecast_horizon)
        else:  # Random Forest
            X_reshaped_ensemble = X.reshape(X.shape[0], -1)  # Reshape para 2D para o ensemble
            predictions[model_name] = model.predict(X_reshaped_ensemble).reshape(X.shape[0], forecast_horizon)  # Previsões com dados remodelados



    # Pesos para a média ponderada (exemplo: você pode otimizar esses pesos)
    weights = {'lstm': 0.4, 'gru': 0.3, 'rf': 0.3}
    ensemble_predictions = np.average([predictions[model_name] for model_name in predictions], axis=0, weights=[weights[model_name] for model_name in predictions])

    ensemble_predictions = target_scaler.inverse_transform(ensemble_predictions)
    y_original = target_scaler.inverse_transform(y)

    # Métricas de avaliação
    mape = mean_absolute_percentage_error(y_original, ensemble_predictions)
    r2 = r2_score(y_original, ensemble_predictions)

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
        st.write(f"{signal} Dia {i}: R$ {price:.2f} ({pct_change:+.2f}%)")  # Use st.write

    # Plotting with Streamlit
    fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axes
    ax.plot(processed_data.index[1:], cumulative_returns, label="Retorno Acumulado", color='b')
    ax.plot(processed_data.index, buy_and_hold, label="Buy & Hold", linestyle='dashed', color='g')
    ax.set_title(f'Rendimento Acumulado vs Buy & Hold - {stock_ticker} ({interval})')
    ax.set_xlabel('Data')
    ax.set_ylabel('Retorno (%)')
    ax.legend()
    st.pyplot(fig)  # Display the plot with st.pyplot

    y_pred = best_model_cv.predict(X.reshape(X.shape[0], -1)) if isinstance(best_model_cv, RandomForestRegressor) else best_model_cv.predict(X)  # Reshape X para 2D se for Random Forest
    y_pred = target_scaler.inverse_transform(y_pred).flatten()
    y_test_original = target_scaler.inverse_transform(y.reshape(y.shape[0], -1)).flatten() # Avaliando no conjunto completo

    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    mae = mean_absolute_error(y_test_original, y_pred)


    # Backtesting - reshape para Random Forest
    backtest_start = int(len(processed_data) * 0.8)
    backtest_data = processed_data[backtest_start:]

    all_predictions = []  # Lista para armazenar todas as previsões do backtesting
    all_real_values = [] # Lista para armazenar todos os valores reais do backtesting

    for i in range(lookback, len(backtest_data) - forecast_horizon):
        X_backtest = selected_features_boruta[backtest_start + i - lookback:backtest_start + i].reshape(1, lookback, -1)
        y_backtest = backtest_data['Close'][i:i + forecast_horizon].values

        if isinstance(best_model_cv, RandomForestRegressor):
            X_backtest_reshaped = X_backtest.reshape(1, -1)
            prediction = best_model_cv.predict(X_backtest_reshaped).flatten()
        else:
            prediction = best_model_cv.predict(X_backtest).flatten()

        prediction = target_scaler.inverse_transform(prediction.reshape(1, -1)).flatten()

        all_predictions.extend(prediction)  # Usando a nova lista
        all_real_values.extend(y_backtest)  # Usando a nova lista

    rmse_backtest = np.sqrt(mean_squared_error(all_real_values, all_predictions))
    mae_backtest = mean_absolute_error(all_real_values, all_predictions)

    st.write(f"=== Resultados Ensemble para {stock_ticker} ({interval}) ===")
    st.write(f"MAPE: {mape:.4f}")
    st.write(f"R²: {r2:.4f}")
    
    st.write(f"=== Resultados para {stock_ticker} ({interval}) ===")
    st.write(f"RMSE: {rmse:.2f} ({rmse/current_price*100:.2f}%)")
    st.write(f"MAE: {mae:.2f} ({mae/current_price*100:.2f}%)")
    
    st.write(f"=== Resultados Backtesting para {stock_ticker} ({interval}) ===")
    st.write(f"RMSE: {rmse_backtest:.2f} ({rmse_backtest/current_price*100:.2f}%)")
    st.write(f"MAE: {mae_backtest:.2f} ({mae_backtest/current_price*100:.2f}%)")
    

# Streamlit app
st.title("Stock Price Prediction with LSTM/GRU and Ensemble")

stock_ticker = st.text_input("Enter Stock Ticker (e.g., PETR4.SA)", "PETR4.SA")  # Default value
forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 10) # Slider for forecast horizon
interval = st.selectbox("Interval", ["1d", "1w", "1mo"], index=0)  # Selectbox for interval

if st.button("Analyze"):
    if stock_ticker:
        analyze_stock_with_lstm_and_strategy(stock_ticker, forecast_horizon, interval=interval)
    else:
        st.warning("Please enter a stock ticker.")
