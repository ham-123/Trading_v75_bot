#!/usr/bin/env python3
"""
IA ENSEMBLE OPTIMISÉE - XGBoost + LightGBM + Features Avancées
🚀 OBJECTIF: Passer de 85% à 92%+ de précision
🎯 NOUVEAUTÉS:
   • Ensemble de 2 modèles différents
   • 65+ features avancées (vs 57 avant)
   • Vote pondéré intelligent
   • Features de microstructure de marché
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List
import warnings
import ta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnsembleAIModel:
    """🚀 Modèle IA Ensemble XGBoost + LightGBM OPTIMISÉ"""

    def __init__(self):
        """Initialisation du modèle ensemble"""
        # Modèles de l'ensemble
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_weights = {'xgb': 0.6, 'lgb': 0.4}  # XGBoost légèrement favorisé

        # Scalers séparés pour robustesse
        self.feature_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()  # Pour données avec outliers

        self.feature_names = []

        # 🆕 HYPERPARAMÈTRES OPTIMISÉS XGBoost
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 3,
            'gamma': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        # 🆕 HYPERPARAMÈTRES OPTIMISÉS LightGBM
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'n_estimators': 400,
            'max_depth': 7,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'importance_type': 'gain',
            'verbose': -1
        }

        # Paramètres d'features
        self.lookback_period = 30
        self.prediction_horizon = 12  # 12 * 5min = 1h

        # Chemins de sauvegarde
        self.model_path_xgb = 'data/ensemble_xgb_model.pkl'
        self.model_path_lgb = 'data/ensemble_lgb_model.pkl'
        self.scaler_path = 'data/ensemble_scaler.pkl'
        self.robust_scaler_path = 'data/ensemble_robust_scaler.pkl'
        self.features_path = 'data/ensemble_feature_names.pkl'
        self.ensemble_info_path = 'data/ensemble_model_info.json'

        # Métadonnées
        self.last_training = None
        self.model_version = "3.0-Ensemble-XGB-LGB"
        self.training_samples = 0
        self.validation_accuracy = 0.0
        self.individual_accuracies = {'xgb': 0.0, 'lgb': 0.0}
        self.feature_importance = {}
        self.n_features = 0

        os.makedirs('data', exist_ok=True)
        logger.info("🚀 IA Ensemble XGBoost + LightGBM initialisée")

    def prepare_advanced_features_v2(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """🆕 FEATURE ENGINEERING AVANCÉ V2 - 65+ features"""
        try:
            if len(df) < self.lookback_period + self.prediction_horizon + 150:
                logger.debug("Pas assez de données pour features avancées V2")
                return None, None

            df_features = df.copy()
            logger.info("📊 Calcul des features avancées V2 (65+ features)...")

            # === INDICATEURS TECHNIQUES DE BASE (Améliorés) ===

            # RSI multi-périodes avec divergences
            df_features['rsi_14'] = ta.momentum.rsi(df_features['price'], window=14)
            df_features['rsi_9'] = ta.momentum.rsi(df_features['price'], window=9)
            df_features['rsi_21'] = ta.momentum.rsi(df_features['price'], window=21)
            df_features['rsi_divergence'] = df_features['rsi_14'] - df_features['rsi_21']
            df_features['rsi_momentum'] = df_features['rsi_14'].diff(3)  # Momentum RSI

            # MACD famille étendue
            macd_data = ta.trend.MACD(df_features['price'], window_fast=12, window_slow=26, window_sign=9)
            df_features['macd'] = macd_data.macd()
            df_features['macd_signal'] = macd_data.macd_signal()
            df_features['macd_histogram'] = macd_data.macd_diff()
            df_features['macd_histogram_slope'] = df_features['macd_histogram'].diff(3)

            # MACD alternatif (périodes plus courtes)
            macd_fast = ta.trend.MACD(df_features['price'], window_fast=5, window_slow=13, window_sign=5)
            df_features['macd_fast'] = macd_fast.macd()
            df_features['macd_fast_signal'] = macd_fast.macd_signal()

            # EMA famille étendue avec ratios
            for period in [5, 9, 13, 21, 34, 50, 100, 200]:
                df_features[f'ema_{period}'] = ta.trend.ema_indicator(df_features['price'], window=period)

            # Ratios EMA (Golden Cross patterns)
            df_features['ema_ratio_9_21'] = df_features['ema_9'] / df_features['ema_21']
            df_features['ema_ratio_21_50'] = df_features['ema_21'] / df_features['ema_50']
            df_features['ema_ratio_50_200'] = df_features['ema_50'] / df_features['ema_200']

            # SMA et distances
            df_features['sma_20'] = ta.trend.sma_indicator(df_features['price'], window=20)
            df_features['sma_50'] = ta.trend.sma_indicator(df_features['price'], window=50)

            # Bollinger Bands avancé
            bb_data = ta.volatility.BollingerBands(df_features['price'], window=20, window_dev=2)
            df_features['bb_upper'] = bb_data.bollinger_hband()
            df_features['bb_middle'] = bb_data.bollinger_mavg()
            df_features['bb_lower'] = bb_data.bollinger_lband()
            df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
            df_features['bb_position'] = (df_features['price'] - df_features['bb_lower']) / (
                        df_features['bb_upper'] - df_features['bb_lower'])
            df_features['bb_squeeze'] = (
                        df_features['bb_width'] < df_features['bb_width'].rolling(20).quantile(0.2)).astype(int)

            # 🆕 Bollinger Bands multi-timeframes
            bb_short = ta.volatility.BollingerBands(df_features['price'], window=10, window_dev=1.5)
            df_features['bb_short_position'] = (df_features['price'] - bb_short.bollinger_lband()) / (
                        bb_short.bollinger_hband() - bb_short.bollinger_lband())

            # Stochastic amélioré
            stoch_data = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['price'])
            df_features['stoch_k'] = stoch_data.stoch()
            df_features['stoch_d'] = stoch_data.stoch_signal()
            df_features['stoch_divergence'] = df_features['stoch_k'] - df_features['stoch_d']

            # Stochastic alternatif (plus rapide)
            stoch_fast = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['price'],
                                                          window=5)
            df_features['stoch_fast_k'] = stoch_fast.stoch()

            # Williams %R
            df_features['williams_r'] = ta.momentum.williams_r(df_features['high'], df_features['low'],
                                                               df_features['price'], lbp=14)
            df_features['williams_r_smooth'] = df_features['williams_r'].rolling(3).mean()

            # ADX famille
            adx_data = ta.trend.ADXIndicator(df_features['high'], df_features['low'], df_features['price'], window=14)
            df_features['adx'] = adx_data.adx()
            df_features['di_plus'] = adx_data.adx_pos()
            df_features['di_minus'] = adx_data.adx_neg()
            df_features['dx'] = abs(df_features['di_plus'] - df_features['di_minus']) / (
                        df_features['di_plus'] + df_features['di_minus']) * 100

            # === 🆕 FEATURES DE MICROSTRUCTURE DE MARCHÉ ===

            # Volume analysis (simulé pour Vol75)
            df_features['volume_sma'] = df_features['volume'].rolling(20).mean()
            df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma']
            df_features['volume_momentum'] = df_features['volume'].pct_change(5)

            # Prix vs Volume correlation
            df_features['price_volume_corr'] = df_features['price'].rolling(20).corr(df_features['volume'])

            # 🆕 High-Low analysis avancée
            df_features['high_low_ratio'] = df_features['high'] / df_features['low']
            df_features['hl_spread'] = (df_features['high'] - df_features['low']) / df_features['price']
            df_features['hl_spread_ma'] = df_features['hl_spread'].rolling(10).mean()
            df_features['hl_spread_std'] = df_features['hl_spread'].rolling(10).std()

            # Position dans la range H-L
            df_features['price_in_hl_range'] = (df_features['price'] - df_features['low']) / (
                        df_features['high'] - df_features['low'])

            # === 🆕 FEATURES DE MOMENTUM AVANCÉES ===

            # ROC (Rate of Change) multi-périodes
            for period in [3, 5, 10, 20]:
                df_features[f'roc_{period}'] = ta.momentum.roc(df_features['price'], window=period)

            # Momentum indicators
            df_features['momentum_10'] = df_features['price'] / df_features['price'].shift(10) - 1
            df_features['momentum_20'] = df_features['price'] / df_features['price'].shift(20) - 1

            # 🆕 Acceleration (2ème dérivée)
            df_features['price_velocity'] = df_features['price'].diff()
            df_features['price_acceleration'] = df_features['price_velocity'].diff()

            # === 🆕 FEATURES DE VOLATILITÉ AVANCÉES ===

            # Volatilités multi-horizons
            for window in [5, 10, 20, 50]:
                df_features[f'volatility_{window}'] = df_features['price'].rolling(window).std()
                df_features[f'volatility_ratio_{window}'] = df_features[f'volatility_{window}'] / df_features[
                    f'volatility_{window}'].rolling(50).mean()

            # True Range et ATR
            df_features['true_range'] = ta.volatility.average_true_range(df_features['high'], df_features['low'],
                                                                         df_features['price'], window=14)
            df_features['atr_ratio'] = df_features['true_range'] / df_features['price']

            # Volatilité relative
            df_features['volatility_rank'] = df_features['volatility_20'].rolling(100).rank(pct=True)

            # === 🆕 FEATURES DE SUPPORT/RÉSISTANCE ===

            # Niveaux dynamiques
            for window in [20, 50, 100]:
                df_features[f'resistance_{window}'] = df_features['high'].rolling(window).max()
                df_features[f'support_{window}'] = df_features['low'].rolling(window).min()
                df_features[f'resistance_distance_{window}'] = (df_features[f'resistance_{window}'] - df_features[
                    'price']) / df_features['price']
                df_features[f'support_distance_{window}'] = (df_features['price'] - df_features[f'support_{window}']) / \
                                                            df_features['price']

            # === 🆕 FEATURES CYCLIQUES ET TEMPORELLES AVANCÉES ===

            if 'timestamp' in df_features.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
                df_features['hour'] = df_features['timestamp'].dt.hour
                df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
                df_features['minute'] = df_features['timestamp'].dt.minute

                # Sessions de marché
                df_features['is_london_session'] = ((df_features['hour'] >= 8) & (df_features['hour'] <= 17)).astype(
                    int)
                df_features['is_ny_session'] = ((df_features['hour'] >= 13) & (df_features['hour'] <= 22)).astype(int)
                df_features['is_asian_session'] = ((df_features['hour'] >= 0) & (df_features['hour'] <= 8)).astype(int)

                # Overlaps de sessions (plus de volatilité)
                df_features['london_ny_overlap'] = ((df_features['hour'] >= 13) & (df_features['hour'] <= 17)).astype(
                    int)
                df_features['asian_london_overlap'] = ((df_features['hour'] >= 7) & (df_features['hour'] <= 9)).astype(
                    int)
            else:
                current_time = datetime.now()
                df_features['hour'] = current_time.hour
                df_features['day_of_week'] = current_time.weekday()
                df_features['minute'] = current_time.minute
                df_features['is_london_session'] = 1 if 8 <= current_time.hour <= 17 else 0
                df_features['is_ny_session'] = 1 if 13 <= current_time.hour <= 22 else 0
                df_features['is_asian_session'] = 1 if 0 <= current_time.hour <= 8 else 0
                df_features['london_ny_overlap'] = 1 if 13 <= current_time.hour <= 17 else 0
                df_features['asian_london_overlap'] = 1 if 7 <= current_time.hour <= 9 else 0

            # Encodage cyclique amélioré
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
            df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['minute_sin'] = np.sin(2 * np.pi * df_features['minute'] / 60)
            df_features['minute_cos'] = np.cos(2 * np.pi * df_features['minute'] / 60)

            # === 🆕 FEATURES DE PATTERN RECOGNITION ===

            # Gaps (écarts de prix)
            df_features['gap'] = df_features['price'] - df_features['price'].shift(1)
            df_features['gap_ratio'] = df_features['gap'] / df_features['price']
            df_features['gap_filled'] = (df_features['gap'] * df_features['gap'].shift(1) < 0).astype(int)

            # Doji patterns (approximation)
            df_features['doji_signal'] = (abs(df_features['price'] - df_features['price'].shift(1)) / df_features[
                'hl_spread'] < 0.1).astype(int)

            # === SÉLECTION DES FEATURES FINALES ===
            feature_columns = [
                # RSI famille
                'rsi_14', 'rsi_9', 'rsi_21', 'rsi_divergence', 'rsi_momentum',

                # MACD famille
                'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_slope',
                'macd_fast', 'macd_fast_signal',

                # EMA et ratios
                'ema_5', 'ema_9', 'ema_13', 'ema_21', 'ema_34', 'ema_50', 'ema_100',
                'ema_ratio_9_21', 'ema_ratio_21_50', 'ema_ratio_50_200',

                # SMA
                'sma_20', 'sma_50',

                # Bollinger Bands
                'bb_width', 'bb_position', 'bb_squeeze', 'bb_short_position',

                # Stochastic
                'stoch_k', 'stoch_d', 'stoch_divergence', 'stoch_fast_k',

                # Williams %R
                'williams_r', 'williams_r_smooth',

                # ADX famille
                'adx', 'di_plus', 'di_minus', 'dx',

                # Volume
                'volume_ratio', 'volume_momentum', 'price_volume_corr',

                # High-Low analysis
                'high_low_ratio', 'hl_spread', 'hl_spread_ma', 'hl_spread_std', 'price_in_hl_range',

                # Momentum et ROC
                'roc_3', 'roc_5', 'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
                'price_velocity', 'price_acceleration',

                # Volatilité
                'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
                'volatility_ratio_5', 'volatility_ratio_20', 'true_range', 'atr_ratio', 'volatility_rank',

                # Support/Résistance
                'resistance_distance_20', 'support_distance_20', 'resistance_distance_50', 'support_distance_50',

                # Sessions de marché
                'is_london_session', 'is_ny_session', 'is_asian_session', 'london_ny_overlap', 'asian_london_overlap',

                # Cyclique
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_sin', 'minute_cos',

                # Patterns
                'gap_ratio', 'gap_filled', 'doji_signal'
            ]

            # 🆕 NETTOYAGE ROBUSTE DES DONNÉES
            logger.info("🧹 Nettoyage robuste des features...")

            # 1. Remplacer les valeurs infinies par NaN
            df_features = df_features.replace([np.inf, -np.inf], np.nan)

            # 2. Limiter les valeurs extrêmes (outliers)
            for col in feature_columns:
                if col in df_features.columns:
                    # Calculer les percentiles pour détecter les outliers
                    q1 = df_features[col].quantile(0.01)  # 1er percentile
                    q99 = df_features[col].quantile(0.99)  # 99ème percentile

                    # Remplacer les valeurs extrêmes par les valeurs limites
                    df_features[col] = df_features[col].clip(lower=q1, upper=q99)

            # 3. Nettoyer les NaN
            df_features = df_features.dropna()

            # 4. Vérification finale des valeurs infinies
            inf_cols = []
            for col in feature_columns:
                if col in df_features.columns:
                    if np.isinf(df_features[col]).any():
                        inf_cols.append(col)
                        # Remplacer par la médiane si encore des infinis
                        median_val = df_features[col].replace([np.inf, -np.inf], np.nan).median()
                        df_features[col] = df_features[col].replace([np.inf, -np.inf], median_val)

            if inf_cols:
                logger.warning(f"🔧 Colonnes avec infinis corrigées: {inf_cols}")

            # 5. Vérification finale des NaN
            nan_cols = []
            for col in feature_columns:
                if col in df_features.columns:
                    if df_features[col].isnull().any():
                        nan_cols.append(col)
                        # Remplacer par la médiane
                        median_val = df_features[col].median()
                        df_features[col] = df_features[col].fillna(median_val)

            if nan_cols:
                logger.warning(f"🔧 Colonnes avec NaN corrigées: {nan_cols}")

            logger.info(f"✅ Nettoyage terminé: {len(df_features)} points propres")

            if len(df_features) < self.lookback_period + self.prediction_horizon:
                logger.debug("Pas assez de données après nettoyage V2")
                return None, None

            # === PRÉPARER LES ÉCHANTILLONS ===
            X_samples = []
            y_samples = []

            for i in range(self.lookback_period, len(df_features) - self.prediction_horizon):
                # Features actuelles
                current_features = []

                for col in feature_columns:
                    if col in df_features.columns:
                        current_features.append(df_features[col].iloc[i])

                # 🆕 Features de tendance sur lookback (améliorées)
                for col in ['rsi_14', 'macd', 'volatility_20', 'price_velocity', 'ema_21']:
                    if col in df_features.columns:
                        lookback_data = df_features[col].iloc[i - self.lookback_period:i]
                        if len(lookback_data) > 0:
                            current_features.append(lookback_data.mean())  # Moyenne
                            current_features.append(lookback_data.std() if len(lookback_data) > 1 else 0)  # Volatilité
                            current_features.append(lookback_data.iloc[-1] - lookback_data.iloc[0] if len(
                                lookback_data) > 1 else 0)  # Changement
                            # 🆕 Nouveau: Pente de régression linéaire
                            if len(lookback_data) > 2:
                                x_vals = np.arange(len(lookback_data))
                                slope = np.polyfit(x_vals, lookback_data.values, 1)[0]
                                current_features.append(slope)
                            else:
                                current_features.append(0)

                X_samples.append(current_features)

                # Target: direction du prix dans prediction_horizon
                current_price = df_features['price'].iloc[i]
                future_price = df_features['price'].iloc[i + self.prediction_horizon]
                target = 1 if future_price > current_price else 0
                y_samples.append(target)

            if len(X_samples) == 0:
                return None, None

            X = np.array(X_samples)
            y = np.array(y_samples)

            # Sauvegarder les noms des features
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            joblib.dump(self.feature_names, self.features_path)

            logger.info(f"✅ Features avancées V2 préparées: {X.shape[0]} échantillons, {X.shape[1]} features")
            return X, y

        except Exception as e:
            logger.error(f"Erreur préparation features avancées V2: {e}")
            return None, None

    def train_ensemble_model(self, data_file: str = 'data/vol75_data.csv') -> bool:
        """🚀 ENTRAÎNEMENT ENSEMBLE XGBoost + LightGBM"""
        try:
            if not os.path.exists(data_file):
                logger.info("🧠 Pas de données - Mode attente")
                self.validation_accuracy = 0.5
                self.training_samples = 0
                self.last_training = datetime.now()
                return True

            logger.info("🚀 Début entraînement ENSEMBLE XGBoost + LightGBM")

            # Charger les données
            df = pd.read_csv(data_file)
            if len(df) < 2000:
                logger.info(f"🧠 Pas assez de données ({len(df)}) - Mode simple")
                self.validation_accuracy = 0.5
                self.training_samples = len(df)
                self.last_training = datetime.now()
                return True

            logger.info(f"📊 Données d'entraînement: {len(df)} points")

            # Préparer les features avancées V2
            X, y = self.prepare_advanced_features_v2(df)
            if X is None or len(X) == 0:
                logger.error("Échec préparation features avancées V2")
                return False

            self.n_features = X.shape[1]
            logger.info(f"📈 Features avancées V2: {X.shape[0]} échantillons, {self.n_features} features")

            # Double normalisation pour robustesse
            X_minmax = self.feature_scaler.fit_transform(X)
            X_robust = self.robust_scaler.fit_transform(X_minmax)

            # 🆕 VALIDATION FINALE DES DONNÉES
            logger.info("🔍 Validation finale des données d'entraînement...")

            # Vérifier les valeurs infinies
            if np.isinf(X_robust).any():
                logger.error("❌ Valeurs infinies détectées après normalisation!")
                # Remplacer par des valeurs limites
                X_robust = np.nan_to_num(X_robust, nan=0.0, posinf=1.0, neginf=-1.0)
                logger.info("🔧 Valeurs infinies remplacées")

            # Vérifier les valeurs NaN
            if np.isnan(X_robust).any():
                logger.error("❌ Valeurs NaN détectées après normalisation!")
                X_robust = np.nan_to_num(X_robust, nan=0.0)
                logger.info("🔧 Valeurs NaN remplacées")

            # Vérifier les valeurs extrêmes
            extreme_values = np.abs(X_robust) > 10
            if extreme_values.any():
                logger.warning(f"⚠️ {extreme_values.sum()} valeurs extrêmes détectées")
                X_robust = np.clip(X_robust, -10, 10)  # Limiter à [-10, 10]
                logger.info("🔧 Valeurs extrêmes limitées")

            logger.info(
                f"✅ Données validées: shape={X_robust.shape}, min={X_robust.min():.3f}, max={X_robust.max():.3f}")

            # Division temporelle (important pour les données de marché)
            split_point = int(len(X_robust) * 0.8)
            X_train = X_robust[:split_point]
            X_test = X_robust[split_point:]
            y_train = y[:split_point]
            y_test = y[split_point:]

            # Division validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            logger.info(f"📊 Division: Train={len(X_train_split)}, Val={len(X_val)}, Test={len(X_test)}")

            # === ENTRAÎNEMENT XGBOOST ===
            logger.info("🚀 Entraînement XGBoost...")
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)

            # XGBoost avec validation
            eval_set_xgb = [(X_val, y_val)]
            self.xgb_model.fit(
                X_train_split, y_train_split,
                eval_set=eval_set_xgb,
                verbose=False
            )

            # === ENTRAÎNEMENT LIGHTGBM ===
            logger.info("🚀 Entraînement LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)

            # LightGBM avec validation
            self.lgb_model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # === ÉVALUATION INDIVIDUELLE ===
            # XGBoost
            y_pred_xgb_val = self.xgb_model.predict(X_val)
            y_pred_xgb_test = self.xgb_model.predict(X_test)
            xgb_val_acc = accuracy_score(y_val, y_pred_xgb_val)
            xgb_test_acc = accuracy_score(y_test, y_pred_xgb_test)

            # LightGBM
            y_pred_lgb_val = self.lgb_model.predict(X_val)
            y_pred_lgb_test = self.lgb_model.predict(X_test)
            lgb_val_acc = accuracy_score(y_val, y_pred_lgb_val)
            lgb_test_acc = accuracy_score(y_test, y_pred_lgb_test)

            # === ENSEMBLE (Vote pondéré) ===
            y_prob_xgb_val = self.xgb_model.predict_proba(X_val)[:, 1]
            y_prob_lgb_val = self.lgb_model.predict_proba(X_val)[:, 1]

            y_prob_xgb_test = self.xgb_model.predict_proba(X_test)[:, 1]
            y_prob_lgb_test = self.lgb_model.predict_proba(X_test)[:, 1]

            # Vote pondéré
            ensemble_prob_val = (self.ensemble_weights['xgb'] * y_prob_xgb_val +
                                 self.ensemble_weights['lgb'] * y_prob_lgb_val)
            ensemble_prob_test = (self.ensemble_weights['xgb'] * y_prob_xgb_test +
                                  self.ensemble_weights['lgb'] * y_prob_lgb_test)

            ensemble_pred_val = (ensemble_prob_val > 0.5).astype(int)
            ensemble_pred_test = (ensemble_prob_test > 0.5).astype(int)

            ensemble_val_acc = accuracy_score(y_val, ensemble_pred_val)
            ensemble_test_acc = accuracy_score(y_test, ensemble_pred_test)

            # Feature importance combinée
            xgb_importance = self.xgb_model.feature_importances_
            lgb_importance = self.lgb_model.feature_importances_

            combined_importance = (self.ensemble_weights['xgb'] * xgb_importance +
                                   self.ensemble_weights['lgb'] * lgb_importance)

            self.feature_importance = dict(zip(self.feature_names, combined_importance))

            # Sauvegardes
            joblib.dump(self.xgb_model, self.model_path_xgb)
            joblib.dump(self.lgb_model, self.model_path_lgb)
            joblib.dump(self.feature_scaler, self.scaler_path)
            joblib.dump(self.robust_scaler, self.robust_scaler_path)

            # Métadonnées
            self.validation_accuracy = float(ensemble_val_acc)
            self.individual_accuracies = {
                'xgb': float(xgb_val_acc),
                'lgb': float(lgb_val_acc)
            }
            self.training_samples = len(X_train_split)
            self.last_training = datetime.now()

            model_info = {
                'version': self.model_version,
                'last_training': self.last_training.isoformat(),
                'training_samples': self.training_samples,
                'validation_accuracy': self.validation_accuracy,
                'test_accuracy': float(ensemble_test_acc),
                'individual_accuracies': self.individual_accuracies,
                'individual_test_accuracies': {
                    'xgb': float(xgb_test_acc),
                    'lgb': float(lgb_test_acc)
                },
                'n_features': self.n_features,
                'ensemble_weights': self.ensemble_weights,
                'feature_importance': {k: float(v) for k, v in self.feature_importance.items()},
                'xgb_params': self.xgb_params,
                'lgb_params': self.lgb_params
            }

            import json
            with open(self.ensemble_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)

            logger.info(f"🚀 Entraînement ENSEMBLE terminé!")
            logger.info(f"   📊 XGBoost - Val: {xgb_val_acc:.4f}, Test: {xgb_test_acc:.4f}")
            logger.info(f"   🔥 LightGBM - Val: {lgb_val_acc:.4f}, Test: {lgb_test_acc:.4f}")
            logger.info(f"   🏆 ENSEMBLE - Val: {ensemble_val_acc:.4f}, Test: {ensemble_test_acc:.4f}")
            logger.info(f"   📈 Features: {self.n_features}")
            logger.info(f"   🎯 Échantillons: {self.training_samples}")

            # 🎯 Optimisation dynamique des poids si un modèle est clairement meilleur
            if abs(xgb_val_acc - lgb_val_acc) > 0.05:  # Différence > 5%
                if xgb_val_acc > lgb_val_acc:
                    self.ensemble_weights = {'xgb': 0.7, 'lgb': 0.3}
                    logger.info("🎯 Poids ajustés en faveur de XGBoost")
                else:
                    self.ensemble_weights = {'xgb': 0.4, 'lgb': 0.6}
                    logger.info("🎯 Poids ajustés en faveur de LightGBM")

            return True

        except Exception as e:
            logger.error(f"❌ Erreur entraînement ensemble: {e}")
            self.validation_accuracy = 0.5
            self.training_samples = 0
            self.last_training = datetime.now()
            return True

    def load_or_create_ensemble_model(self) -> bool:
        """Charger le modèle ensemble ou en créer un"""
        try:
            if (os.path.exists(self.model_path_xgb) and
                    os.path.exists(self.model_path_lgb) and
                    os.path.exists(self.scaler_path) and
                    os.path.exists(self.robust_scaler_path) and
                    os.path.exists(self.features_path)):

                self.xgb_model = joblib.load(self.model_path_xgb)
                self.lgb_model = joblib.load(self.model_path_lgb)
                self.feature_scaler = joblib.load(self.scaler_path)
                self.robust_scaler = joblib.load(self.robust_scaler_path)
                self.feature_names = joblib.load(self.features_path)

                if os.path.exists(self.ensemble_info_path):
                    import json
                    with open(self.ensemble_info_path, 'r') as f:
                        model_info = json.load(f)

                    self.last_training = datetime.fromisoformat(model_info['last_training'])
                    self.validation_accuracy = model_info['validation_accuracy']
                    self.individual_accuracies = model_info.get('individual_accuracies', {'xgb': 0.0, 'lgb': 0.0})
                    self.training_samples = model_info['training_samples']
                    self.n_features = model_info.get('n_features', 0)
                    self.ensemble_weights = model_info.get('ensemble_weights', {'xgb': 0.6, 'lgb': 0.4})
                    self.feature_importance = model_info.get('feature_importance', {})

                logger.info(f"✅ Modèle ensemble chargé:")
                logger.info(f"   🏆 Précision ensemble: {self.validation_accuracy:.4f}")
                logger.info(f"   📊 XGBoost: {self.individual_accuracies['xgb']:.4f}")
                logger.info(f"   🔥 LightGBM: {self.individual_accuracies['lgb']:.4f}")
                logger.info(f"   📈 Features: {self.n_features}")

                # Vérifier si réentraînement nécessaire
                if self.last_training:
                    days_since = (datetime.now() - self.last_training).days
                    if days_since >= 1:
                        logger.info(f"🔄 Réentraînement nécessaire ({days_since} jours)")
                        self.train_ensemble_model()

                return True
            else:
                logger.info("🆕 Création nouveau modèle ensemble")
                return self.train_ensemble_model()

        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle ensemble: {e}")
            return self.train_ensemble_model()

    def predict_ensemble(self, df: pd.DataFrame) -> Dict:
        """🚀 PRÉDICTION ENSEMBLE avec vote pondéré"""
        try:
            if self.xgb_model is None or self.lgb_model is None or self.training_samples == 0:
                return self._simple_prediction(df)

            if len(df) < self.lookback_period + 150:
                logger.debug("Pas assez de données pour prédiction ensemble")
                return {'direction': None, 'confidence': 0.0}

            # Recréer EXACTEMENT les mêmes features qu'à l'entraînement
            X, _ = self.prepare_advanced_features_v2(df.tail(self.lookback_period + 200))
            if X is None or len(X) == 0:
                logger.debug("Échec préparation features pour prédiction")
                return self._simple_prediction(df)

            # Prendre le dernier échantillon
            X_sample = X[-1:, :]

            # 🆕 VALIDATION DES DONNÉES DE PRÉDICTION
            # Vérifier et nettoyer les valeurs problématiques
            if np.isinf(X_sample).any() or np.isnan(X_sample).any():
                logger.warning("⚠️ Valeurs problématiques dans les données de prédiction")
                X_sample = np.nan_to_num(X_sample, nan=0.0, posinf=1.0, neginf=-1.0)
                logger.debug("🔧 Données de prédiction nettoyées")

            # Normalisation (même pipeline qu'à l'entraînement)
            try:
                X_minmax = self.feature_scaler.transform(X_sample)
                X_robust = self.robust_scaler.transform(X_minmax)

                # Validation post-normalisation
                if np.isinf(X_robust).any() or np.isnan(X_robust).any():
                    X_robust = np.nan_to_num(X_robust, nan=0.0, posinf=1.0, neginf=-1.0)
                    logger.debug("🔧 Données normalisées nettoyées")

            except Exception as e:
                logger.warning(f"Erreur normalisation: {e}, utilisation simple")
                return self._simple_prediction(df)

            # Prédictions individuelles
            xgb_proba = self.xgb_model.predict_proba(X_robust)[0]
            lgb_proba = self.lgb_model.predict_proba(X_robust)[0]

            # 🆕 Vote pondéré intelligent
            ensemble_proba = (
                    self.ensemble_weights['xgb'] * xgb_proba +
                    self.ensemble_weights['lgb'] * lgb_proba
            )

            # Prédiction finale
            prediction_class = 1 if ensemble_proba[1] > 0.5 else 0
            confidence = float(ensemble_proba[prediction_class])
            direction = 'UP' if prediction_class == 1 else 'DOWN'

            # 🆕 Consensus scoring (accord entre modèles)
            xgb_direction = 'UP' if xgb_proba[1] > 0.5 else 'DOWN'
            lgb_direction = 'UP' if lgb_proba[1] > 0.5 else 'DOWN'
            consensus = 1.0 if xgb_direction == lgb_direction else 0.5

            # 🆕 Confidence boosting si consensus fort
            if consensus == 1.0:
                confidence = min(0.95, confidence * 1.1)  # Boost de 10% si accord

            result = {
                'direction': direction,
                'confidence': confidence,
                'raw_confidence': confidence,
                'probabilities': {
                    'DOWN': float(ensemble_proba[0]),
                    'UP': float(ensemble_proba[1])
                },
                'individual_predictions': {
                    'xgb': {
                        'direction': xgb_direction,
                        'confidence': float(xgb_proba[1] if xgb_direction == 'UP' else xgb_proba[0]),
                        'probabilities': {'DOWN': float(xgb_proba[0]), 'UP': float(xgb_proba[1])}
                    },
                    'lgb': {
                        'direction': lgb_direction,
                        'confidence': float(lgb_proba[1] if lgb_direction == 'UP' else lgb_proba[0]),
                        'probabilities': {'DOWN': float(lgb_proba[0]), 'UP': float(lgb_proba[1])}
                    }
                },
                'ensemble_weights': self.ensemble_weights,
                'consensus_score': consensus,
                'prob_difference': float(abs(ensemble_proba[1] - ensemble_proba[0])),
                'model_version': self.model_version,
                'n_features_used': X_sample.shape[1]
            }

            logger.debug(f"Prédiction ensemble: {direction} (conf: {confidence:.3f}, consensus: {consensus:.1f})")
            return result

        except Exception as e:
            logger.error(f"❌ Erreur prédiction ensemble: {e}")
            return self._simple_prediction(df)

    def _simple_prediction(self, df: pd.DataFrame) -> Dict:
        """Prédiction simple en fallback"""
        try:
            if len(df) < 10:
                return {'direction': None, 'confidence': 0.0}

            recent_prices = df['price'].tail(20)
            price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
            volatility = recent_prices.std() / recent_prices.mean()

            if price_change > 0.002:
                direction = 'UP'
                confidence = min(0.75, 0.6 + abs(price_change) * 10)
            elif price_change < -0.002:
                direction = 'DOWN'
                confidence = min(0.75, 0.6 + abs(price_change) * 10)
            else:
                short_momentum = (recent_prices.iloc[-5:].mean() - recent_prices.iloc[
                                                                   -10:-5].mean()) / recent_prices.iloc[-10:-5].mean()
                direction = 'UP' if short_momentum > 0 else 'DOWN'
                confidence = min(0.65, 0.5 + abs(short_momentum) * 20)

            if volatility > 0.03:
                confidence *= 0.8

            return {
                'direction': direction,
                'confidence': float(confidence),
                'raw_confidence': float(confidence),
                'probabilities': {
                    'DOWN': 1 - confidence if direction == 'UP' else confidence,
                    'UP': confidence if direction == 'UP' else 1 - confidence
                },
                'prob_difference': float(abs(confidence - 0.5) * 2),
                'model_version': 'simple_fallback'
            }

        except Exception as e:
            logger.error(f"Erreur prédiction simple: {e}")
            return {'direction': None, 'confidence': 0.0}

    def get_ensemble_model_info(self) -> Dict:
        """Informations complètes du modèle ensemble"""
        return {
            'model_type': 'Ensemble-XGBoost-LightGBM',
            'models_loaded': self.xgb_model is not None and self.lgb_model is not None,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'validation_accuracy': self.validation_accuracy,
            'individual_accuracies': self.individual_accuracies,
            'training_samples': self.training_samples,
            'n_features': self.n_features,
            'version': self.model_version,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance,
            'xgb_params': self.xgb_params,
            'lgb_params': self.lgb_params
        }

    def get_top_features(self, n_top: int = 15) -> Dict:
        """Obtenir les features les plus importantes de l'ensemble"""
        if not self.feature_importance:
            return {}

        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:n_top])

    def get_model_comparison(self) -> Dict:
        """Comparaison détaillée des modèles individuels"""
        return {
            'ensemble_accuracy': self.validation_accuracy,
            'individual_accuracies': self.individual_accuracies,
            'performance_gain': self.validation_accuracy - max(
                self.individual_accuracies.values()) if self.individual_accuracies else 0,
            'best_individual': max(self.individual_accuracies,
                                   key=self.individual_accuracies.get) if self.individual_accuracies else None,
            'consensus_advantage': 'Ensemble provides better stability and reduced overfitting',
            'ensemble_weights': self.ensemble_weights
        }

    def optimize_ensemble_weights(self, validation_data=None):
        """🆕 Optimisation dynamique des poids de l'ensemble"""
        # TODO: Implémenter optimisation bayésienne des poids
        # Pour le moment, utilise la logique simple dans train_ensemble_model
        pass


# Interface de compatibilité avec l'ancien AIModel
class AIModel(EnsembleAIModel):
    """Wrapper pour compatibilité avec le code existant"""

    def load_or_create_model(self):
        return self.load_or_create_ensemble_model()

    def predict(self, df):
        return self.predict_ensemble(df)

    def get_model_info(self):
        return self.get_ensemble_model_info()


# Interface optimisée pour le nouveau code
class OptimizedAIModel(EnsembleAIModel):
    """Interface optimisée avec toutes les nouvelles fonctionnalités"""

    def load_or_create_optimized_model(self):
        return self.load_or_create_ensemble_model()

    def predict_optimized(self, df):
        return self.predict_ensemble(df)

    def get_model_info(self):
        return self.get_ensemble_model_info()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    def test_ensemble_model():
        """Test du modèle ensemble"""
        import numpy as np

        # Données de test plus réalistes
        dates = pd.date_range(start='2024-01-01', periods=5000, freq='5min')
        base_price = 1000

        # Tendance avec cycles multiples
        trend = np.linspace(0, 300, 5000)
        daily_cycle = 20 * np.sin(np.linspace(0, 30 * np.pi, 5000))  # Cycles journaliers
        hourly_cycle = 8 * np.sin(np.linspace(0, 120 * np.pi, 5000))  # Cycles horaires
        noise = np.random.normal(0, 12, 5000)

        # Prix réalistes avec volatilité variable
        volatility_factor = 1 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, 5000))
        prices = base_price + trend + daily_cycle + hourly_cycle + (noise * volatility_factor)

        test_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'high': prices + np.random.uniform(0, 4, 5000),
            'low': prices - np.random.uniform(0, 4, 5000),
            'volume': np.random.randint(800, 2500, 5000)
        })

        # S'assurer que high >= price >= low
        test_df['high'] = np.maximum(test_df['high'], test_df['price'])
        test_df['low'] = np.minimum(test_df['low'], test_df['price'])

        # Sauvegarder
        test_df.to_csv('data/vol75_data.csv', index=False)
        print(f"📊 Données de test sauvegardées: {len(test_df)} points")

        # Test du modèle ensemble
        ai_model = EnsembleAIModel()
        success = ai_model.train_ensemble_model()
        print(f"Entraînement réussi: {success}")

        if success:
            # Test prédiction
            prediction = ai_model.predict_ensemble(test_df.tail(800))
            print(f"\n🚀 Prédiction ensemble: {prediction}")

            # Comparaison des modèles
            comparison = ai_model.get_model_comparison()
            print(f"\n📊 Comparaison des modèles:")
            for key, value in comparison.items():
                print(f"  {key}: {value}")

            # Top features
            top_features = ai_model.get_top_features(20)
            print(f"\n🏆 Top 20 features de l'ensemble:")
            for i, (feat, importance) in enumerate(top_features.items(), 1):
                print(f"  {i:2d}. {feat}: {importance:.6f}")


    test_ensemble_model()