#!/usr/bin/env python3
"""
Technical Analysis - Calcul des indicateurs techniques pour Vol75
CORRECTION: Fix Williams %R et gestion des colonnes manquantes
"""

import pandas as pd
import numpy as np
import ta  # Utiliser 'ta' au lieu de 'pandas_ta'
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Classe pour l'analyse technique des données Vol75"""

    def __init__(self):
        """Initialisation des paramètres d'analyse technique"""
        # Paramètres RSI
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Paramètres MACD
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

        # Périodes EMA
        self.ema_periods = [9, 21, 50]

        # Paramètres de volatilité
        self.volatility_period = 20

        # Paramètres support/résistance
        self.sr_lookback = 20  # Période pour identifier S/R

        # Paramètres Williams %R
        self.williams_r_period = 14

        # Paramètres ADX
        self.adx_period = 14

        logger.info("📊 Module d'analyse technique initialisé")

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculer tous les indicateurs techniques - CORRIGÉ"""
        try:
            if df is None or len(df) < max(self.ema_periods):
                logger.debug("Pas assez de données pour calculer les indicateurs")
                return None

            # 🆕 NOUVEAU: S'assurer que toutes les colonnes nécessaires existent
            df_work = self._ensure_required_columns(df)

            indicators = {}

            # Prix actuel
            indicators['current_price'] = float(df_work['price'].iloc[-1])

            # RSI
            try:
                rsi_values = ta.momentum.rsi(df_work['price'], window=self.rsi_period)
                if not rsi_values.empty and not pd.isna(rsi_values.iloc[-1]):
                    indicators['rsi'] = float(rsi_values.iloc[-1])
                else:
                    indicators['rsi'] = 50.0  # Valeur neutre par défaut
            except Exception as e:
                logger.debug(f"Erreur RSI: {e}")
                indicators['rsi'] = 50.0

            # MACD
            try:
                macd_data = ta.trend.MACD(df_work['price'],
                                          window_fast=self.macd_fast,
                                          window_slow=self.macd_slow,
                                          window_sign=self.macd_signal)

                if macd_data is not None:
                    macd_line = macd_data.macd()
                    macd_signal_line = macd_data.macd_signal()
                    macd_histogram = macd_data.macd_diff()

                    indicators['macd'] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
                    indicators['macd_signal'] = float(macd_signal_line.iloc[-1]) if not pd.isna(
                        macd_signal_line.iloc[-1]) else 0.0
                    indicators['macd_histogram'] = float(macd_histogram.iloc[-1]) if not pd.isna(
                        macd_histogram.iloc[-1]) else 0.0
                else:
                    indicators['macd'] = 0.0
                    indicators['macd_signal'] = 0.0
                    indicators['macd_histogram'] = 0.0
            except Exception as e:
                logger.debug(f"Erreur MACD: {e}")
                indicators['macd'] = 0.0
                indicators['macd_signal'] = 0.0
                indicators['macd_histogram'] = 0.0

            # EMA (Exponential Moving Averages)
            for period in self.ema_periods:
                try:
                    ema_values = ta.trend.ema_indicator(df_work['price'], window=period)
                    if not ema_values.empty and not pd.isna(ema_values.iloc[-1]):
                        indicators[f'ema_{period}'] = float(ema_values.iloc[-1])
                    else:
                        indicators[f'ema_{period}'] = indicators['current_price']
                except Exception as e:
                    logger.debug(f"Erreur EMA {period}: {e}")
                    indicators[f'ema_{period}'] = indicators['current_price']

            # Volatilité (écart-type mobile)
            try:
                volatility = df_work['price'].rolling(window=self.volatility_period).std()
                if not volatility.empty and not pd.isna(volatility.iloc[-1]):
                    indicators['volatility'] = float(volatility.iloc[-1])
                    # Volatilité moyenne sur la période complète
                    indicators['avg_volatility'] = float(volatility.mean())
                else:
                    indicators['volatility'] = 0.0
                    indicators['avg_volatility'] = 0.0
            except Exception as e:
                logger.debug(f"Erreur volatilité: {e}")
                indicators['volatility'] = 0.0
                indicators['avg_volatility'] = 0.0

            # Support et Résistance
            try:
                support, resistance = self._calculate_support_resistance(df_work)
                indicators['support'] = support
                indicators['resistance'] = resistance
            except Exception as e:
                logger.debug(f"Erreur S/R: {e}")
                indicators['support'] = indicators['current_price']
                indicators['resistance'] = indicators['current_price']

            # Bollinger Bands
            try:
                bb_data = ta.volatility.BollingerBands(df_work['price'], window=20)
                if bb_data is not None:
                    bb_upper = bb_data.bollinger_hband()
                    bb_middle = bb_data.bollinger_mavg()
                    bb_lower = bb_data.bollinger_lband()

                    indicators['bb_upper'] = float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else indicators[
                                                                                                                 'current_price'] * 1.02
                    indicators['bb_middle'] = float(bb_middle.iloc[-1]) if not pd.isna(bb_middle.iloc[-1]) else \
                    indicators['current_price']
                    indicators['bb_lower'] = float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else indicators[
                                                                                                                 'current_price'] * 0.98
                else:
                    indicators['bb_upper'] = indicators['current_price'] * 1.02
                    indicators['bb_middle'] = indicators['current_price']
                    indicators['bb_lower'] = indicators['current_price'] * 0.98
            except Exception as e:
                logger.debug(f"Erreur Bollinger Bands: {e}")
                indicators['bb_upper'] = indicators['current_price'] * 1.02
                indicators['bb_middle'] = indicators['current_price']
                indicators['bb_lower'] = indicators['current_price'] * 0.98

            # Stochastic
            try:
                stoch_data = ta.momentum.StochasticOscillator(df_work['high'], df_work['low'], df_work['price'])
                if stoch_data is not None:
                    stoch_k = stoch_data.stoch()
                    stoch_d = stoch_data.stoch_signal()

                    indicators['stoch_k'] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50.0
                    indicators['stoch_d'] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50.0
                else:
                    indicators['stoch_k'] = 50.0
                    indicators['stoch_d'] = 50.0
            except Exception as e:
                logger.debug(f"Erreur Stochastic: {e}")
                indicators['stoch_k'] = 50.0
                indicators['stoch_d'] = 50.0

            # SMA
            try:
                sma_20 = ta.trend.sma_indicator(df_work['price'], window=20)
                if not sma_20.empty and not pd.isna(sma_20.iloc[-1]):
                    indicators['sma_20'] = float(sma_20.iloc[-1])
                else:
                    indicators['sma_20'] = indicators['current_price']
            except Exception as e:
                logger.debug(f"Erreur SMA: {e}")
                indicators['sma_20'] = indicators['current_price']

            # 🆕 CORRECTION: Williams %R avec le bon paramètre
            try:
                # Utiliser 'lbp' au lieu de 'window'
                williams_r = ta.momentum.williams_r(df_work['high'], df_work['low'], df_work['price'],
                                                    lbp=self.williams_r_period)
                if not williams_r.empty and not pd.isna(williams_r.iloc[-1]):
                    indicators['williams_r'] = float(williams_r.iloc[-1])
                else:
                    indicators['williams_r'] = -50.0  # Valeur neutre
            except Exception as e:
                logger.debug(f"Erreur Williams %R: {e}")
                indicators['williams_r'] = -50.0

            # ADX
            try:
                adx_indicator = ta.trend.ADXIndicator(df_work['high'], df_work['low'], df_work['price'],
                                                      window=self.adx_period)
                if adx_indicator is not None:
                    adx_value = adx_indicator.adx()
                    di_plus = adx_indicator.adx_pos()
                    di_minus = adx_indicator.adx_neg()

                    indicators['adx'] = float(adx_value.iloc[-1]) if not pd.isna(adx_value.iloc[-1]) else 0.0
                    indicators['di_plus'] = float(di_plus.iloc[-1]) if not pd.isna(di_plus.iloc[-1]) else 0.0
                    indicators['di_minus'] = float(di_minus.iloc[-1]) if not pd.isna(di_minus.iloc[-1]) else 0.0
                else:
                    indicators['adx'] = 0.0
                    indicators['di_plus'] = 0.0
                    indicators['di_minus'] = 0.0
            except Exception as e:
                logger.debug(f"Erreur ADX: {e}")
                indicators['adx'] = 0.0
                indicators['di_plus'] = 0.0
                indicators['di_minus'] = 0.0

            logger.debug(f"✅ Indicateurs calculés: RSI={indicators['rsi']:.2f}, MACD={indicators['macd']:.5f}")

            return indicators

        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
            return None

    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """🆕 NOUVEAU: S'assurer que toutes les colonnes nécessaires existent"""
        try:
            df_work = df.copy()

            # Créer high/low/volume si manquants
            if 'high' not in df_work.columns:
                df_work['high'] = df_work['price'].rolling(window=3, min_periods=1).max()
                logger.debug("Colonne 'high' créée à partir de 'price'")

            if 'low' not in df_work.columns:
                df_work['low'] = df_work['price'].rolling(window=3, min_periods=1).min()
                logger.debug("Colonne 'low' créée à partir de 'price'")

            if 'volume' not in df_work.columns:
                df_work['volume'] = 1000  # Volume fictif constant
                logger.debug("Colonne 'volume' créée (fictive)")

            # S'assurer que les valeurs ne sont pas NaN
            df_work['high'] = df_work['high'].fillna(df_work['price'])
            df_work['low'] = df_work['low'].fillna(df_work['price'])
            df_work['volume'] = df_work['volume'].fillna(1000)

            # S'assurer que high >= price >= low
            df_work['high'] = np.maximum(df_work['high'], df_work['price'])
            df_work['low'] = np.minimum(df_work['low'], df_work['price'])

            return df_work

        except Exception as e:
            logger.error(f"Erreur préparation colonnes: {e}")
            return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculer les niveaux de support et résistance"""
        try:
            if len(df) < self.sr_lookback:
                return float(df['price'].min()), float(df['price'].max())

            # Prendre les données récentes
            recent_prices = df['price'].tail(self.sr_lookback)

            # Support = minimum récent
            support = float(recent_prices.min())

            # Résistance = maximum récent
            resistance = float(recent_prices.max())

            return support, resistance

        except Exception as e:
            logger.error(f"Erreur calcul S/R: {e}")
            return float(df['price'].iloc[-1]), float(df['price'].iloc[-1])

    def calculate_score(self, df: pd.DataFrame) -> int:
        """Calculer le score technique global (0-100)"""
        try:
            indicators = self.calculate_indicators(df)
            if not indicators:
                return 0

            total_score = 0
            max_possible_score = 100

            # Score RSI (25 points)
            rsi_score = self._calculate_rsi_score(indicators['rsi'])
            total_score += rsi_score

            # Score MACD (25 points)
            macd_score = self._calculate_macd_score(indicators)
            total_score += macd_score

            # Score EMA Trend (25 points)
            ema_score = self._calculate_ema_score(indicators)
            total_score += ema_score

            # Score Volatilité (25 points)
            volatility_score = self._calculate_volatility_score(indicators)
            total_score += volatility_score

            # S'assurer que le score est entre 0 et 100
            final_score = max(0, min(total_score, max_possible_score))

            logger.debug(
                f"Score technique: {final_score} (RSI:{rsi_score}, MACD:{macd_score}, EMA:{ema_score}, Vol:{volatility_score})")

            return int(final_score)

        except Exception as e:
            logger.error(f"Erreur calcul score: {e}")
            return 0

    def _calculate_rsi_score(self, rsi: float) -> int:
        """Calculer le score RSI (0-25 points)"""
        try:
            # RSI optimal entre 30-70 (zone de trading)
            if 30 <= rsi <= 70:
                # Plus proche de 50 = meilleur score
                distance_from_50 = abs(rsi - 50)
                score = 25 - (distance_from_50 / 20 * 10)  # Max 25 points
                return max(15, int(score))  # Minimum 15 points dans la zone
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                return 10  # Zone d'alerte
            else:
                return 0  # Zone dangereuse (surachat/survente extrême)

        except Exception:
            return 0

    def _calculate_macd_score(self, indicators: Dict) -> int:
        """Calculer le score MACD (0-25 points)"""
        try:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_histogram = indicators['macd_histogram']

            score = 0

            # MACD au-dessus de sa ligne de signal (tendance haussière)
            if macd > macd_signal:
                score += 15

                # Histogramme positif (momentum croissant)
                if macd_histogram > 0:
                    score += 10

            # MACD et signal tous deux positifs (forte tendance haussière)
            elif macd > 0 and macd_signal > 0:
                score += 8

            # Au moins une convergence
            elif abs(macd - macd_signal) < abs(indicators.get('prev_macd_diff', 999)):
                score += 5

            return min(score, 25)

        except Exception:
            return 0

    def _calculate_ema_score(self, indicators: Dict) -> int:
        """Calculer le score EMA/Trend (0-25 points)"""
        try:
            price = indicators['current_price']
            ema_9 = indicators['ema_9']
            ema_21 = indicators['ema_21']
            ema_50 = indicators['ema_50']

            score = 0

            # Tendance haussière parfaite: Prix > EMA9 > EMA21 > EMA50
            if price > ema_9 > ema_21 > ema_50:
                score = 25
            # Tendance haussière forte: Prix > EMA21 > EMA50
            elif price > ema_21 > ema_50:
                score = 20
            # Tendance haussière modérée: Prix > EMA21
            elif price > ema_21:
                score = 15
            # Prix au-dessus de EMA9 seulement
            elif price > ema_9:
                score = 10
            # Tendance baissière mais proche des EMA
            elif price > ema_21 * 0.99:  # Dans les 1% de EMA21
                score = 5
            else:
                score = 0

            return score

        except Exception:
            return 0

    def _calculate_volatility_score(self, indicators: Dict) -> int:
        """Calculer le score de volatilité (0-25 points)"""
        try:
            current_vol = indicators['volatility']
            avg_vol = indicators['avg_volatility']

            if avg_vol == 0:
                return 15  # Score neutre si pas de données

            vol_ratio = current_vol / avg_vol

            # Volatilité optimale: proche de la moyenne
            if 0.8 <= vol_ratio <= 1.2:
                score = 25
            elif 0.6 <= vol_ratio <= 1.5:
                score = 20
            elif 0.4 <= vol_ratio <= 2.0:
                score = 15
            elif vol_ratio <= 3.0:
                score = 10
            else:
                score = 0  # Volatilité extrême

            return score

        except Exception:
            return 0

    def get_signal_direction(self, df: pd.DataFrame) -> Optional[str]:
        """Déterminer la direction du signal basée sur l'analyse technique"""
        try:
            indicators = self.calculate_indicators(df)
            if not indicators:
                return None

            bullish_signals = 0
            bearish_signals = 0
            signal_strength = 0

            # Signal RSI
            rsi = indicators['rsi']
            if 30 <= rsi <= 50:
                bullish_signals += 1
                signal_strength += 1
            elif 50 <= rsi <= 70:
                bearish_signals += 1
                signal_strength += 1

            # Signal MACD
            if indicators['macd'] > indicators['macd_signal']:
                bullish_signals += 1
                signal_strength += 1
                if indicators['macd_histogram'] > 0:
                    signal_strength += 1
            else:
                bearish_signals += 1
                signal_strength += 1

            # Signal EMA
            price = indicators['current_price']
            if price > indicators['ema_21']:
                bullish_signals += 1
                signal_strength += 1
                if price > indicators['ema_9'] > indicators['ema_21']:
                    signal_strength += 1
            else:
                bearish_signals += 1
                signal_strength += 1

            # Signal Bollinger Bands
            if price < indicators['bb_lower']:
                bullish_signals += 1  # Oversold
            elif price > indicators['bb_upper']:
                bearish_signals += 1  # Overbought

            # Signal ADX
            if indicators['adx'] > 25:
                signal_strength += 1

            # Décision finale
            if bullish_signals > bearish_signals and signal_strength >= 3:
                return 'BUY'
            elif bearish_signals > bullish_signals and signal_strength >= 3:
                return 'SELL'
            else:
                return None  # Signal pas assez fort

        except Exception as e:
            logger.error(f"Erreur détermination direction: {e}")
            return None

    def get_market_condition(self, df: pd.DataFrame) -> Dict:
        """Analyser les conditions de marché"""
        try:
            indicators = self.calculate_indicators(df)
            if not indicators:
                return {'condition': 'unknown', 'strength': 0}

            # Déterminer la tendance
            price = indicators['current_price']
            ema_21 = indicators['ema_21']
            ema_50 = indicators['ema_50']

            if price > ema_21 > ema_50:
                trend = 'uptrend'
                strength = min(((price - ema_50) / ema_50) * 100, 10)
            elif price < ema_21 < ema_50:
                trend = 'downtrend'
                strength = min(((ema_50 - price) / ema_50) * 100, 10)
            else:
                trend = 'sideways'
                strength = 0

            # Analyser la volatilité
            vol_ratio = indicators['volatility'] / indicators['avg_volatility'] if indicators[
                                                                                       'avg_volatility'] > 0 else 1

            if vol_ratio > 2:
                volatility = 'high'
            elif vol_ratio > 1.5:
                volatility = 'elevated'
            elif vol_ratio < 0.5:
                volatility = 'low'
            else:
                volatility = 'normal'

            # Analyser le momentum
            rsi = indicators['rsi']
            if rsi > 70:
                momentum = 'overbought'
            elif rsi < 30:
                momentum = 'oversold'
            elif 45 <= rsi <= 55:
                momentum = 'neutral'
            else:
                momentum = 'trending'

            return {
                'condition': trend,
                'strength': round(strength, 2),
                'volatility': volatility,
                'momentum': momentum,
                'rsi': round(rsi, 2),
                'price_vs_ema21': round(((price - ema_21) / ema_21) * 100, 2),
                'adx_trend': 'trending' if indicators['adx'] > 25 else 'no_trend'
            }

        except Exception as e:
            logger.error(f"Erreur analyse conditions marché: {e}")
            return {'condition': 'unknown', 'strength': 0}


# Test de la classe si exécuté directement
if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO)


    def test_technical_analysis():
        """Test de l'analyse technique"""
        # Créer des données de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

        # Générer des prix avec tendance et bruit
        base_price = 1000
        trend = np.linspace(0, 50, 100)
        noise = np.random.normal(0, 5, 100)
        prices = base_price + trend + noise

        # Créer DataFrame de test avec toutes les colonnes
        test_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'high': prices + np.random.uniform(0, 2, 100),
            'low': prices - np.random.uniform(0, 2, 100),
            'volume': np.random.randint(100, 1000, 100)
        })

        # Initialiser l'analyse technique
        ta = TechnicalAnalysis()

        # Calculer les indicateurs
        indicators = ta.calculate_indicators(test_df)
        print("Indicateurs calculés:")
        for key, value in indicators.items():
            print(f"  {key}: {value}")

        # Calculer le score
        score = ta.calculate_score(test_df)
        print(f"\nScore technique: {score}/100")

        # Déterminer la direction
        direction = ta.get_signal_direction(test_df)
        print(f"Direction suggérée: {direction}")

        # Analyser les conditions de marché
        market_condition = ta.get_market_condition(test_df)
        print(f"Conditions de marché: {market_condition}")


    test_technical_analysis()