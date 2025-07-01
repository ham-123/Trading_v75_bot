#!/usr/bin/env python3
"""
Enhanced Technical Analysis - Version complète avec tous les indicateurs
Remplace votre technical_analysis.py avec les améliorations du paste.txt
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedTechnicalAnalysis:
    """Classe améliorée pour l'analyse technique Vol75"""

    def __init__(self):
        """Initialisation avec tous les paramètres"""
        # Paramètres existants
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.ema_periods = [9, 21, 50]
        self.volatility_period = 20
        self.sr_lookback = 20

        # NOUVEAUX paramètres d'indicateurs avancés
        self.williams_r_period = 14
        self.adx_period = 14
        self.cci_period = 20
        self.roc_period = 12
        self.atr_period = 14

        # Ichimoku parameters
        self.ichimoku_conversion = 9
        self.ichimoku_base = 26
        self.ichimoku_span_b = 52

        # Volume Profile
        self.volume_profile_bins = 20
        self.value_area_percent = 0.70

        # Fibonacci
        self.fib_lookback = 100

        # Pattern recognition
        self.pattern_window = 20

        logger.info("📊 Enhanced Technical Analysis initialisé avec tous les indicateurs")

    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculer TOUS les indicateurs techniques"""
        try:
            if df is None or len(df) < max(self.ema_periods + [self.ichimoku_span_b]):
                logger.debug("Pas assez de données pour calculer les indicateurs")
                return None

            indicators = {}

            # Prix et colonnes de base
            indicators['current_price'] = float(df['price'].iloc[-1])

            # Créer high/low si manquants (pour les données tick)
            if 'high' not in df.columns:
                df['high'] = df['price'].rolling(window=3).max()
            if 'low' not in df.columns:
                df['low'] = df['price'].rolling(window=3).min()
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Volume fictif pour Vol75

            # === INDICATEURS DE MOMENTUM ===

            # RSI
            rsi_values = ta.momentum.rsi(df['price'], window=self.rsi_period)
            indicators['rsi'] = float(rsi_values.iloc[-1]) if not rsi_values.empty else 50.0

            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['price'])
            indicators['stoch_k'] = float(stoch.stoch().iloc[-1]) if stoch.stoch().iloc[-1] is not None else 50.0
            indicators['stoch_d'] = float(stoch.stoch_signal().iloc[-1]) if stoch.stoch_signal().iloc[
                                                                                -1] is not None else 50.0

            # Williams %R
            williams = ta.momentum.williams_r(df['high'], df['low'], df['price'], length=self.williams_r_period)
            indicators['williams_r'] = float(williams.iloc[-1]) if not williams.empty else -50.0

            # CCI (Commodity Channel Index)
            cci = ta.trend.cci(df['high'], df['low'], df['price'], window=self.cci_period)
            indicators['cci'] = float(cci.iloc[-1]) if not cci.empty else 0.0

            # ROC (Rate of Change)
            roc = ta.momentum.roc(df['price'], window=self.roc_period)
            indicators['roc'] = float(roc.iloc[-1]) if not roc.empty else 0.0

            # === INDICATEURS DE TENDANCE ===

            # MACD
            macd_indicator = ta.trend.MACD(df['price'],
                                           window_fast=self.macd_fast,
                                           window_slow=self.macd_slow,
                                           window_sign=self.macd_signal)
            indicators['macd'] = float(macd_indicator.macd().iloc[-1])
            indicators['macd_signal'] = float(macd_indicator.macd_signal().iloc[-1])
            indicators['macd_histogram'] = float(macd_indicator.macd_diff().iloc[-1])

            # EMA
            for period in self.ema_periods:
                ema = ta.trend.ema_indicator(df['price'], window=period)
                indicators[f'ema_{period}'] = float(ema.iloc[-1]) if not ema.empty else indicators['current_price']

            # SMA
            sma_20 = ta.trend.sma_indicator(df['price'], window=20)
            indicators['sma_20'] = float(sma_20.iloc[-1]) if not sma_20.empty else indicators['current_price']

            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['price'], window=self.adx_period)
            indicators['adx'] = float(adx.adx().iloc[-1]) if adx.adx().iloc[-1] is not None else 0.0
            indicators['di_plus'] = float(adx.adx_pos().iloc[-1]) if adx.adx_pos().iloc[-1] is not None else 0.0
            indicators['di_minus'] = float(adx.adx_neg().iloc[-1]) if adx.adx_neg().iloc[-1] is not None else 0.0

            # Ichimoku Cloud
            ichimoku = self._calculate_ichimoku(df)
            indicators.update(ichimoku)

            # === INDICATEURS DE VOLATILITÉ ===

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['price'], window=20)
            indicators['bb_upper'] = float(bb.bollinger_hband().iloc[-1])
            indicators['bb_middle'] = float(bb.bollinger_mavg().iloc[-1])
            indicators['bb_lower'] = float(bb.bollinger_lband().iloc[-1])
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (indicators['current_price'] - indicators['bb_lower']) / (
                        indicators['bb_upper'] - indicators['bb_lower'])

            # ATR (Average True Range)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['price'], window=self.atr_period)
            indicators['atr'] = float(atr.iloc[-1]) if not atr.empty else 0.0

            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['price'])
            indicators['keltner_upper'] = float(keltner.keltner_channel_hband().iloc[-1])
            indicators['keltner_middle'] = float(keltner.keltner_channel_mband().iloc[-1])
            indicators['keltner_lower'] = float(keltner.keltner_channel_lband().iloc[-1])

            # Volatilité personnalisée
            volatility = df['price'].rolling(window=self.volatility_period).std()
            indicators['volatility'] = float(volatility.iloc[-1]) if not volatility.empty else 0.0
            indicators['avg_volatility'] = float(volatility.mean()) if not volatility.empty else 0.0

            # === NIVEAUX DE SUPPORT/RÉSISTANCE ===

            support, resistance = self._calculate_support_resistance(df)
            indicators['support'] = support
            indicators['resistance'] = resistance

            # Pivot Points
            pivot_points = self._calculate_pivot_points(df)
            indicators.update(pivot_points)

            # Fibonacci Levels
            fib_levels = self._calculate_fibonacci_levels(df)
            indicators.update(fib_levels)

            # === ANALYSE DE VOLUME ===

            # Volume Profile (si volume disponible)
            if df['volume'].sum() > len(df):  # Volume réel
                volume_profile = self._calculate_volume_profile(df)
                indicators.update(volume_profile)
            else:
                indicators.update({'poc': 0.0, 'value_area_high': 0.0, 'value_area_low': 0.0})

            # === PATTERNS ET SIGNAUX ===

            # Divergences
            divergences = self._detect_divergences(df, indicators)
            indicators.update(divergences)

            # Candlestick patterns
            patterns = self._detect_candlestick_patterns(df)
            indicators.update(patterns)

            logger.debug(f"✅ {len(indicators)} indicateurs calculés")
            return indicators

        except Exception as e:
            logger.error(f"Erreur calcul indicateurs: {e}")
            return None

    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculer l'Ichimoku Kinko Hyo"""
        try:
            high = df['high']
            low = df['low']
            close = df['price']

            # Tenkan-sen (Conversion Line)
            tenkan_high = high.rolling(window=self.ichimoku_conversion).max()
            tenkan_low = low.rolling(window=self.ichimoku_conversion).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2

            # Kijun-sen (Base Line)
            kijun_high = high.rolling(window=self.ichimoku_base).max()
            kijun_low = low.rolling(window=self.ichimoku_base).min()
            kijun_sen = (kijun_high + kijun_low) / 2

            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.ichimoku_base)

            # Senkou Span B (Leading Span B)
            senkou_high = high.rolling(window=self.ichimoku_span_b).max()
            senkou_low = low.rolling(window=self.ichimoku_span_b).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.ichimoku_base)

            # Chikou Span (Lagging Span)
            chikou_span = close.shift(-self.ichimoku_base)

            return {
                'tenkan_sen': float(tenkan_sen.iloc[-1]) if not tenkan_sen.empty else 0.0,
                'kijun_sen': float(kijun_sen.iloc[-1]) if not kijun_sen.empty else 0.0,
                'senkou_span_a': float(senkou_span_a.iloc[-1]) if not senkou_span_a.empty else 0.0,
                'senkou_span_b': float(senkou_span_b.iloc[-1]) if not senkou_span_b.empty else 0.0,
                'chikou_span': float(chikou_span.iloc[-1]) if not chikou_span.empty else 0.0,
                'cloud_top': max(float(senkou_span_a.iloc[-1]),
                                 float(senkou_span_b.iloc[-1])) if not senkou_span_a.empty else 0.0,
                'cloud_bottom': min(float(senkou_span_a.iloc[-1]),
                                    float(senkou_span_b.iloc[-1])) if not senkou_span_a.empty else 0.0
            }

        except Exception as e:
            logger.error(f"Erreur calcul Ichimoku: {e}")
            return {
                'tenkan_sen': 0.0, 'kijun_sen': 0.0, 'senkou_span_a': 0.0,
                'senkou_span_b': 0.0, 'chikou_span': 0.0, 'cloud_top': 0.0, 'cloud_bottom': 0.0
            }

    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculer les points pivot"""
        try:
            # Utiliser les dernières 24h de données
            recent_data = df.tail(min(len(df), 288))  # 288 = 24h * 12 (5min intervals)

            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            close_price = recent_data['price'].iloc[-1]

            # Point pivot central
            pivot = (high_price + low_price + close_price) / 3

            # Résistances
            r1 = 2 * pivot - low_price
            r2 = pivot + (high_price - low_price)
            r3 = high_price + 2 * (pivot - low_price)

            # Supports
            s1 = 2 * pivot - high_price
            s2 = pivot - (high_price - low_price)
            s3 = low_price - 2 * (high_price - pivot)

            return {
                'pivot': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3
            }

        except Exception as e:
            logger.error(f"Erreur calcul pivot points: {e}")
            return {}

    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculer les retracements de Fibonacci"""
        try:
            lookback_df = df.tail(self.fib_lookback)
            high_price = lookback_df['high'].max()
            low_price = lookback_df['low'].min()
            price_range = high_price - low_price

            # Déterminer si tendance haussière ou baissière
            price_start = lookback_df['price'].iloc[0]
            price_end = lookback_df['price'].iloc[-1]
            is_uptrend = price_end > price_start

            if is_uptrend:
                # Retracements depuis le haut
                return {
                    'fib_0.0': high_price,
                    'fib_23.6': high_price - (price_range * 0.236),
                    'fib_38.2': high_price - (price_range * 0.382),
                    'fib_50.0': high_price - (price_range * 0.5),
                    'fib_61.8': high_price - (price_range * 0.618),
                    'fib_78.6': high_price - (price_range * 0.786),
                    'fib_100.0': low_price,
                    'fib_trend': 'uptrend'
                }
            else:
                # Extensions depuis le bas
                return {
                    'fib_0.0': low_price,
                    'fib_23.6': low_price + (price_range * 0.236),
                    'fib_38.2': low_price + (price_range * 0.382),
                    'fib_50.0': low_price + (price_range * 0.5),
                    'fib_61.8': low_price + (price_range * 0.618),
                    'fib_78.6': low_price + (price_range * 0.786),
                    'fib_100.0': high_price,
                    'fib_trend': 'downtrend'
                }

        except Exception as e:
            logger.error(f"Erreur calcul Fibonacci: {e}")
            return {}

    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculer le profil de volume"""
        try:
            if 'volume' not in df.columns or df['volume'].sum() <= len(df):
                return {'poc': 0.0, 'value_area_high': 0.0, 'value_area_low': 0.0}

            # Créer des bins de prix
            price_bins = pd.cut(df['price'], bins=self.volume_profile_bins, duplicates='drop')
            volume_by_price = df['volume'].groupby(price_bins).sum()

            if volume_by_price.empty:
                return {'poc': 0.0, 'value_area_high': 0.0, 'value_area_low': 0.0}

            # Point of Control (POC) - prix avec le plus de volume
            poc_bin = volume_by_price.idxmax()
            poc_level = poc_bin.mid

            # Value Area (70% du volume total)
            total_volume = volume_by_price.sum()
            value_area_volume = total_volume * self.value_area_percent

            # Trier par volume et calculer la value area
            sorted_volume = volume_by_price.sort_values(ascending=False)
            cumulative_volume = sorted_volume.cumsum()
            value_area_bins = sorted_volume[cumulative_volume <= value_area_volume]

            if not value_area_bins.empty:
                value_area_low = min([bin.left for bin in value_area_bins.index])
                value_area_high = max([bin.right for bin in value_area_bins.index])
            else:
                value_area_low = poc_level
                value_area_high = poc_level

            return {
                'poc': poc_level,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'volume_concentration': len(value_area_bins) / len(volume_by_price)
            }

        except Exception as e:
            logger.error(f"Erreur calcul Volume Profile: {e}")
            return {'poc': 0.0, 'value_area_high': 0.0, 'value_area_low': 0.0}

    def _detect_divergences(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détecter les divergences RSI/MACD vs Prix"""
        try:
            if len(df) < 20:
                return {'rsi_divergence': 'none', 'macd_divergence': 'none'}

            # Calculer RSI et MACD sur la période
            rsi_series = ta.momentum.rsi(df['price'], window=14)
            macd_series = ta.trend.MACD(df['price']).macd()

            # Prendre les derniers 20 points
            recent_prices = df['price'].tail(20)
            recent_rsi = rsi_series.tail(20)
            recent_macd = macd_series.tail(20)

            # Détection divergence RSI
            rsi_divergence = 'none'
            if len(recent_rsi.dropna()) >= 10:
                price_trend = recent_prices.iloc[-1] - recent_prices.iloc[-10]
                rsi_trend = recent_rsi.iloc[-1] - recent_rsi.iloc[-10]

                if price_trend > 0 and rsi_trend < 0:
                    rsi_divergence = 'bearish'  # Prix monte, RSI baisse
                elif price_trend < 0 and rsi_trend > 0:
                    rsi_divergence = 'bullish'  # Prix baisse, RSI monte

            # Détection divergence MACD
            macd_divergence = 'none'
            if len(recent_macd.dropna()) >= 10:
                price_trend = recent_prices.iloc[-1] - recent_prices.iloc[-10]
                macd_trend = recent_macd.iloc[-1] - recent_macd.iloc[-10]

                if price_trend > 0 and macd_trend < 0:
                    macd_divergence = 'bearish'
                elif price_trend < 0 and macd_trend > 0:
                    macd_divergence = 'bullish'

            return {
                'rsi_divergence': rsi_divergence,
                'macd_divergence': macd_divergence,
                'divergence_strength': self._calculate_divergence_strength(rsi_divergence, macd_divergence)
            }

        except Exception as e:
            logger.error(f"Erreur détection divergences: {e}")
            return {'rsi_divergence': 'none', 'macd_divergence': 'none'}

    def _calculate_divergence_strength(self, rsi_div: str, macd_div: str) -> str:
        """Calculer la force des divergences"""
        if rsi_div != 'none' and macd_div != 'none' and rsi_div == macd_div:
            return 'strong'  # Deux divergences dans la même direction
        elif rsi_div != 'none' or macd_div != 'none':
            return 'moderate'  # Une seule divergence
        else:
            return 'none'

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Détecter les patterns de chandelles"""
        try:
            if len(df) < 3:
                return {'pattern': 'none', 'pattern_strength': 0}

            # Prendre les 3 dernières bougies
            recent = df.tail(3).copy()

            # Calculer open, close si manquants
            if 'open' not in recent.columns:
                recent['open'] = recent['price'].shift(1).fillna(recent['price'])
            if 'close' not in recent.columns:
                recent['close'] = recent['price']

            patterns = []

            # Doji
            last_candle = recent.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            candle_range = last_candle['high'] - last_candle['low']

            if candle_range > 0 and body_size / candle_range < 0.1:
                patterns.append('doji')

            # Hammer/Shooting Star
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']

            if candle_range > 0:
                if lower_shadow > 2 * body_size and upper_shadow < body_size:
                    patterns.append('hammer')
                elif upper_shadow > 2 * body_size and lower_shadow < body_size:
                    patterns.append('shooting_star')

            # Engulfing pattern (2 bougies)
            if len(recent) >= 2:
                prev_candle = recent.iloc[-2]
                curr_candle = recent.iloc[-1]

                prev_body = abs(prev_candle['close'] - prev_candle['open'])
                curr_body = abs(curr_candle['close'] - curr_candle['open'])

                if curr_body > prev_body * 1.5:
                    if (prev_candle['close'] < prev_candle['open'] and
                            curr_candle['close'] > curr_candle['open']):
                        patterns.append('bullish_engulfing')
                    elif (prev_candle['close'] > prev_candle['open'] and
                          curr_candle['close'] < curr_candle['open']):
                        patterns.append('bearish_engulfing')

            pattern_name = patterns[0] if patterns else 'none'
            pattern_strength = len(patterns)

            return {
                'pattern': pattern_name,
                'pattern_strength': pattern_strength,
                'all_patterns': patterns
            }

        except Exception as e:
            logger.error(f"Erreur détection patterns: {e}")
            return {'pattern': 'none', 'pattern_strength': 0}

    def calculate_enhanced_score(self, df: pd.DataFrame) -> int:
        """Score technique amélioré avec tous les nouveaux indicateurs"""
        try:
            indicators = self.calculate_indicators(df)
            if not indicators:
                return 0

            total_score = 0
            max_possible_score = 100

            # Score RSI (15 points)
            rsi_score = self._calculate_rsi_score(indicators['rsi'])
            total_score += rsi_score * 0.15

            # Score MACD (15 points)
            macd_score = self._calculate_macd_score(indicators)
            total_score += macd_score * 0.15

            # Score EMA Trend (15 points)
            ema_score = self._calculate_ema_score(indicators)
            total_score += ema_score * 0.15

            # Score ADX (10 points) - Force de tendance
            adx_score = self._calculate_adx_score(indicators)
            total_score += adx_score * 0.10

            # Score Ichimoku (10 points)
            ichimoku_score = self._calculate_ichimoku_score(indicators)
            total_score += ichimoku_score * 0.10

            # Score Volatilité (10 points)
            volatility_score = self._calculate_volatility_score(indicators)
            total_score += volatility_score * 0.10

            # Score Divergences (10 points)
            divergence_score = self._calculate_divergence_score(indicators)
            total_score += divergence_score * 0.10

            # Score Patterns (10 points)
            pattern_score = self._calculate_pattern_score(indicators)
            total_score += pattern_score * 0.10

            # Score Volume Profile (5 points)
            volume_score = self._calculate_volume_score(indicators)
            total_score += volume_score * 0.05

            final_score = max(0, min(int(total_score), 100))

            logger.debug(f"Score technique amélioré: {final_score}/100")
            return final_score

        except Exception as e:
            logger.error(f"Erreur calcul score amélioré: {e}")
            return 0

    def _calculate_adx_score(self, indicators: Dict) -> int:
        """Score ADX - Force de tendance"""
        try:
            adx = indicators.get('adx', 0)
            di_plus = indicators.get('di_plus', 0)
            di_minus = indicators.get('di_minus', 0)

            score = 0

            # Force de tendance basée sur ADX
            if adx > 50:
                score += 100  # Tendance très forte
            elif adx > 25:
                score += 80  # Tendance forte
            elif adx > 20:
                score += 60  # Tendance modérée
            else:
                score += 20  # Pas de tendance claire

            # Direction de la tendance
            if di_plus > di_minus and adx > 20:
                score = min(score + 20, 100)  # Bonus tendance haussière
            elif di_minus > di_plus and adx > 20:
                score = min(score + 20, 100)  # Bonus tendance baissière

            return min(score, 100)

        except Exception:
            return 0

    def _calculate_ichimoku_score(self, indicators: Dict) -> int:
        """Score Ichimoku Cloud"""
        try:
            price = indicators.get('current_price', 0)
            tenkan = indicators.get('tenkan_sen', 0)
            kijun = indicators.get('kijun_sen', 0)
            cloud_top = indicators.get('cloud_top', 0)
            cloud_bottom = indicators.get('cloud_bottom', 0)

            score = 0

            # Position vs Cloud
            if price > cloud_top:
                score += 40  # Au-dessus du nuage = haussier
            elif price < cloud_bottom:
                score += 40  # En-dessous du nuage = baissier
            else:
                score += 10  # Dans le nuage = neutre

            # Tenkan vs Kijun
            if tenkan > kijun:
                score += 30  # Signal haussier
            elif tenkan < kijun:
                score += 30  # Signal baissier

            # Position vs Tenkan et Kijun
            if price > tenkan and price > kijun:
                score += 30  # Configuration haussière
            elif price < tenkan and price < kijun:
                score += 30  # Configuration baissière

            return min(score, 100)

        except Exception:
            return 0

    def _calculate_divergence_score(self, indicators: Dict) -> int:
        """Score basé sur les divergences"""
        try:
            rsi_div = indicators.get('rsi_divergence', 'none')
            macd_div = indicators.get('macd_divergence', 'none')
            div_strength = indicators.get('divergence_strength', 'none')

            if div_strength == 'strong':
                return 100  # Divergences multiples
            elif div_strength == 'moderate':
                return 70  # Une divergence
            elif rsi_div != 'none' or macd_div != 'none':
                return 40  # Divergence faible
            else:
                return 20  # Pas de divergence

        except Exception:
            return 0

    def _calculate_pattern_score(self, indicators: Dict) -> int:
        """Score basé sur les patterns de chandelles"""
        try:
            pattern = indicators.get('pattern', 'none')
            strength = indicators.get('pattern_strength', 0)

            pattern_scores = {
                'bullish_engulfing': 100,
                'bearish_engulfing': 100,
                'hammer': 80,
                'shooting_star': 80,
                'doji': 60,
                'none': 20
            }

            base_score = pattern_scores.get(pattern, 20)
            return min(base_score + (strength * 10), 100)

        except Exception:
            return 0

    def _calculate_volume_score(self, indicators: Dict) -> int:
        """Score basé sur le volume profile"""
        try:
            poc = indicators.get('poc', 0)
            current_price = indicators.get('current_price', 0)
            concentration = indicators.get('volume_concentration', 0.5)

            if poc == 0:
                return 50  # Pas de données de volume

            # Distance du POC
            if current_price > 0:
                distance_from_poc = abs(current_price - poc) / current_price
                if distance_from_poc < 0.001:  # Très proche du POC
                    return 100
                elif distance_from_poc < 0.005:
                    return 80
                else:
                    return 20
            else:
                return 50

        except Exception:
            return 50

    # Méthodes existantes (héritées de votre code original)
    def _calculate_rsi_score(self, rsi: float) -> int:
        """Score RSI (conservé de votre code)"""
        try:
            if 30 <= rsi <= 70:
                distance_from_50 = abs(rsi - 50)
                score = 25 - (distance_from_50 / 20 * 10)
                return max(15, int(score))
            elif 20 <= rsi < 30 or 70 < rsi <= 80:
                return 10
            else:
                return 0
        except Exception:
            return 0

    def _calculate_macd_score(self, indicators: Dict) -> int:
        """Score MACD (conservé de votre code)"""
        try:
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            macd_histogram = indicators['macd_histogram']

            score = 0
            if macd > macd_signal:
                score += 15
                if macd_histogram > 0:
                    score += 10
            elif macd > 0 and macd_signal > 0:
                score += 8
            elif abs(macd - macd_signal) < abs(indicators.get('prev_macd_diff', 999)):
                score += 5

            return min(score, 25)
        except Exception:
            return 0

    def _calculate_ema_score(self, indicators: Dict) -> int:
        """Score EMA (conservé de votre code)"""
        try:
            price = indicators['current_price']
            ema_9 = indicators['ema_9']
            ema_21 = indicators['ema_21']
            ema_50 = indicators['ema_50']

            if price > ema_9 > ema_21 > ema_50:
                return 25
            elif price > ema_21 > ema_50:
                return 20
            elif price > ema_21:
                return 15
            elif price > ema_9:
                return 10
            elif price > ema_21 * 0.99:
                return 5
            else:
                return 0
        except Exception:
            return 0

    def _calculate_volatility_score(self, indicators: Dict) -> int:
        """Score volatilité (conservé de votre code)"""
        try:
            current_vol = indicators['volatility']
            avg_vol = indicators['avg_volatility']

            if avg_vol == 0:
                return 15

            vol_ratio = current_vol / avg_vol

            if 0.8 <= vol_ratio <= 1.2:
                return 25
            elif 0.6 <= vol_ratio <= 1.5:
                return 20
            elif 0.4 <= vol_ratio <= 2.0:
                return 15
            elif vol_ratio <= 3.0:
                return 10
            else:
                return 0
        except Exception:
            return 0

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Support/Résistance (conservé de votre code)"""
        try:
            if len(df) < self.sr_lookback:
                return float(df['price'].min()), float(df['price'].max())

            recent_prices = df['price'].tail(self.sr_lookback)
            support = float(recent_prices.min())
            resistance = float(recent_prices.max())

            return support, resistance
        except Exception as e:
            logger.error(f"Erreur calcul S/R: {e}")
            return float(df['price'].iloc[-1]), float(df['price'].iloc[-1])

    def get_enhanced_signal_direction(self, df: pd.DataFrame) -> Optional[str]:
        """Direction du signal avec tous les nouveaux indicateurs"""
        try:
            indicators = self.calculate_indicators(df)
            if not indicators:
                return None

            bullish_signals = 0
            bearish_signals = 0
            signal_weight = 0

            # Poids différents selon l'importance de l'indicateur
            weights = {
                'trend': 3,  # EMA, Ichimoku, ADX
                'momentum': 2,  # RSI, MACD, Stoch
                'reversal': 2,  # Divergences, Patterns
                'levels': 1  # S/R, Fibonacci, Volume Profile
            }

            # === SIGNAUX DE TENDANCE (Poids: 3) ===
            price = indicators['current_price']

            # EMA Trend
            if price > indicators['ema_21']:
                bullish_signals += weights['trend']
            else:
                bearish_signals += weights['trend']
            signal_weight += weights['trend']

            # Ichimoku
            cloud_top = indicators.get('cloud_top', 0)
            cloud_bottom = indicators.get('cloud_bottom', 0)
            if price > cloud_top:
                bullish_signals += weights['trend']
            elif price < cloud_bottom:
                bearish_signals += weights['trend']
            signal_weight += weights['trend']

            # ADX Direction
            adx = indicators.get('adx', 0)
            di_plus = indicators.get('di_plus', 0)
            di_minus = indicators.get('di_minus', 0)
            if adx > 20 and di_plus > di_minus:
                bullish_signals += weights['trend']
            elif adx > 20 and di_minus > di_plus:
                bearish_signals += weights['trend']
            signal_weight += weights['trend']

            # === SIGNAUX DE MOMENTUM (Poids: 2) ===

            # RSI
            rsi = indicators['rsi']
            if 30 <= rsi <= 50:
                bullish_signals += weights['momentum']
            elif 50 <= rsi <= 70:
                bearish_signals += weights['momentum']
            signal_weight += weights['momentum']

            # MACD
            if indicators['macd'] > indicators['macd_signal']:
                bullish_signals += weights['momentum']
            else:
                bearish_signals += weights['momentum']
            signal_weight += weights['momentum']

            # === SIGNAUX DE RETOURNEMENT (Poids: 2) ===

            # Divergences
            rsi_div = indicators.get('rsi_divergence', 'none')
            if rsi_div == 'bullish':
                bullish_signals += weights['reversal']
            elif rsi_div == 'bearish':
                bearish_signals += weights['reversal']
            signal_weight += weights['reversal']

            # Patterns de chandelles
            pattern = indicators.get('pattern', 'none')
            if 'bullish' in pattern or pattern == 'hammer':
                bullish_signals += weights['reversal']
            elif 'bearish' in pattern or pattern == 'shooting_star':
                bearish_signals += weights['reversal']
            signal_weight += weights['reversal']

            # === SIGNAUX DE NIVEAUX (Poids: 1) ===

            # Bollinger Bands
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.2:  # Près de la bande basse
                bullish_signals += weights['levels']
            elif bb_position > 0.8:  # Près de la bande haute
                bearish_signals += weights['levels']
            signal_weight += weights['levels']

            # Décision finale
            if signal_weight == 0:
                return None

            bullish_ratio = bullish_signals / signal_weight
            bearish_ratio = bearish_signals / signal_weight

            # Seuil plus strict pour la qualité
            min_ratio = 0.6  # 60% minimum de confluence

            if bullish_ratio >= min_ratio:
                return 'BUY'
            elif bearish_ratio >= min_ratio:
                return 'SELL'
            else:
                return None  # Signal pas assez fort

        except Exception as e:
            logger.error(f"Erreur détermination direction avancée: {e}")
            return None


# Test rapide
if __name__ == "__main__":
    print("✅ Enhanced Technical Analysis créé avec succès!")
    print("\n🔧 INSTRUCTIONS D'INTÉGRATION:")
    print("1. Remplacez votre technical_analysis.py par ce code")
    print("2. Dans signal_generator.py, remplacez TechnicalAnalysis par EnhancedTechnicalAnalysis")
    print("3. Utilisez calculate_enhanced_score() au lieu de calculate_score()")
    print("4. Utilisez get_enhanced_signal_direction() pour de meilleurs signaux")