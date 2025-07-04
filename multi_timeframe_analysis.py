#!/usr/bin/env python3
"""
Multi-Timeframe Analysis - Analyse M5, M15, H1 simultanément
Ce fichier améliore votre bot en analysant plusieurs périodes de temps
pour éviter les faux signaux et améliorer la précision
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MultiTimeframeAnalysis:
    """Classe pour analyser plusieurs timeframes simultanément"""

    def __init__(self):
        """Initialisation des timeframes"""
        # Les 3 timeframes à analyser
        self.timeframes = {
            'M5': '5min',  # 5 minutes (données actuelles)
            'M15': '15min',  # 15 minutes
            'H1': '1h'  # 1 heure
        }

        # Importance de chaque timeframe (total = 100%)
        self.tf_weights = {
            'M5': 0.2,  # 20% - court terme
            'M15': 0.3,  # 30% - moyen terme
            'H1': 0.5  # 50% - long terme (plus important)
        }

        # Seuils pour accepter un signal
        self.min_confluence_score = 0.65  # Minimum 65% d'accord entre timeframes
        self.strong_confluence_score = 0.80  # 80%+ = signal très fort

        logger.info("📊 Multi-Timeframe Analysis initialisé")

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Transformer les données 5min vers 15min ou 1h - VERSION CORRIGÉE"""
        try:
            if 'timestamp' not in df.columns:
                logger.warning("Pas de colonne timestamp")
                return df

            # 🆕 DEBUG: Vérifier les données d'entrée
            logger.info(f"🔍 DEBUG Resample {timeframe}:")
            logger.info(f"   📊 Données d'entrée: {len(df)} lignes")

            # Préparer les données
            df_copy = df.copy()

            # 🆕 VÉRIFICATION TIMESTAMPS
            logger.info(f"   📅 Premier timestamp: {df_copy['timestamp'].iloc[0]}")
            logger.info(f"   📅 Dernier timestamp: {df_copy['timestamp'].iloc[-1]}")

            # Conversion timestamp avec gestion d'erreur
            try:
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            except Exception as e:
                logger.error(f"❌ Erreur conversion timestamp: {e}")
                return df

            # 🆕 VÉRIFICATION PLAGE TEMPORELLE
            time_span = df_copy['timestamp'].iloc[-1] - df_copy['timestamp'].iloc[0]
            hours = time_span.total_seconds() / 3600
            logger.info(f"   ⏱️ Plage temporelle: {hours:.2f} heures")

            # 🆕 VÉRIFICATION SEUILS MINIMUM
            min_hours_needed = {'15min': 0.5, '1h': 1.5}
            min_hours = min_hours_needed.get(timeframe, 1.0)

            if hours < min_hours:
                logger.warning(f"⚠️ Pas assez de données pour {timeframe}: {hours:.2f}h < {min_hours}h")
                # Retourner un DataFrame vide mais avec structure correcte
                empty_df = pd.DataFrame(columns=['timestamp', 'price', 'high', 'low', 'volume'])
                return empty_df

            # Set index timestamp
            df_copy.set_index('timestamp', inplace=True)

            # 🆕 Créer high/low/volume si manquants AVANT resampling
            if 'high' not in df_copy.columns:
                df_copy['high'] = df_copy['price']
                logger.debug("   🔧 Colonne 'high' créée")
            if 'low' not in df_copy.columns:
                df_copy['low'] = df_copy['price']
                logger.debug("   🔧 Colonne 'low' créée")
            if 'volume' not in df_copy.columns:
                df_copy['volume'] = 1000  # Volume fictif
                logger.debug("   🔧 Colonne 'volume' créée")

            # 🆕 S'assurer que high >= price >= low
            df_copy['high'] = np.maximum(df_copy['high'], df_copy['price'])
            df_copy['low'] = np.minimum(df_copy['low'], df_copy['price'])

            # 🆕 CORRECTION: Utiliser les bons codes pandas pour timeframe
            timeframe_mapping = {
                '5min': '5T',  # 5 minutes
                '15min': '15T',  # 15 minutes
                '1h': '1H'  # 1 heure
            }

            pandas_timeframe = timeframe_mapping.get(timeframe, timeframe)
            logger.info(f"   🔧 Timeframe: {timeframe} -> {pandas_timeframe}")

            # Règles pour transformer les données
            rules = {
                'price': 'last',  # Prix de clôture = dernier prix
                'high': 'max',  # Plus haut = maximum
                'low': 'min',  # Plus bas = minimum
                'volume': 'sum'  # Volume = somme
            }

            # 🆕 RESAMPLE avec gestion d'erreur
            try:
                logger.info(f"   🔄 Début resampling vers {pandas_timeframe}...")
                resampled = df_copy.resample(pandas_timeframe).agg(rules)
                logger.info(f"   📊 Après resampling: {len(resampled)} bougies brutes")
            except Exception as e:
                logger.error(f"❌ Erreur pendant resampling: {e}")
                return df

            # 🆕 NETTOYAGE PLUS STRICT
            # Supprimer les lignes avec NaN
            resampled = resampled.dropna()
            logger.info(f"   🧹 Après nettoyage NaN: {len(resampled)} bougies")

            # Supprimer les bougies avec volume = 0 (périodes sans données)
            if 'volume' in resampled.columns:
                resampled = resampled[resampled['volume'] > 0]
                logger.info(f"   🧹 Après nettoyage volume: {len(resampled)} bougies")

            # Reset index pour remettre timestamp en colonne
            resampled.reset_index(inplace=True)

            # 🆕 VÉRIFICATION FINALE
            if len(resampled) > 0:
                logger.info(f"   ✅ SUCCÈS: {len(df)} -> {len(resampled)} bougies {timeframe}")
                logger.info(f"   📅 Première bougie: {resampled['timestamp'].iloc[0]}")
                logger.info(f"   📅 Dernière bougie: {resampled['timestamp'].iloc[-1]}")
            else:
                logger.warning(f"   ❌ ÉCHEC: Aucune bougie générée pour {timeframe}")

            return resampled

        except Exception as e:
            logger.error(f"❌ Erreur transformation {timeframe}: {e}")
            import traceback
            logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return df


    def analyze_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyser UN timeframe spécifique"""
        try:
            # Utiliser votre analyseur technique existant
            from technical_analysis import TechnicalAnalysis
            ta_analyzer = TechnicalAnalysis()

            # Calculer les indicateurs pour ce timeframe
            indicators = ta_analyzer.calculate_indicators(df)
            if not indicators:
                return {'score': 0, 'direction': None, 'strength': 0}

            # Score technique (0-100)
            score = ta_analyzer.calculate_score(df)

            # Direction du signal (BUY/SELL/None)
            direction = ta_analyzer.get_signal_direction(df)

            # Force du signal (0-1)
            strength = self._calculate_signal_strength(indicators)

            # Conditions de marché
            market_condition = ta_analyzer.get_market_condition(df)

            # Résultat pour ce timeframe
            result = {
                'timeframe': timeframe,
                'score': score,
                'direction': direction,
                'strength': strength,
                'trend': market_condition.get('condition', 'unknown'),
                'volatility': market_condition.get('volatility', 'normal'),
                'rsi': indicators.get('rsi', 50),
                'macd_bullish': indicators.get('macd', 0) > indicators.get('macd_signal', 0),
                'price_vs_ema21': ((indicators.get('current_price', 0) - indicators.get('ema_21', 0)) / indicators.get(
                    'ema_21', 1)) * 100 if indicators.get('ema_21', 0) > 0 else 0
            }

            logger.debug(f"{timeframe}: Score={score}, Direction={direction}, Force={strength:.2f}")
            return result

        except Exception as e:
            logger.error(f"Erreur analyse {timeframe}: {e}")
            return {'score': 0, 'direction': None, 'strength': 0, 'timeframe': timeframe}

    def _calculate_signal_strength(self, indicators: Dict) -> float:
        """Calculer la force du signal (0 à 1)"""
        try:
            strength_scores = []

            # Force RSI (plus proche de 50 = plus stable)
            rsi = indicators.get('rsi', 50)
            if 30 <= rsi <= 70:
                rsi_strength = 1.0 - abs(rsi - 50) / 20
            else:
                rsi_strength = 0.3  # Zones extrêmes = moins fiable
            strength_scores.append(rsi_strength)

            # Force MACD (amplitude de divergence)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_diff = abs(macd - macd_signal)
            macd_strength = min(macd_diff * 10000, 1.0)  # Normaliser
            strength_scores.append(macd_strength)

            # Force tendance EMA
            price = indicators.get('current_price', 0)
            ema_21 = indicators.get('ema_21', price)
            if price > 0 and ema_21 > 0:
                ema_distance = abs(price - ema_21) / ema_21
                ema_strength = min(ema_distance * 100, 1.0)
            else:
                ema_strength = 0.5
            strength_scores.append(ema_strength)

            # Volatilité (mouvement = signal plus clair)
            volatility = indicators.get('volatility', 0)
            avg_volatility = indicators.get('avg_volatility', volatility if volatility > 0 else 0.01)
            if avg_volatility > 0:
                vol_ratio = volatility / avg_volatility
                vol_strength = min(vol_ratio, 1.0) if vol_ratio <= 2 else 0.5
            else:
                vol_strength = 0.5
            strength_scores.append(vol_strength)

            # Moyenne de tous les scores
            final_strength = np.mean(strength_scores)
            return float(max(0.0, min(1.0, final_strength)))

        except Exception as e:
            logger.error(f"Erreur calcul force signal: {e}")
            return 0.5

    def multi_timeframe_analysis(self, df: pd.DataFrame) -> Dict:
        """FONCTION PRINCIPALE: Analyser tous les timeframes"""
        try:
            if len(df) < 100:
                logger.warning("Pas assez de données pour analyse multi-timeframes")
                return {
                    'confluence_score': 0,
                    'direction': None,
                    'strength': 'weak',
                    'valid_signal': False
                }

            analyses = {}

            # Analyser chaque timeframe
            for tf_name, tf_period in self.timeframes.items():
                try:
                    if tf_name == 'M5':
                        # M5 = données originales
                        tf_data = df
                    else:
                        # M15 et H1 = transformation des données
                        tf_data = self.resample_data(df, tf_period)

                    if len(tf_data) >= 50:  # Minimum pour une analyse
                        analysis = self.analyze_timeframe(tf_data, tf_name)
                        analyses[tf_name] = analysis
                    else:
                        logger.warning(f"Pas assez de données {tf_name}: {len(tf_data)}")

                except Exception as e:
                    logger.error(f"Erreur analyse {tf_name}: {e}")
                    continue

            if not analyses:
                return {
                    'confluence_score': 0,
                    'direction': None,
                    'strength': 'weak',
                    'valid_signal': False
                }

            # Calculer la confluence (accord entre timeframes)
            confluence_result = self._calculate_confluence(analyses)

            # Ajouter les détails
            confluence_result['timeframes'] = analyses
            confluence_result['summary'] = self._generate_summary(analyses, confluence_result)

            return confluence_result

        except Exception as e:
            logger.error(f"Erreur analyse multi-timeframes: {e}")
            return {
                'confluence_score': 0,
                'direction': None,
                'strength': 'weak',
                'valid_signal': False
            }

    def _calculate_confluence(self, analyses: Dict) -> Dict:
        """Calculer l'accord entre les timeframes"""
        try:
            if not analyses:
                return {
                    'confluence_score': 0,
                    'direction': None,
                    'strength': 'weak',
                    'valid_signal': False
                }

            total_weight = 0
            weighted_score = 0
            direction_votes = {'BUY': 0, 'SELL': 0, None: 0}
            strength_sum = 0

            # Calculer pour chaque timeframe
            for tf_name, analysis in analyses.items():
                weight = self.tf_weights.get(tf_name, 0.33)
                total_weight += weight

                # Score pondéré
                score = analysis.get('score', 0)
                weighted_score += score * weight

                # Votes pour la direction
                direction = analysis.get('direction')
                direction_votes[direction] += weight

                # Force moyenne
                strength = analysis.get('strength', 0)
                strength_sum += strength * weight

            # Calculs finaux
            if total_weight > 0:
                avg_score = weighted_score / total_weight
                avg_strength = strength_sum / total_weight
            else:
                avg_score = 0
                avg_strength = 0

            # Direction gagnante
            max_vote = max(direction_votes.values())
            final_direction = None
            for direction, votes in direction_votes.items():
                if votes == max_vote and direction is not None:
                    final_direction = direction
                    break

            # Score de confluence (% d'accord)
            confluence_score = max_vote / total_weight if total_weight > 0 else 0

            # Évaluer la force du signal
            if confluence_score >= self.strong_confluence_score:
                signal_strength = 'very_strong'
            elif confluence_score >= self.min_confluence_score:
                signal_strength = 'strong'
            elif confluence_score >= 0.5:
                signal_strength = 'moderate'
            else:
                signal_strength = 'weak'

            # Signal valide ?
            valid_signal = (
                    confluence_score >= self.min_confluence_score and
                    final_direction is not None
            )

            result = {
                'confluence_score': round(confluence_score, 3),
                'direction': final_direction,
                'strength': signal_strength,
                'avg_technical_score': round(avg_score, 1),
                'avg_signal_strength': round(avg_strength, 3),
                'direction_votes': direction_votes,
                'valid_signal': valid_signal
            }

            logger.info(f"🎯 Confluence: {confluence_score:.1%} pour {final_direction} ({signal_strength})")
            return result

        except Exception as e:
            logger.error(f"Erreur calcul confluence: {e}")
            return {
                'confluence_score': 0,
                'direction': None,
                'strength': 'weak',
                'valid_signal': False
            }

    def _generate_summary(self, analyses: Dict, confluence: Dict) -> str:
        """Créer un résumé textuel"""
        try:
            summary = []

            # Résumé global
            conf_score = confluence.get('confluence_score', 0) * 100
            direction = confluence.get('direction', 'AUCUNE')
            strength = confluence.get('strength', 'weak')

            summary.append(f"🎯 Confluence: {conf_score:.0f}% pour {direction} ({strength.upper()})")

            # Détail par timeframe
            for tf_name, analysis in analyses.items():
                tf_dir = analysis.get('direction', 'NEUTRE')
                tf_score = analysis.get('score', 0)
                trend = analysis.get('trend', 'unknown')

                if tf_dir == 'BUY':
                    emoji = "🟢"
                elif tf_dir == 'SELL':
                    emoji = "🔴"
                else:
                    emoji = "⚪"

                summary.append(f"{emoji} {tf_name}: {tf_dir} (Score: {tf_score}, Tendance: {trend})")

            return "\n".join(summary)

        except Exception as e:
            logger.error(f"Erreur génération résumé: {e}")
            return "Erreur génération résumé"

    def should_trade(self, confluence_result: Dict) -> bool:
        """DÉCISION: Faut-il trader ce signal ?"""
        try:
            # Vérifications de base
            if not confluence_result.get('valid_signal', False):
                logger.debug("❌ Signal rejeté: pas de signal valide")
                return False

            confluence_score = confluence_result.get('confluence_score', 0)
            direction = confluence_result.get('direction')
            strength = confluence_result.get('strength', 'weak')
            avg_score = confluence_result.get('avg_technical_score', 0)

            # Critères pour trader
            conditions = [
                confluence_score >= self.min_confluence_score,  # 65%+ de confluence
                direction in ['BUY', 'SELL'],  # Direction claire
                strength in ['strong', 'very_strong'],  # Signal fort
                avg_score >= 70  # Score technique OK
            ]

            should_trade = all(conditions)

            if should_trade:
                logger.info(f"✅ Signal VALIDÉ par multi-timeframes: {direction} "
                            f"(confluence: {confluence_score:.1%})")
            else:
                reasons = []
                if confluence_score < self.min_confluence_score:
                    reasons.append(f"confluence faible ({confluence_score:.1%})")
                if direction not in ['BUY', 'SELL']:
                    reasons.append("pas de direction claire")
                if strength not in ['strong', 'very_strong']:
                    reasons.append(f"signal faible ({strength})")
                if avg_score < 70:
                    reasons.append(f"score technique bas ({avg_score})")

                logger.debug(f"❌ Signal rejeté: {', '.join(reasons)}")

            return should_trade

        except Exception as e:
            logger.error(f"Erreur décision trading: {e}")
            return False

    def get_mtf_stats(self) -> Dict:
        """Obtenir les statistiques du module"""
        return {
            'timeframes': list(self.timeframes.keys()),
            'weights': self.tf_weights,
            'min_confluence': self.min_confluence_score,
            'strong_confluence': self.strong_confluence_score
        }


# Test rapide si le fichier est exécuté directement
if __name__ == "__main__":
    import numpy as np
    from datetime import timedelta

    print("🧪 Test Multi-Timeframe Analysis...")

    # Créer des données de test
    dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')

    # Prix avec tendance haussière + bruit
    base_price = 1000
    trend = np.linspace(0, 100, 500)  # Tendance haussière
    noise = np.random.normal(0, 5, 500)  # Bruit
    prices = base_price + trend + noise

    # DataFrame de test
    test_df = pd.DataFrame({
        'timestamp': dates,
        'price': prices
    })

    # Tester l'analyseur
    mtf = MultiTimeframeAnalysis()

    print(f"📊 Données de test: {len(test_df)} points")
    print(f"📈 Prix: {test_df['price'].iloc[0]:.2f} → {test_df['price'].iloc[-1]:.2f}")

    # Test de transformation
    for tf_name, tf_period in mtf.timeframes.items():
        if tf_name != 'M5':
            transformed = mtf.resample_data(test_df, tf_period)
            print(f"🔄 {tf_name}: {len(test_df)} → {len(transformed)} bougies")

    # Test d'analyse complète
    result = mtf.multi_timeframe_analysis(test_df)

    print("\n📋 Résultat:")
    print(f"  Confluence: {result.get('confluence_score', 0):.1%}")
    print(f"  Direction: {result.get('direction', 'None')}")
    print(f"  Force: {result.get('strength', 'None')}")
    print(f"  Signal valide: {result.get('valid_signal', False)}")

    if result.get('summary'):
        print(f"\n📊 Résumé:\n{result['summary']}")

    # Test de décision
    should_trade = mtf.should_trade(result)
    print(f"\n🎯 Décision: {'TRADER ✅' if should_trade else 'ATTENDRE ❌'}")

    print("\n✅ Test terminé!")