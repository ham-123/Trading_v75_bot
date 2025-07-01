#!/usr/bin/env python3
"""
Signal Generator OPTIMISÉ - Multi-Timeframes Analysis intégré
🎯 PRÉCISION BOOST: Multi-timeframes M5+M15+H1 pour filtrer les faux signaux
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class MultiTimeframeSignalGenerator:
    """Générateur de signaux avec analyse multi-timeframes"""

    def __init__(self):
        """Initialisation du générateur optimisé"""
        # Paramètres de configuration
        self.risk_amount = float(os.getenv('RISK_AMOUNT', 10))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', 3))
        self.min_tech_score = int(os.getenv('MIN_TECH_SCORE', 70))
        self.min_ai_confidence = float(os.getenv('MIN_AI_CONFIDENCE', 0.75))

        # 🆕 SEUILS MULTI-TIMEFRAMES
        self.min_confluence_score = float(os.getenv('MIN_CONFLUENCE_SCORE', 0.65))  # 65%
        self.strong_confluence_score = float(os.getenv('STRONG_CONFLUENCE_SCORE', 0.80))  # 80%

        # Pondération des signaux
        self.tech_weight = 0.25  # 25% analyse technique
        self.ai_weight = 0.50  # 50% IA
        self.mtf_weight = 0.25  # 25% multi-timeframes

        # Paramètres de risque dynamiques
        self.base_stop_loss_pct = 0.001
        self.max_stop_loss_pct = 0.005
        self.min_stop_loss_pct = 0.0005

        logger.info(f"🎯 Générateur Multi-Timeframes initialisé (Confluence min: {self.min_confluence_score:.0%})")

    def generate_signal(self, df: pd.DataFrame, tech_score: int, ai_prediction: Dict) -> Optional[Dict]:
        """🚀 GÉNÉRATION DE SIGNAL AVEC MULTI-TIMEFRAMES"""
        try:
            if df is None or len(df) == 0:
                logger.debug("Aucune donnée fournie")
                return None

            current_price = float(df['price'].iloc[-1])
            logger.debug(f"🔍 Analyse signal à {current_price:.5f}")

            # === ÉTAPE 1: ANALYSE MULTI-TIMEFRAMES (PRIORITÉ ABSOLUE) ===
            logger.debug("📊 Démarrage analyse multi-timeframes...")
            mtf_result = self._analyze_multi_timeframes(df)

            if not mtf_result or not mtf_result.get('valid_signal', False):
                logger.debug("❌ Signal rejeté par multi-timeframes")
                return None

            confluence_score = mtf_result.get('confluence_score', 0)
            mtf_direction = mtf_result.get('direction')
            mtf_strength = mtf_result.get('strength', 'weak')

            logger.info(f"✅ Multi-timeframes validé: {mtf_direction} (confluence: {confluence_score:.1%})")

            # === ÉTAPE 2: AMÉLIORATION DU SCORE TECHNIQUE ===
            original_tech_score = tech_score

            # Bonus selon la confluence
            if confluence_score >= self.strong_confluence_score:
                tech_score = min(100, tech_score + 20)  # +20 points pour confluence très forte
                logger.debug(f"🚀 Bonus confluence très forte: {original_tech_score} → {tech_score}")
            elif confluence_score >= self.min_confluence_score:
                tech_score = min(100, tech_score + 10)  # +10 points pour confluence forte
                logger.debug(f"📈 Bonus confluence forte: {original_tech_score} → {tech_score}")

            # === ÉTAPE 3: VÉRIFICATIONS DE BASE ===
            if not self._check_basic_conditions(tech_score, ai_prediction):
                logger.debug("❌ Conditions de base non remplies")
                return None

            # === ÉTAPE 4: ALIGNEMENT DES DIRECTIONS ===
            ai_direction = 'BUY' if ai_prediction.get('direction') == 'UP' else 'SELL'

            if mtf_direction != ai_direction:
                logger.debug(f"❌ Directions non alignées: MTF={mtf_direction}, IA={ai_direction}")
                return None

            signal_direction = mtf_direction
            logger.debug(f"✅ Directions alignées: {signal_direction}")

            # === ÉTAPE 5: SCORE COMBINÉ AVEC MTF ===
            combined_score = self._calculate_combined_score_with_mtf(
                tech_score, ai_prediction['confidence'], confluence_score
            )

            # === ÉTAPE 6: FILTRES DE QUALITÉ AVANCÉS ===
            if not self._quality_filters(df, combined_score, confluence_score):
                logger.debug("❌ Filtres de qualité non passés")
                return None

            # === ÉTAPE 7: CALCUL DES NIVEAUX DE PRIX ===
            levels = self._calculate_price_levels_advanced(
                current_price, signal_direction, combined_score, confluence_score, df
            )

            if not levels:
                logger.debug("❌ Impossible de calculer les niveaux")
                return None

            # === ÉTAPE 8: CRÉATION DU SIGNAL OPTIMISÉ ===
            signal = {
                'timestamp': datetime.now().isoformat(),
                'direction': signal_direction,
                'entry_price': levels['entry_price'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_amount': levels['risk_amount'],
                'reward_amount': levels['reward_amount'],
                'risk_reward_ratio': levels['actual_ratio'],

                # Scores
                'tech_score': tech_score,
                'original_tech_score': original_tech_score,
                'ai_confidence': round(ai_prediction['confidence'], 3),
                'ai_direction': ai_prediction['direction'],
                'combined_score': round(combined_score, 1),

                # Multi-timeframes
                'multi_timeframe': {
                    'confluence_score': round(confluence_score, 3),
                    'confluence_percentage': round(confluence_score * 100, 1),
                    'strength': mtf_strength,
                    'direction': mtf_direction,
                    'timeframes_detail': mtf_result.get('timeframes', {}),
                    'summary': mtf_result.get('summary', '')
                },

                # Niveaux
                'stop_loss_pct': levels['stop_loss_pct'],
                'take_profit_pct': levels['take_profit_pct'],
                'market_conditions': self._get_market_context(df),

                # Métadonnées
                'signal_quality': self._assess_signal_quality(combined_score, confluence_score),
                'filter_passed': True
            }

            logger.info(f"🎯 Signal MULTI-TIMEFRAMES généré:")
            logger.info(f"   📊 Direction: {signal_direction}")
            logger.info(f"   🎯 Score combiné: {combined_score:.1f}/100")
            logger.info(f"   📈 Confluence: {confluence_score:.1%} ({mtf_strength})")
            logger.info(f"   💰 R:R: 1:{levels['actual_ratio']:.1f}")

            return signal

        except Exception as e:
            logger.error(f"Erreur génération signal MTF: {e}")
            return None

    def _analyze_multi_timeframes(self, df: pd.DataFrame) -> Optional[Dict]:
        """🚀 ANALYSE MULTI-TIMEFRAMES M5+M15+H1"""
        try:
            # 🆕 SEUIL ADAPTATIF selon les données disponibles
            min_data_required = 200
            if len(df) < min_data_required:
                logger.debug(f"Pas assez de données pour MTF: {len(df)} < {min_data_required}")

                # 🆕 MODE DÉGRADÉ: Analyse simple sur M5 uniquement
                if len(df) >= 50:
                    logger.debug("Mode dégradé MTF: M5 seulement")
                    return self._simple_mtf_analysis(df)
                else:
                    return None

            from multi_timeframe_analysis import MultiTimeframeAnalysis
            mtf_analyzer = MultiTimeframeAnalysis()

            # Analyse complète multi-timeframes
            result = mtf_analyzer.multi_timeframe_analysis(df)

            # Vérifier si le signal est tradable
            should_trade = mtf_analyzer.should_trade(result)
            result['valid_signal'] = should_trade

            return result

        except Exception as e:
            logger.error(f"Erreur analyse MTF: {e}")
            # 🆕 FALLBACK vers mode simple
            return self._simple_mtf_analysis(df) if len(df) >= 50 else None

    def _simple_mtf_analysis(self, df: pd.DataFrame) -> Dict:
        """🆕 ANALYSE MTF SIMPLIFIÉE pour données limitées"""
        try:
            from technical_analysis import TechnicalAnalysis
            ta_analyzer = TechnicalAnalysis()

            # Analyse technique simple
            indicators = ta_analyzer.calculate_indicators(df)
            if not indicators:
                return {'valid_signal': False, 'confluence_score': 0}

            score = ta_analyzer.calculate_score(df)
            direction = ta_analyzer.get_signal_direction(df)

            # Simuler une confluence basée sur le score technique seul
            confluence_score = min(score / 100.0, 0.85)  # Max 85% en mode simple

            # Signal valide si score décent
            valid_signal = score >= 70 and direction is not None and confluence_score >= 0.60

            result = {
                'confluence_score': confluence_score,
                'direction': direction,
                'strength': 'moderate' if confluence_score >= 0.70 else 'weak',
                'valid_signal': valid_signal,
                'timeframes': {
                    'M5': {
                        'direction': direction,
                        'score': score,
                        'strength': confluence_score,
                        'trend': 'simple_mode'
                    }
                },
                'summary': f"Mode simple M5: {direction} (Score: {score})"
            }

            logger.debug(f"MTF Simple: {direction} (conf: {confluence_score:.1%})")
            return result

        except Exception as e:
            logger.error(f"Erreur MTF simple: {e}")
            return {'valid_signal': False, 'confluence_score': 0}

    def _check_basic_conditions(self, tech_score: int, ai_prediction: Dict) -> bool:
        """Vérifier les conditions de base (améliorées)"""
        try:
            # Score technique minimum (abaissé car on a MTF)
            if tech_score < max(self.min_tech_score - 10, 60):
                logger.debug(f"Score technique insuffisant: {tech_score}")
                return False

            # Confiance IA minimum (abaissée car on a MTF)
            min_ai_conf = max(self.min_ai_confidence - 0.05, 0.70)
            if ai_prediction.get('confidence', 0) < min_ai_conf:
                logger.debug(f"Confiance IA insuffisante: {ai_prediction.get('confidence', 0)}")
                return False

            # Direction IA claire
            if not ai_prediction.get('direction'):
                logger.debug("IA sans direction claire")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification conditions: {e}")
            return False

    def _calculate_combined_score_with_mtf(self, tech_score: int, ai_confidence: float,
                                           confluence_score: float) -> float:
        """🆕 SCORE COMBINÉ avec Multi-Timeframes"""
        try:
            # Normalisation
            tech_normalized = tech_score / 100.0
            ai_normalized = ai_confidence
            mtf_normalized = confluence_score

            # Score pondéré avec MTF
            combined = (
                    (tech_normalized * self.tech_weight) +
                    (ai_normalized * self.ai_weight) +
                    (mtf_normalized * self.mtf_weight)
            )

            # Bonus si toutes les sources sont fortes
            if tech_normalized > 0.8 and ai_normalized > 0.8 and mtf_normalized > 0.8:
                combined = min(1.0, combined + 0.05)  # Bonus 5%

            return combined * 100

        except Exception as e:
            logger.error(f"Erreur calcul score combiné MTF: {e}")
            return 0.0

    def _quality_filters(self, df: pd.DataFrame, combined_score: float, confluence_score: float) -> bool:
        """🆕 FILTRES DE QUALITÉ AVANCÉS"""
        try:
            # Filtre 1: Score combiné minimum
            if combined_score < 75:
                logger.debug(f"Score combiné trop faible: {combined_score}")
                return False

            # Filtre 2: Confluence minimum renforcée
            if confluence_score < self.min_confluence_score:
                logger.debug(f"Confluence insuffisante: {confluence_score:.1%}")
                return False

            # Filtre 3: Heures de trading (éviter 22h-6h UTC)
            current_hour = datetime.now().hour
            if 22 <= current_hour or current_hour < 6:
                logger.debug(f"Heure de faible liquidité: {current_hour}h")
                return False

            # Filtre 4: Volatilité extrême
            if len(df) >= 20:
                recent_volatility = df['price'].tail(20).std() / df['price'].tail(20).mean()
                if recent_volatility > 0.05:  # Plus de 5% de volatilité
                    logger.debug(f"Volatilité excessive: {recent_volatility:.3f}")
                    return False

            # Filtre 5: Spread approximatif (éviter les périodes à fort spread)
            if len(df) >= 10:
                recent_high_low = (df['high'].tail(10).mean() - df['low'].tail(10).mean()) / df['price'].tail(10).mean()
                if recent_high_low > 0.003:  # Plus de 0.3% de spread moyen
                    logger.debug(f"Spread élevé détecté: {recent_high_low:.4f}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Erreur filtres qualité: {e}")
            return True  # En cas d'erreur, laisser passer

    def _calculate_price_levels_advanced(self, current_price: float, direction: str,
                                         combined_score: float, confluence_score: float, df: pd.DataFrame) -> Optional[
        Dict]:
        """🆕 CALCUL AVANCÉ DES NIVEAUX avec MTF"""
        try:
            # Volatilité récente
            volatility = self._calculate_volatility(df)

            # Stop loss dynamique basé sur score ET confluence
            base_sl = self._calculate_dynamic_stop_loss_advanced(volatility, combined_score, confluence_score)

            # Take profit adaptatif selon la confluence
            if confluence_score >= self.strong_confluence_score:
                # Signal très fort = TP plus ambitieux
                tp_multiplier = self.risk_reward_ratio * 1.2
            elif confluence_score >= self.min_confluence_score:
                # Signal fort = TP normal
                tp_multiplier = self.risk_reward_ratio
            else:
                # Signal moyen = TP conservateur
                tp_multiplier = self.risk_reward_ratio * 0.8

            take_profit_pct = base_sl * tp_multiplier
            entry_price = current_price

            if direction == 'BUY':
                stop_loss = entry_price * (1 - base_sl)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + base_sl)
                take_profit = entry_price * (1 - take_profit_pct)

            # Calculs financiers
            risk_amount = abs(entry_price - stop_loss) * (self.risk_amount / abs(entry_price - stop_loss))
            reward_amount = abs(take_profit - entry_price) * (self.risk_amount / abs(entry_price - stop_loss))
            actual_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            return {
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_amount': round(risk_amount, 2),
                'reward_amount': round(reward_amount, 2),
                'actual_ratio': round(actual_ratio, 2),
                'stop_loss_pct': round(base_sl * 100, 3),
                'take_profit_pct': round(take_profit_pct * 100, 3),
                'tp_multiplier_used': round(tp_multiplier, 2)
            }

        except Exception as e:
            logger.error(f"Erreur calcul niveaux avancés: {e}")
            return None

    def _calculate_dynamic_stop_loss_advanced(self, volatility: float, combined_score: float,
                                              confluence_score: float) -> float:
        """🆕 STOP LOSS DYNAMIQUE avec confluence"""
        try:
            # Base
            base_sl = self.base_stop_loss_pct

            # Ajustement volatilité
            vol_factor = max(0.6, min(1.8, volatility * 120))
            volatility_adjusted_sl = base_sl * vol_factor

            # Ajustement score (score élevé = SL plus serré)
            score_factor = max(0.7, min(1.2, 1.0 - (combined_score - 70) / 150))

            # 🆕 Ajustement confluence (confluence élevée = SL plus serré)
            confluence_factor = max(0.8, min(1.1, 1.0 - (confluence_score - 0.5) / 2))

            # Calcul final
            final_sl = volatility_adjusted_sl * score_factor * confluence_factor
            final_sl = max(self.min_stop_loss_pct, min(self.max_stop_loss_pct, final_sl))

            logger.debug(
                f"SL dynamique: {final_sl * 100:.3f}% (vol:{volatility:.4f}, score:{combined_score:.1f}, conf:{confluence_score:.2f})")
            return final_sl

        except Exception as e:
            logger.error(f"Erreur SL dynamique: {e}")
            return self.base_stop_loss_pct

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculer la volatilité récente"""
        try:
            if len(df) >= 20:
                recent_prices = df['price'].tail(20)
                volatility = recent_prices.std() / recent_prices.mean()
            else:
                volatility = df['price'].std() / df['price'].mean()
            return float(volatility)
        except Exception:
            return 0.01

    def _get_market_context(self, df: pd.DataFrame) -> Dict:
        """Contexte de marché enrichi"""
        try:
            from technical_analysis import TechnicalAnalysis
            ta = TechnicalAnalysis()
            market_conditions = ta.get_market_condition(df)

            current_price = float(df['price'].iloc[-1])
            context = {
                'trend': market_conditions.get('condition', 'unknown'),
                'volatility': market_conditions.get('volatility', 'normal'),
                'momentum': market_conditions.get('momentum', 'neutral'),
                'current_price': current_price
            }

            # Variations de prix
            if len(df) >= 5:
                price_5min_ago = float(df['price'].iloc[-6])
                context['price_change_5min'] = round(((current_price - price_5min_ago) / price_5min_ago) * 100, 3)

            if len(df) >= 60:
                price_1h_ago = float(df['price'].iloc[-61])
                context['price_change_1h'] = round(((current_price - price_1h_ago) / price_1h_ago) * 100, 3)

            return context

        except Exception as e:
            logger.error(f"Erreur contexte marché: {e}")
            return {'trend': 'unknown', 'volatility': 'normal'}

    def _assess_signal_quality(self, combined_score: float, confluence_score: float) -> str:
        """🆕 ÉVALUATION DE LA QUALITÉ DU SIGNAL"""
        try:
            # Critères de qualité
            if combined_score >= 90 and confluence_score >= self.strong_confluence_score:
                return "PREMIUM"  # Signal premium
            elif combined_score >= 85 and confluence_score >= 0.75:
                return "HIGH"  # Haute qualité
            elif combined_score >= 80 and confluence_score >= self.min_confluence_score:
                return "GOOD"  # Bonne qualité
            elif combined_score >= 75 and confluence_score >= 0.60:
                return "AVERAGE"  # Qualité moyenne
            else:
                return "LOW"  # Qualité faible (ne devrait pas arriver grâce aux filtres)

        except Exception:
            return "UNKNOWN"

    def validate_signal(self, signal: Dict) -> bool:
        """Validation avancée du signal MTF"""
        try:
            required_fields = [
                'direction', 'entry_price', 'stop_loss', 'take_profit',
                'tech_score', 'ai_confidence', 'combined_score', 'multi_timeframe'
            ]

            # Champs requis
            for field in required_fields:
                if field not in signal:
                    logger.warning(f"Champ manquant: {field}")
                    return False

            # Cohérence des prix
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']

            if signal['direction'] == 'BUY':
                if not (sl < entry < tp):
                    logger.warning(f"Prix incohérents BUY: SL={sl}, Entry={entry}, TP={tp}")
                    return False
            else:  # SELL
                if not (tp < entry < sl):
                    logger.warning(f"Prix incohérents SELL: TP={tp}, Entry={entry}, SL={sl}")
                    return False

            # Ratio R:R minimum
            actual_ratio = signal.get('actual_ratio', 0)
            if actual_ratio < 1.5:
                logger.warning(f"Ratio R:R trop faible: {actual_ratio}")
                return False

            # 🆕 Validation MTF
            mtf_data = signal.get('multi_timeframe', {})
            confluence_score = mtf_data.get('confluence_score', 0)

            if confluence_score < self.min_confluence_score:
                logger.warning(f"Confluence insuffisante: {confluence_score:.1%}")
                return False

            # Qualité minimum
            signal_quality = signal.get('signal_quality', 'UNKNOWN')
            if signal_quality in ['LOW', 'UNKNOWN']:
                logger.warning(f"Qualité de signal insuffisante: {signal_quality}")
                return False

            logger.debug("✅ Signal MTF validé avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur validation signal MTF: {e}")
            return False

    def get_generator_stats(self) -> Dict:
        """Statistiques du générateur MTF"""
        return {
            'type': 'MultiTimeframeSignalGenerator',
            'version': '2.0-MTF',
            'risk_amount': self.risk_amount,
            'risk_reward_ratio': self.risk_reward_ratio,
            'min_tech_score': self.min_tech_score,
            'min_ai_confidence': self.min_ai_confidence,
            'min_confluence_score': self.min_confluence_score,
            'strong_confluence_score': self.strong_confluence_score,
            'weights': {
                'technical': self.tech_weight,
                'ai': self.ai_weight,
                'multi_timeframe': self.mtf_weight
            },
            'filters_enabled': [
                'basic_conditions',
                'multi_timeframe_confluence',
                'quality_filters',
                'trading_hours',
                'volatility_check',
                'spread_check'
            ]
        }


# Wrapper pour compatibilité avec le code existant
class SignalGenerator(MultiTimeframeSignalGenerator):
    """Wrapper pour compatibilité avec le code existant"""
    pass


# Test du générateur MTF
if __name__ == "__main__":
    import numpy as np
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)


    def test_mtf_signal_generator():
        """Test du générateur multi-timeframes"""

        # Données de test avec tendance claire
        dates = pd.date_range(start='2024-01-01', periods=500, freq='5min')
        base_price = 1000

        # Tendance haussière progressive
        trend = np.linspace(0, 100, 500)
        noise = np.random.normal(0, 3, 500)
        prices = base_price + trend + noise

        test_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'high': prices + np.random.uniform(0, 2, 500),
            'low': prices - np.random.uniform(0, 2, 500),
            'volume': np.random.randint(500, 1500, 500)
        })

        # Simulation signaux forts
        tech_score = 82  # Score technique élevé
        ai_prediction = {
            'direction': 'UP',
            'confidence': 0.87  # Confiance IA élevée
        }

        # Test du générateur MTF
        generator = MultiTimeframeSignalGenerator()

        print("🧪 Test du générateur Multi-Timeframes...")
        print(f"📊 Données: {len(test_df)} points")
        print(f"📈 Tendance: {test_df['price'].iloc[0]:.2f} → {test_df['price'].iloc[-1]:.2f}")
        print(f"🎯 Score technique: {tech_score}")
        print(f"🧠 IA: {ai_prediction['direction']} ({ai_prediction['confidence']:.1%})")
        print()

        # Générer signal
        signal = generator.generate_signal(test_df, tech_score, ai_prediction)

        if signal:
            print("✅ Signal Multi-Timeframes généré:")
            print(f"   Direction: {signal['direction']}")
            print(f"   Prix entrée: {signal['entry_price']}")
            print(f"   Stop Loss: {signal['stop_loss']} ({signal['stop_loss_pct']:.2f}%)")
            print(f"   Take Profit: {signal['take_profit']} ({signal['take_profit_pct']:.2f}%)")
            print(f"   Ratio R:R: 1:{signal['actual_ratio']:.1f}")
            print(f"   Score combiné: {signal['combined_score']:.1f}/100")
            print()

            # Détails MTF
            mtf_info = signal.get('multi_timeframe', {})
            print("📊 Multi-Timeframes:")
            print(f"   Confluence: {mtf_info.get('confluence_percentage', 0):.0f}%")
            print(f"   Force: {mtf_info.get('strength', 'unknown').upper()}")
            print(f"   Direction: {mtf_info.get('direction', 'unknown')}")
            print()

            print(f"🏆 Qualité: {signal.get('signal_quality', 'UNKNOWN')}")

            # Validation
            is_valid = generator.validate_signal(signal)
            print(f"✅ Signal valide: {is_valid}")

        else:
            print("❌ Aucun signal généré (filtres MTF)")

        # Stats du générateur
        stats = generator.get_generator_stats()
        print(f"\n📊 Statistiques générateur:")
        print(f"   Type: {stats['type']}")
        print(f"   Version: {stats['version']}")
        print(f"   Confluence min: {stats['min_confluence_score']:.0%}")
        print(f"   Filtres actifs: {len(stats['filters_enabled'])}")


    test_mtf_signal_generator()