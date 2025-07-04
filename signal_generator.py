#!/usr/bin/env python3
"""
Signal Generator RECONSTRUIT - Version 3.1 COMPLÈTE
🚀 RECONSTRUCTION TOTALE avec toutes les méthodes fonctionnelles
📊 Multi-Timeframes + IA Ensemble + Risk Management Avancé
"""

import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import time

logger = logging.getLogger(__name__)


class MultiTimeframeSignalGenerator:
    """🚀 Générateur de signaux Multi-Timeframes COMPLET"""

    def __init__(self):
        """Initialisation complète du générateur"""
        # Paramètres de configuration depuis .env
        self.risk_amount = float(os.getenv('RISK_AMOUNT', 10))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', 3))
        self.min_tech_score = int(os.getenv('MIN_TECH_SCORE', 65))
        self.min_ai_confidence = float(os.getenv('MIN_AI_CONFIDENCE', 0.70))
        self.min_confluence_score = float(os.getenv('MIN_CONFLUENCE_SCORE', 0.55))
        self.strong_confluence_score = float(os.getenv('STRONG_CONFLUENCE_SCORE', 0.80))

        # Paramètres de stop loss adaptatifs
        self.base_stop_loss_pct = 0.002  # 0.2% de base
        self.min_stop_loss_pct = 0.001  # 0.1% minimum
        self.max_stop_loss_pct = 0.01  # 1% maximum

        # Historique des signaux pour apprentissage
        self.signal_history = []
        self.max_history = 100

        # Mode avancé (peut être activé via env)
        self.advanced_mode = os.getenv('ADVANCED_SIGNAL_MODE', 'false').lower() == 'true'

        logger.info(f"🎯 Générateur MTF initialisé:")
        logger.info(f"   📊 Score technique min: {self.min_tech_score}")
        logger.info(f"   🧠 Confiance IA min: {self.min_ai_confidence}")
        logger.info(f"   🎯 Confluence min: {self.min_confluence_score}")
        logger.info(f"   ⚡ Mode avancé: {self.advanced_mode}")

    def generate_signal(self, df: pd.DataFrame, tech_score: int, ai_prediction: Dict) -> Optional[Dict]:
        """🎯 MÉTHODE PRINCIPALE - Générer un signal Multi-Timeframes"""
        try:
            if df is None or len(df) == 0:
                logger.debug("❌ Pas de données pour générer signal")
                return None

            current_price = float(df['price'].iloc[-1])
            logger.debug(
                f"🔍 Génération signal: Prix={current_price:.5f}, Tech={tech_score}, IA={ai_prediction.get('confidence', 0):.2f}")

            # ÉTAPE 1: Vérifications de base
            if not self._basic_checks(tech_score, ai_prediction):
                return None

            # ÉTAPE 2: Analyse Multi-Timeframes
            mtf_result = self._analyze_multi_timeframes(df)
            if not mtf_result or not mtf_result.get('valid_signal', False):
                logger.debug("❌ Signal rejeté par analyse MTF")
                return None

            # ÉTAPE 3: Vérifier alignement des directions
            mtf_direction = mtf_result.get('direction')
            ai_direction = 'BUY' if ai_prediction.get('direction') == 'UP' else 'SELL'

            if mtf_direction != ai_direction:
                logger.debug(f"❌ Directions non alignées: MTF={mtf_direction}, IA={ai_direction}")
                return None

            # ÉTAPE 4: Calculer score combiné
            confluence_score = mtf_result.get('confluence_score', 0)
            combined_score = self._calculate_combined_score(tech_score, ai_prediction['confidence'], confluence_score)

            # ÉTAPE 5: Vérifier confluence minimum
            if confluence_score < self.min_confluence_score:
                logger.debug(f"❌ Confluence insuffisante: {confluence_score:.1%} < {self.min_confluence_score:.1%}")
                return None

            # ÉTAPE 6: Calculer les niveaux de trading
            levels = self._calculate_trading_levels(current_price, mtf_direction, combined_score, confluence_score)
            if not levels:
                logger.debug("❌ Impossible de calculer les niveaux")
                return None

            # ÉTAPE 7: Créer le signal final
            signal = self._create_signal_object(
                mtf_direction, levels, tech_score, ai_prediction,
                combined_score, mtf_result, current_price
            )

            # ÉTAPE 8: Enregistrer pour apprentissage
            self._record_signal(signal)

            logger.info(
                f"🎯 Signal MTF généré: {mtf_direction} à {current_price:.5f} (Confluence: {confluence_score:.1%})")
            return signal

        except Exception as e:
            logger.error(f"❌ Erreur génération signal: {e}")
            return None

    def _basic_checks(self, tech_score: int, ai_prediction: Dict) -> bool:
        """Vérifications de base avant analyse MTF"""
        try:
            # Vérifier score technique
            if tech_score < self.min_tech_score:
                logger.debug(f"❌ Score technique insuffisant: {tech_score} < {self.min_tech_score}")
                return False

            # Vérifier confiance IA
            ai_confidence = ai_prediction.get('confidence', 0)
            if ai_confidence < self.min_ai_confidence:
                logger.debug(f"❌ Confiance IA insuffisante: {ai_confidence:.2f} < {self.min_ai_confidence}")
                return False

            # Vérifier que l'IA a une direction
            ai_direction = ai_prediction.get('direction')
            if ai_direction not in ['UP', 'DOWN']:
                logger.debug(f"❌ Direction IA invalide: {ai_direction}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Erreur vérifications de base: {e}")
            return False

    def _analyze_multi_timeframes(self, df: pd.DataFrame) -> Optional[Dict]:
        """🎯 Analyse Multi-Timeframes COMPLÈTE"""
        try:
            # Importer le module MTF
            from multi_timeframe_analysis import MultiTimeframeAnalysis

            mtf_analyzer = MultiTimeframeAnalysis()
            result = mtf_analyzer.multi_timeframe_analysis(df)

            if not result:
                logger.debug("❌ Échec analyse MTF")
                return None

            # Vérifier si le signal est valide selon les critères MTF
            valid_signal = mtf_analyzer.should_trade(result)
            result['valid_signal'] = valid_signal

            if valid_signal:
                confluence = result.get('confluence_score', 0)
                direction = result.get('direction', 'UNKNOWN')
                logger.debug(f"✅ MTF valide: {direction} (Confluence: {confluence:.1%})")
            else:
                logger.debug("❌ Signal MTF invalide selon critères")

            return result

        except ImportError:
            logger.warning("⚠️ Module MTF non disponible, utilisation analyse simple")
            return self._simple_mtf_fallback(df)
        except Exception as e:
            logger.error(f"❌ Erreur analyse MTF: {e}")
            return self._simple_mtf_fallback(df)

    def _simple_mtf_fallback(self, df: pd.DataFrame) -> Dict:
        """Analyse simple si MTF module non disponible"""
        try:
            from technical_analysis import TechnicalAnalysis

            ta_analyzer = TechnicalAnalysis()
            indicators = ta_analyzer.calculate_indicators(df)

            if not indicators:
                return {'valid_signal': False, 'confluence_score': 0}

            score = ta_analyzer.calculate_score(df)
            direction = ta_analyzer.get_signal_direction(df)

            # Simuler une confluence basée sur le score technique
            confluence_score = min(score / 100.0, 0.85)

            valid_signal = (
                    score >= self.min_tech_score and
                    direction is not None and
                    confluence_score >= self.min_confluence_score
            )

            return {
                'confluence_score': confluence_score,
                'direction': direction,
                'strength': 'moderate' if confluence_score >= 0.70 else 'weak',
                'valid_signal': valid_signal,
                'timeframes': {
                    'M5': {
                        'direction': direction,
                        'score': score,
                        'strength': confluence_score,
                        'trend': 'simple_fallback'
                    }
                },
                'summary': f"Mode simple M5: {direction} (Score: {score})"
            }

        except Exception as e:
            logger.error(f"❌ Erreur fallback simple: {e}")
            return {'valid_signal': False, 'confluence_score': 0}

    def _calculate_combined_score(self, tech_score: int, ai_confidence: float, confluence_score: float) -> float:
        """Calculer le score combiné avec pondération"""
        try:
            # Pondération: 25% technique, 50% IA, 25% confluence
            combined = (
                               (tech_score / 100.0 * 0.25) +
                               (ai_confidence * 0.50) +
                               (confluence_score * 0.25)
                       ) * 100

            return max(0, min(100, combined))

        except Exception as e:
            logger.error(f"❌ Erreur calcul score combiné: {e}")
            return 0.0

    def _calculate_trading_levels(self, current_price: float, direction: str,
                                  combined_score: float, confluence_score: float) -> Optional[Dict]:
        """🎯 Calculer les niveaux de trading avec adaptation intelligente"""
        try:
            # Stop loss adaptatif selon la qualité du signal
            score_factor = max(0.7, min(1.3, 1.0 - (combined_score - 70) / 100))
            confluence_factor = max(0.8, min(1.2, 1.0 - (confluence_score - 0.5) / 2))

            # Stop loss final
            final_sl = self.base_stop_loss_pct * score_factor * confluence_factor
            final_sl = max(self.min_stop_loss_pct, min(self.max_stop_loss_pct, final_sl))

            # Take profit adaptatif selon confluence
            if confluence_score >= self.strong_confluence_score:
                tp_multiplier = self.risk_reward_ratio * 1.2  # Plus ambitieux
            else:
                tp_multiplier = self.risk_reward_ratio

            take_profit_pct = final_sl * tp_multiplier

            # Calculer les prix
            entry_price = current_price

            if direction == 'BUY':
                stop_loss = entry_price * (1 - final_sl)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + final_sl)
                take_profit = entry_price * (1 - take_profit_pct)

            # Calculer les montants
            risk_amount = self.risk_amount
            reward_amount = risk_amount * tp_multiplier
            actual_ratio = tp_multiplier

            return {
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_amount': round(risk_amount, 2),
                'reward_amount': round(reward_amount, 2),
                'actual_ratio': round(actual_ratio, 2),
                'stop_loss_pct': round(final_sl * 100, 3),
                'take_profit_pct': round(take_profit_pct * 100, 3)
            }

        except Exception as e:
            logger.error(f"❌ Erreur calcul niveaux: {e}")
            return None

    def _create_signal_object(self, direction: str, levels: Dict, tech_score: int,
                              ai_prediction: Dict, combined_score: float,
                              mtf_result: Dict, current_price: float) -> Dict:
        """🎯 Créer l'objet signal final avec toutes les données"""
        try:
            confluence_score = mtf_result.get('confluence_score', 0)

            signal = {
                # Données de base
                'timestamp': datetime.now().isoformat(),
                'direction': direction,
                'entry_price': levels['entry_price'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_amount': levels['risk_amount'],
                'reward_amount': levels['reward_amount'],
                'actual_ratio': levels['actual_ratio'],

                # Scores et confiance
                'tech_score': tech_score,
                'ai_confidence': ai_prediction['confidence'],
                'combined_score': combined_score,
                'ai_direction': ai_prediction.get('direction'),

                # Multi-timeframes (structure complète)
                'multi_timeframe': {
                    'confluence_score': confluence_score,
                    'confluence_percentage': confluence_score * 100,
                    'strength': mtf_result.get('strength', 'moderate'),
                    'direction': mtf_result.get('direction'),
                    'timeframes_detail': mtf_result.get('timeframes', {}),
                    'summary': mtf_result.get('summary', ''),
                    'valid_signal': mtf_result.get('valid_signal', False)
                },

                # Niveaux en pourcentage
                'stop_loss_pct': levels['stop_loss_pct'],
                'take_profit_pct': levels['take_profit_pct'],

                # Qualité et contexte
                'signal_quality': self._assess_signal_quality(combined_score, confluence_score),
                'filter_passed': True,
                'market_conditions': self._get_market_context(current_price),

                # Métadonnées
                'generator_version': '3.1-MTF',
                'creation_time': time.time()
            }

            return signal

        except Exception as e:
            logger.error(f"❌ Erreur création objet signal: {e}")
            return {}

    def _assess_signal_quality(self, combined_score: float, confluence_score: float) -> str:
        """Évaluer la qualité du signal"""
        try:
            if combined_score >= 90 and confluence_score >= self.strong_confluence_score:
                return "PREMIUM"
            elif combined_score >= 85 and confluence_score >= 0.75:
                return "HIGH"
            elif combined_score >= 80 and confluence_score >= self.min_confluence_score:
                return "GOOD"
            elif combined_score >= 75 and confluence_score >= 0.60:
                return "AVERAGE"
            else:
                return "LOW"

        except Exception:
            return "UNKNOWN"

    def _get_market_context(self, current_price: float) -> Dict:
        """Obtenir le contexte de marché simple"""
        try:
            return {
                'current_price': current_price,
                'trend': 'unknown',
                'volatility': 'normal',
                'momentum': 'neutral',
                'timestamp': datetime.now().isoformat()
            }

        except Exception:
            return {}

    def _record_signal(self, signal: Dict):
        """Enregistrer le signal pour apprentissage"""
        try:
            # Ajouter à l'historique
            signal_record = {
                'timestamp': signal['timestamp'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'combined_score': signal['combined_score'],
                'confluence_score': signal['multi_timeframe']['confluence_score'],
                'signal_quality': signal['signal_quality']
            }

            self.signal_history.append(signal_record)

            # Garder seulement les derniers signaux
            if len(self.signal_history) > self.max_history:
                self.signal_history = self.signal_history[-self.max_history:]

            logger.debug(f"📝 Signal enregistré (Total: {len(self.signal_history)})")

        except Exception as e:
            logger.error(f"❌ Erreur enregistrement signal: {e}")

    def get_generator_stats(self) -> Dict:
        """Obtenir les statistiques du générateur"""
        return {
            'type': 'Multi-Timeframes Signal Generator',
            'version': '3.1-MTF',
            'advanced_mode': self.advanced_mode,
            'parameters': {
                'min_tech_score': self.min_tech_score,
                'min_ai_confidence': self.min_ai_confidence,
                'min_confluence_score': self.min_confluence_score,
                'strong_confluence_score': self.strong_confluence_score,
                'risk_reward_ratio': self.risk_reward_ratio,
                'base_stop_loss_pct': self.base_stop_loss_pct
            },
            'signal_history_count': len(self.signal_history),
            'filters_enabled': [
                'tech_score_filter',
                'ai_confidence_filter',
                'confluence_filter',
                'direction_alignment',
                'mtf_analysis'
            ]
        }

    def get_recent_signals_summary(self, count: int = 10) -> Dict:
        """Résumé des signaux récents"""
        try:
            if not self.signal_history:
                return {'total': 0, 'summary': 'Aucun signal généré'}

            recent = self.signal_history[-count:]

            # Statistiques
            buy_count = sum(1 for s in recent if s['direction'] == 'BUY')
            sell_count = len(recent) - buy_count

            avg_score = sum(s['combined_score'] for s in recent) / len(recent)
            avg_confluence = sum(s['confluence_score'] for s in recent) / len(recent)

            # Qualité
            quality_counts = {}
            for s in recent:
                quality = s['signal_quality']
                quality_counts[quality] = quality_counts.get(quality, 0) + 1

            return {
                'total': len(recent),
                'buy_signals': buy_count,
                'sell_signals': sell_count,
                'avg_combined_score': round(avg_score, 1),
                'avg_confluence': round(avg_confluence * 100, 1),
                'quality_distribution': quality_counts,
                'last_signal_time': recent[-1]['timestamp'] if recent else None
            }

        except Exception as e:
            logger.error(f"❌ Erreur résumé signaux: {e}")
            return {'total': 0, 'summary': 'Erreur calcul résumé'}

    def reset_history(self):
        """Réinitialiser l'historique des signaux"""
        self.signal_history = []
        logger.info("🔄 Historique des signaux réinitialisé")

    def update_parameters(self, **kwargs):
        """Mettre à jour les paramètres du générateur"""
        try:
            updated = []

            for param, value in kwargs.items():
                if hasattr(self, param):
                    old_value = getattr(self, param)
                    setattr(self, param, value)
                    updated.append(f"{param}: {old_value} → {value}")

            if updated:
                logger.info(f"🔧 Paramètres mis à jour: {', '.join(updated)}")
            else:
                logger.warning("⚠️ Aucun paramètre valide fourni")

        except Exception as e:
            logger.error(f"❌ Erreur mise à jour paramètres: {e}")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def test_signal_generator():
    """Test du générateur de signaux"""
    try:
        print("🧪 Test du générateur de signaux MTF...")

        # Créer générateur
        generator = MultiTimeframeSignalGenerator()

        # Données de test
        import numpy as np
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5min')
        prices = 1000 + np.cumsum(np.random.randn(200) * 0.5)

        df_test = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'high': prices + np.random.uniform(0, 2, 200),
            'low': prices - np.random.uniform(0, 2, 200),
            'volume': np.random.randint(100, 1000, 200)
        })

        # Test avec différents paramètres
        test_cases = [
            {'tech_score': 85, 'ai_confidence': 0.82, 'direction': 'UP'},
            {'tech_score': 75, 'ai_confidence': 0.71, 'direction': 'DOWN'},
            {'tech_score': 60, 'ai_confidence': 0.65, 'direction': 'UP'},  # Devrait échouer
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i} ---")

            ai_prediction = {
                'confidence': test_case['ai_confidence'],
                'direction': test_case['direction']
            }

            signal = generator.generate_signal(df_test, test_case['tech_score'], ai_prediction)

            if signal:
                print(f"✅ Signal généré: {signal['direction']} à {signal['entry_price']}")
                print(f"   Qualité: {signal['signal_quality']}")
                print(f"   Confluence: {signal['multi_timeframe']['confluence_score']:.1%}")
            else:
                print(f"❌ Signal rejeté (score={test_case['tech_score']}, conf={test_case['ai_confidence']})")

        # Statistiques
        stats = generator.get_generator_stats()
        print(f"\n📊 Statistiques:")
        print(f"   Version: {stats['version']}")
        print(f"   Signaux générés: {stats['signal_history_count']}")

        summary = generator.get_recent_signals_summary()
        print(f"   Résumé: {summary}")

        print("\n✅ Test terminé!")

    except Exception as e:
        print(f"❌ Erreur test: {e}")


if __name__ == "__main__":
    # Lancer le test si exécuté directement
    test_signal_generator()