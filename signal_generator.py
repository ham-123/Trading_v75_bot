#!/usr/bin/env python3
"""
Signal Generator - Logique de génération des signaux de trading
Combine analyse technique (30%) + IA (70%) pour générer signaux Vol75
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Classe pour générer les signaux de trading Vol75"""

    def __init__(self):
        """Initialisation du générateur de signaux"""
        # Paramètres de configuration
        self.risk_amount = float(os.getenv('RISK_AMOUNT', 10))
        self.risk_reward_ratio = float(os.getenv('RISK_REWARD_RATIO', 3))
        self.min_tech_score = int(os.getenv('MIN_TECH_SCORE', 70))
        self.min_ai_confidence = float(os.getenv('MIN_AI_CONFIDENCE', 0.75))

        # Pondération des signaux
        self.tech_weight = 0.3  # 30% analyse technique
        self.ai_weight = 0.7  # 70% IA

        # Paramètres de risque dynamiques
        self.base_stop_loss_pct = 0.001  # 0.1% par défaut
        self.max_stop_loss_pct = 0.005  # 0.5% maximum
        self.min_stop_loss_pct = 0.0005  # 0.05% minimum

        logger.info(
            f"🎯 Générateur de signaux initialisé (Risk: {self.risk_amount}$, Ratio: 1:{self.risk_reward_ratio})")

    def generate_signal(self, df: pd.DataFrame, tech_score: int, ai_prediction: Dict) -> Optional[Dict]:
        """Générer un signal de trading avec validation multi-timeframes"""
        try:
            if df is None or len(df) == 0:
                logger.debug("Aucune donnée fournie pour génération de signal")
                return None

            # 🆕 NOUVEAU : Analyse multi-timeframes AVANT tout
            logger.debug("🔍 Démarrage analyse multi-timeframes...")
            mtf_analyzer = MultiTimeframeAnalysis()
            confluence_result = mtf_analyzer.multi_timeframe_analysis(df)

            # 🆕 NOUVEAU : Vérifier si le signal est valide selon la confluence
            if not mtf_analyzer.should_trade(confluence_result):
                logger.debug("❌ Signal rejeté par analyse multi-timeframes")
                return None

            logger.info("✅ Signal validé par multi-timeframes!")

            # 🆕 NOUVEAU : Améliorer le score technique selon la confluence
            conf_score = confluence_result.get('confluence_score', 0)
            original_tech_score = tech_score

            if conf_score >= 0.8:  # Confluence très forte (80%+)
                tech_score = min(100, tech_score + 15)  # Bonus +15 points
                logger.debug(f"🚀 Bonus confluence très forte: {original_tech_score} → {tech_score}")
            elif conf_score >= 0.65:  # Confluence forte (65%+)
                tech_score = min(100, tech_score + 10)  # Bonus +10 points
                logger.debug(f"📈 Bonus confluence forte: {original_tech_score} → {tech_score}")

            # 🆕 NOUVEAU : Vérifier que la direction IA est alignée avec multi-timeframes
            mtf_direction = confluence_result.get('direction')
            ai_direction = 'BUY' if ai_prediction.get('direction') == 'UP' else 'SELL'

            if mtf_direction != ai_direction:
                logger.debug(f"❌ Directions non alignées: MTF={mtf_direction}, IA={ai_direction}")
                return None

            logger.debug(f"✅ Directions alignées: MTF={mtf_direction}, IA={ai_direction}")

            # === RESTE DE VOTRE CODE EXISTANT ===
            current_price = float(df['price'].iloc[-1])

            # Vérifier les conditions préliminaires (VOTRE CODE EXISTANT)
            if not self._check_basic_conditions(tech_score, ai_prediction):
                return None

            # Déterminer la direction du signal (VOTRE CODE EXISTANT)
            signal_direction = self._determine_signal_direction(df, tech_score, ai_prediction)
            if not signal_direction:
                return None

            # Calculer le score combiné (VOTRE CODE EXISTANT)
            combined_score = self._calculate_combined_score(tech_score, ai_prediction['confidence'])

            # Calculer les niveaux de prix (VOTRE CODE EXISTANT)
            levels = self._calculate_price_levels(
                current_price,
                signal_direction,
                combined_score,
                df
            )

            if not levels:
                logger.debug("Impossible de calculer les niveaux de prix")
                return None

            # 🆕 AMÉLIORER : Créer le signal avec les infos multi-timeframes
            signal = {
                'timestamp': datetime.now().isoformat(),
                'direction': signal_direction,
                'entry_price': levels['entry_price'],
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_amount': levels['risk_amount'],
                'reward_amount': levels['reward_amount'],
                'risk_reward_ratio': levels['actual_ratio'],
                'tech_score': tech_score,
                'original_tech_score': original_tech_score,  # 🆕 Score original
                'ai_confidence': round(ai_prediction['confidence'], 3),
                'ai_direction': ai_prediction['direction'],
                'combined_score': round(combined_score, 1),
                'stop_loss_pct': levels['stop_loss_pct'],
                'take_profit_pct': levels['take_profit_pct'],
                'market_conditions': self._get_market_context(df),

                # 🆕 NOUVEAU : Informations multi-timeframes
                'multi_timeframe': {
                    'confluence_score': round(conf_score, 3),
                    'confluence_percentage': round(conf_score * 100, 1),
                    'strength': confluence_result.get('strength', 'unknown'),
                    'mtf_direction': mtf_direction,
                    'summary': confluence_result.get('summary', ''),
                    'timeframes_detail': {}
                }
            }

            # 🆕 NOUVEAU : Ajouter détails par timeframe
            timeframes_data = confluence_result.get('timeframes', {})
            for tf_name, tf_data in timeframes_data.items():
                signal['multi_timeframe']['timeframes_detail'][tf_name] = {
                    'direction': tf_data.get('direction'),
                    'score': tf_data.get('score', 0),
                    'strength': tf_data.get('strength', 0),
                    'trend': tf_data.get('trend', 'unknown')
                }

            logger.info(f"🎯 Signal multi-timeframes généré: {signal_direction} à {current_price}")
            logger.info(f"   📊 Confluence: {conf_score:.1%} ({confluence_result.get('strength')})")
            logger.info(f"   📈 Score technique: {original_tech_score} → {tech_score}")
            logger.info(f"   🎯 Score combiné: {combined_score:.1f}")

            return signal

        except Exception as e:
            logger.error(f"Erreur génération signal avec multi-timeframes: {e}")
            return None

    def _check_basic_conditions(self, tech_score: int, ai_prediction: Dict) -> bool:
        """Vérifier les conditions de base pour générer un signal"""
        try:
            # Vérifier le score technique minimum
            if tech_score < self.min_tech_score:
                logger.debug(f"Score technique insuffisant: {tech_score} < {self.min_tech_score}")
                return False

            # Vérifier la confiance IA minimum
            if ai_prediction.get('confidence', 0) < self.min_ai_confidence:
                logger.debug(
                    f"Confiance IA insuffisante: {ai_prediction.get('confidence', 0)} < {self.min_ai_confidence}")
                return False

            # Vérifier que l'IA a une direction
            if not ai_prediction.get('direction'):
                logger.debug("IA n'a pas de direction claire")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur vérification conditions: {e}")
            return False

    def _determine_signal_direction(self, df: pd.DataFrame, tech_score: int, ai_prediction: Dict) -> Optional[str]:
        """Déterminer la direction du signal"""
        try:
            # Obtenir la direction de l'analyse technique
            from technical_analysis import TechnicalAnalysis
            ta = TechnicalAnalysis()
            tech_direction = ta.get_signal_direction(df)

            ai_direction = ai_prediction.get('direction')

            # Vérifier l'alignement IA/Technique
            if tech_direction and ai_direction:
                if tech_direction == 'BUY' and ai_direction == 'UP':
                    return 'BUY'
                elif tech_direction == 'SELL' and ai_direction == 'DOWN':
                    return 'SELL'
                else:
                    logger.debug(f"Directions non alignées: Tech={tech_direction}, IA={ai_direction}")

                    # Si score très élevé, privilégier l'IA
                    combined_score = self._calculate_combined_score(tech_score, ai_prediction['confidence'])
                    if combined_score > 85:
                        logger.debug("Score très élevé, privilégier direction IA")
                        return 'BUY' if ai_direction == 'UP' else 'SELL'

                    return None

            # Si seulement l'IA a une direction forte
            if ai_direction and ai_prediction['confidence'] > 0.85:
                return 'BUY' if ai_direction == 'UP' else 'SELL'

            logger.debug("Aucune direction claire déterminée")
            return None

        except Exception as e:
            logger.error(f"Erreur détermination direction: {e}")
            return None

    def _calculate_combined_score(self, tech_score: int, ai_confidence: float) -> float:
        """Calculer le score combiné (technique 30% + IA 70%)"""
        try:
            # Normaliser le score technique (0-100) et la confiance IA (0-1)
            tech_normalized = tech_score / 100.0
            ai_normalized = ai_confidence

            # Score combiné pondéré
            combined = (tech_normalized * self.tech_weight) + (ai_normalized * self.ai_weight)

            # Retourner sur une échelle de 0-100
            return combined * 100

        except Exception as e:
            logger.error(f"Erreur calcul score combiné: {e}")
            return 0.0

    def _calculate_price_levels(self, current_price: float, direction: str, combined_score: float, df: pd.DataFrame) -> \
    Optional[Dict]:
        """Calculer les niveaux d'entrée, stop loss et take profit"""
        try:
            # Calculer la volatilité récente pour ajuster les niveaux
            volatility = self._calculate_volatility(df)

            # Ajuster le stop loss basé sur la volatilité et le score
            stop_loss_pct = self._calculate_dynamic_stop_loss(volatility, combined_score)
            take_profit_pct = stop_loss_pct * self.risk_reward_ratio

            entry_price = current_price

            if direction == 'BUY':
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)

            # Calculer les montants en dollars
            risk_amount = abs(entry_price - stop_loss) * (self.risk_amount / (abs(entry_price - stop_loss)))
            reward_amount = abs(take_profit - entry_price) * (self.risk_amount / (abs(entry_price - stop_loss)))

            # Ratio risque/récompense réel
            actual_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

            levels = {
                'entry_price': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_amount': round(risk_amount, 2),
                'reward_amount': round(reward_amount, 2),
                'actual_ratio': round(actual_ratio, 2),
                'stop_loss_pct': round(stop_loss_pct * 100, 3),
                'take_profit_pct': round(take_profit_pct * 100, 3)
            }

            return levels

        except Exception as e:
            logger.error(f"Erreur calcul niveaux prix: {e}")
            return None

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculer la volatilité récente"""
        try:
            # Volatilité sur les 20 derniers points
            if len(df) >= 20:
                recent_prices = df['price'].tail(20)
                volatility = recent_prices.std() / recent_prices.mean()
            else:
                volatility = df['price'].std() / df['price'].mean()

            return float(volatility)

        except Exception as e:
            logger.error(f"Erreur calcul volatilité: {e}")
            return 0.01  # Volatilité par défaut

    def _calculate_dynamic_stop_loss(self, volatility: float, combined_score: float) -> float:
        """Calculer un stop loss dynamique basé sur la volatilité et le score"""
        try:
            # Stop loss de base
            base_sl = self.base_stop_loss_pct

            # Ajustement basé sur la volatilité
            volatility_factor = max(0.5, min(2.0, volatility * 100))  # Entre 0.5x et 2x
            volatility_adjusted_sl = base_sl * volatility_factor

            # Ajustement basé sur la confiance (score élevé = stop loss plus serré)
            confidence_factor = max(0.7, min(1.3, 1.0 - (combined_score - 70) / 100))

            # Stop loss final
            dynamic_sl = volatility_adjusted_sl * confidence_factor

            # Limiter entre min et max
            final_sl = max(self.min_stop_loss_pct, min(self.max_stop_loss_pct, dynamic_sl))

            logger.debug(
                f"Stop Loss dynamique: {final_sl * 100:.3f}% (vol: {volatility:.4f}, score: {combined_score:.1f})")

            return final_sl

        except Exception as e:
            logger.error(f"Erreur calcul stop loss dynamique: {e}")
            return self.base_stop_loss_pct

    def _get_market_context(self, df: pd.DataFrame) -> Dict:
        """Obtenir le contexte de marché pour le signal"""
        try:
            from technical_analysis import TechnicalAnalysis
            ta = TechnicalAnalysis()

            # Obtenir les conditions de marché
            market_conditions = ta.get_market_condition(df)

            # Ajouter des informations de prix
            current_price = float(df['price'].iloc[-1])
            price_change_5min = 0
            price_change_1h = 0

            if len(df) >= 5:
                price_5min_ago = float(df['price'].iloc[-6])  # 5 points plus tôt
                price_change_5min = ((current_price - price_5min_ago) / price_5min_ago) * 100

            if len(df) >= 60:
                price_1h_ago = float(df['price'].iloc[-61])  # 60 points plus tôt
                price_change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100

            context = {
                'trend': market_conditions.get('condition', 'unknown'),
                'volatility': market_conditions.get('volatility', 'normal'),
                'momentum': market_conditions.get('momentum', 'neutral'),
                'price_change_5min': round(price_change_5min, 3),
                'price_change_1h': round(price_change_1h, 3),
                'current_price': current_price
            }

            return context

        except Exception as e:
            logger.error(f"Erreur contexte marché: {e}")
            return {'trend': 'unknown', 'volatility': 'normal'}

    def validate_signal(self, signal: Dict) -> bool:
        """Valider un signal avant envoi"""
        try:
            required_fields = [
                'direction', 'entry_price', 'stop_loss', 'take_profit',
                'tech_score', 'ai_confidence', 'combined_score'
            ]

            # Vérifier que tous les champs requis sont présents
            for field in required_fields:
                if field not in signal:
                    logger.warning(f"Champ manquant dans le signal: {field}")
                    return False

            # Vérifier la cohérence des prix
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']

            if signal['direction'] == 'BUY':
                if not (sl < entry < tp):
                    logger.warning(f"Prix incohérents pour BUY: SL={sl}, Entry={entry}, TP={tp}")
                    return False
            else:  # SELL
                if not (tp < entry < sl):
                    logger.warning(f"Prix incohérents pour SELL: TP={tp}, Entry={entry}, SL={sl}")
                    return False

            # Vérifier que le ratio risk/reward est acceptable
            actual_ratio = signal.get('actual_ratio', 0)
            if actual_ratio < 1.5:  # Minimum 1:1.5
                logger.warning(f"Ratio risque/récompense trop faible: {actual_ratio}")
                return False

            logger.debug("Signal validé avec succès")
            return True

        except Exception as e:
            logger.error(f"Erreur validation signal: {e}")
            return False

    def get_generator_stats(self) -> Dict:
        """Obtenir les statistiques du générateur"""
        return {
            'risk_amount': self.risk_amount,
            'risk_reward_ratio': self.risk_reward_ratio,
            'min_tech_score': self.min_tech_score,
            'min_ai_confidence': self.min_ai_confidence,
            'tech_weight': self.tech_weight,
            'ai_weight': self.ai_weight,
            'base_stop_loss_pct': self.base_stop_loss_pct * 100
        }


# Test de la classe si exécuté directement
if __name__ == "__main__":
    import numpy as np
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)


    def test_signal_generator():
        """Test du générateur de signaux"""
        # Créer des données de test
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')

        # Prix avec tendance haussière
        base_price = 1000
        trend = np.linspace(0, 20, 100)
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + noise

        test_df = pd.DataFrame({
            'timestamp': dates,
            'price': prices
        })

        # Simulation d'analyse technique et IA
        tech_score = 75  # Score technique élevé
        ai_prediction = {
            'direction': 'UP',
            'confidence': 0.82
        }

        # Tester le générateur
        generator = SignalGenerator()

        # Générer un signal
        signal = generator.generate_signal(test_df, tech_score, ai_prediction)

        if signal:
            print("Signal généré:")
            for key, value in signal.items():
                print(f"  {key}: {value}")

            # Valider le signal
            is_valid = generator.validate_signal(signal)
            print(f"\nSignal valide: {is_valid}")
        else:
            print("Aucun signal généré")

        # Statistiques du générateur
        stats = generator.get_generator_stats()
        print(f"\nStatistiques générateur: {stats}")


    test_signal_generator()