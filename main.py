#!/usr/bin/env python3
"""
Bot Trading Vol75 RECONSTRUIT - Point d'entrée principal
🚀 VERSION 3.1 COMPLÈTE: IA Optimisée + Multi-Timeframes + Dashboard + Telegram
"""

import asyncio
import logging
import time
import os
import signal
import sys
import pandas as pd  # ✅ AJOUT IMPORT PANDAS MANQUANT
from datetime import datetime, timezone
from typing import Dict, Optional
from dotenv import load_dotenv

# Imports des modules optimisés
from deriv_api import DerivAPI
from technical_analysis import TechnicalAnalysis
from signal_generator import MultiTimeframeSignalGenerator
from telegram_bot import EnhancedTelegramBot
from ai_model import ImprovedEnsembleAIModel

# Import intégration dashboard
try:
    from bot_dashboard_integration import DashboardIntegration

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    logging.warning("⚠️ Module dashboard non disponible")

# Charger les variables d'environnement
load_dotenv()


def setup_logging():
    """Configuration du système de logs"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    verbose = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

    os.makedirs('logs', exist_ok=True)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [
        logging.FileHandler('logs/trading_bot_mtf_optimized.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if not verbose:
        logging.getLogger('websocket').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)

    logging.info("📋 Système de logs configuré")


logger = logging.getLogger(__name__)


class OptimizedTradingBotMTF:
    """🚀 Bot de trading Vol75 RECONSTRUIT avec MTF + Dashboard + IA Optimisée"""

    def __init__(self):
        """Initialisation du bot optimisé MTF + Dashboard"""
        # Modules principaux
        self.deriv_api = DerivAPI()
        self.technical_analysis = TechnicalAnalysis()
        self.ai_model = ImprovedEnsembleAIModel()
        self.signal_generator = MultiTimeframeSignalGenerator()
        self.telegram_bot = EnhancedTelegramBot()

        # Dashboard (optionnel)
        if DASHBOARD_AVAILABLE:
            self.dashboard = DashboardIntegration()
        else:
            self.dashboard = None

        # Variables de contrôle
        self.running = True
        self.start_time = datetime.now()

        # Compteurs et statistiques
        self.signals_today = 0
        self.premium_signals = 0
        self.high_quality_signals = 0
        self.mtf_rejections = 0
        self.ai_predictions_today = 0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.last_signal_time = 0
        self.last_signal_time_obj = None
        self.current_price = 0.0

        # Configuration depuis .env
        self.signal_interval = int(os.getenv('SIGNAL_INTERVAL', 1080))  # 18 minutes
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', 12))
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3))

        # Monitoring
        self.last_health_notification = 0
        self.health_interval = 3600  # 1 heure
        self.historical_data_loaded = False

        # Dashboard metrics
        self._last_metrics_update = 0
        self._last_price_update = 0

        # Setup signal handlers pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🚀 Bot Trading Vol75 OPTIMISÉ MTF + Dashboard initialisé")

    def _signal_handler(self, signum, frame):
        """Gestionnaire d'arrêt propre"""
        logger.info(f"Signal {signum} reçu, arrêt du bot...")
        self.running = False

    async def initialize(self):
        """Initialisation COMPLÈTE du bot"""
        try:
            logger.info("🚀 INITIALISATION Bot Vol75 Trading MTF")
            logger.info("   🎯 Mode: Multi-Timeframes Analysis")
            logger.info("   🧠 IA: Ensemble XGBoost + LightGBM")
            logger.info("   📊 Dashboard: Temps réel")

            # Vérifier configuration
            if not self._check_configuration():
                raise Exception("Configuration invalide")

            # ÉTAPE 1: Dashboard
            if self.dashboard and self.dashboard.test_connection():
                logger.info("✅ Dashboard connecté")
                await self.dashboard.send_startup_notification({
                    'version': '3.1-MTF',
                    'trading_mode': os.getenv('TRADING_MODE', 'demo'),
                    'capital': os.getenv('CAPITAL', 1000),
                    'mtf_enabled': True
                })
            else:
                logger.warning("⚠️ Dashboard non disponible")

            # ÉTAPE 2: Connexion Deriv API
            logger.info("🔌 Connexion à Deriv API...")
            await self.deriv_api.connect()
            logger.info("✅ WebSocket Deriv connecté")

            # ÉTAPE 3: Chargement données historiques
            logger.info("📊 Chargement données historiques...")
            self.historical_data_loaded = await self.deriv_api.load_historical_on_startup()

            if self.historical_data_loaded:
                data_count = len(getattr(self.deriv_api, 'data_buffer', []))
                logger.info(f"✅ Données historiques chargées: {data_count} points")
            else:
                logger.warning("⚠️ Pas de données historiques, mode temps réel")

            # ÉTAPE 4: Entraînement IA
            logger.info("🧠 Initialisation modèle IA...")
            logger.info("   🎯 Target: Précision 90%+")
            logger.info("   📊 Features: 85+ optimisées")
            logger.info("   ⏱️ Patience requise pour entraînement optimal...")

            # Notification Telegram de démarrage
            await self.telegram_bot.send_startup_message()

            # Entraînement IA (peut prendre du temps)
            training_success = self.ai_model.load_or_create_ensemble_model()

            ai_info = {}
            if training_success:
                model_info = self.ai_model.get_ensemble_model_info()
                ai_info = {
                    'model_type': model_info.get('model_type', 'EnsembleAI'),
                    'n_features': model_info.get('n_features', 85),
                    'validation_accuracy': model_info.get('validation_accuracy', 0),
                    'training_samples': model_info.get('training_samples', 0)
                }

                logger.info(f"🎯 IA OPTIMISÉE PRÊTE:")
                logger.info(f"   📊 Précision: {ai_info['validation_accuracy']:.2%}")
                logger.info(f"   🎯 Features: {ai_info['n_features']}")
                logger.info(f"   📈 Échantillons: {ai_info['training_samples']:,}")

                # Notification Telegram succès IA
                await self.telegram_bot.send_ai_training_notification(ai_info)
            else:
                logger.error("❌ ÉCHEC entraînement IA")
                await self.telegram_bot.send_error_alert("Échec entraînement IA", "CRITIQUE")

            # ÉTAPE 5: Notification démarrage complète
            await self.telegram_bot.send_mtf_startup_notification(
                historical_loaded=self.historical_data_loaded,
                ai_info=ai_info
            )

            # Dashboard update final
            if self.dashboard:
                await self.dashboard.send_system_metrics({
                    'bot_status': 'RUNNING',
                    'deriv_connected': True,
                    'telegram_connected': True,
                    'ai_accuracy': ai_info.get('validation_accuracy', 0),
                    'signals_today': 0,
                    'mtf_rejections': 0,
                    'uptime_hours': 0
                })

            logger.info("🚀 BOT COMPLÈTEMENT OPÉRATIONNEL")

        except Exception as e:
            logger.error(f"❌ ERREUR CRITIQUE INITIALISATION: {e}")
            await self.telegram_bot.send_error_alert(str(e), "INIT-CRITIQUE")
            raise

    def _check_configuration(self) -> bool:
        """Vérifier la configuration du bot"""
        required_vars = ['DERIV_APP_ID', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"❌ Variables d'environnement manquantes: {missing_vars}")
            return False

        # Log configuration
        trading_mode = os.getenv('TRADING_MODE', 'demo')
        capital = os.getenv('CAPITAL', 1000)
        risk_amount = os.getenv('RISK_AMOUNT', 10)

        logger.info(f"🔧 Configuration:")
        logger.info(f"   Mode: {trading_mode.upper()}")
        logger.info(f"   Capital: {capital}$")
        logger.info(f"   Risque/trade: {risk_amount}$")

        return True

    async def run(self):
        """🚀 Boucle principale du bot"""
        try:
            await self.initialize()
            logger.info("🔄 Démarrage de la boucle principale MTF...")

            while self.running:
                try:
                    current_time = time.time()

                    # Notification de santé périodique
                    if current_time - self.last_health_notification >= self.health_interval:
                        await self.send_health_notification()
                        self.last_health_notification = current_time

                    # Vérifier heures de trading
                    if not self._is_trading_hours():
                        await asyncio.sleep(60)  # Attendre 1 minute
                        continue

                    # Reset compteur journalier
                    self._reset_daily_counter()

                    # Vérifier limites de trading
                    if not self._can_trade():
                        await asyncio.sleep(60)  # Attendre 1 minute
                        continue

                    # 🚀 ANALYSE ET TRAITEMENT PRINCIPAL
                    await self.process_market_data()

                    # Attendre avant prochaine analyse
                    await asyncio.sleep(30)  # 30 secondes entre analyses

                except Exception as e:
                    logger.error(f"❌ Erreur dans la boucle principale: {e}")
                    await asyncio.sleep(60)  # Attendre 1 minute avant retry

        except KeyboardInterrupt:
            logger.info("Arrêt du bot demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur critique boucle principale: {e}")
            await self.telegram_bot.send_error_alert(str(e), "BOUCLE-CRITIQUE")
        finally:
            await self.cleanup()

    async def process_market_data(self):
        """🚀 TRAITEMENT PRINCIPAL des données de marché"""
        try:
            # Récupérer données récentes
            data = await self.deriv_api.get_latest_data()
            if data is None or len(data) < 100:
                logger.debug("❌ Pas assez de données pour analyse")
                return

            self.current_price = float(data['price'].iloc[-1])
            logger.debug(f"🔍 Analyse: {len(data)} points (prix: {self.current_price:.5f})")

            # Envoyer prix au dashboard (throttlé)
            if self.dashboard:
                await self._update_dashboard_price(data)

            # 📊 Analyse technique
            tech_score = self.technical_analysis.calculate_score(data)
            logger.debug(f"📊 Score technique: {tech_score}")

            # 🧠 Prédiction IA
            ai_prediction = self.ai_model.predict_ensemble(data)
            self.ai_predictions_today += 1
            logger.debug(f"🧠 IA: {ai_prediction}")

            # 🎯 Génération signal Multi-Timeframes
            signal = self.signal_generator.generate_signal(data, tech_score, ai_prediction)

            if signal:
                await self.process_signal(signal)
            else:
                self.mtf_rejections += 1
                logger.debug("❌ Signal rejeté par filtres MTF")

            # Mise à jour périodique dashboard
            if self.dashboard:
                await self._update_dashboard_metrics()

        except Exception as e:
            logger.error(f"❌ Erreur traitement données: {e}")

    async def process_signal(self, signal: Dict):
        """🎯 Traitement d'un signal validé"""
        try:
            direction = signal['direction']
            entry_price = signal['entry_price']
            signal_quality = signal.get('signal_quality', 'UNKNOWN')
            confluence_score = signal.get('multi_timeframe', {}).get('confluence_score', 0)

            logger.info(f"🎯 SIGNAL MTF GÉNÉRÉ:")
            logger.info(f"   📊 {direction} à {entry_price:.5f}")
            logger.info(f"   🏆 Qualité: {signal_quality}")
            logger.info(f"   📈 Confluence: {confluence_score:.1%}")

            # Mettre à jour statistiques
            self._update_signal_stats(signal)

            # 📱 Envoyer notification Telegram
            await self.telegram_bot.send_signal(signal)

            # 📊 Envoyer au dashboard
            if self.dashboard:
                await self.dashboard.send_signal(signal)

            # Sauvegarder signal
            self._save_signal(signal)

            # Mise à jour compteurs
            self.last_signal_time = time.time()
            self.daily_trades += 1
            self.signals_today += 1
            self.last_signal_time_obj = datetime.now()

            logger.info(
                f"📊 Stats session: Signaux={self.signals_today}, Premium={self.premium_signals}, Rejets={self.mtf_rejections}")

        except Exception as e:
            logger.error(f"❌ Erreur traitement signal: {e}")

    def _update_signal_stats(self, signal: Dict):
        """Mettre à jour les statistiques des signaux"""
        try:
            signal_quality = signal.get('signal_quality', 'UNKNOWN')

            if signal_quality == 'PREMIUM':
                self.premium_signals += 1
            elif signal_quality in ['HIGH', 'GOOD']:
                self.high_quality_signals += 1

        except Exception as e:
            logger.error(f"❌ Erreur mise à jour stats: {e}")

    async def _update_dashboard_price(self, data: pd.DataFrame):
        """Mettre à jour les prix sur le dashboard (throttlé)"""
        try:
            current_time = time.time()
            if current_time - self._last_price_update > 30:  # Toutes les 30 secondes
                price_data = {
                    'price': self.current_price,
                    'high': float(data['high'].iloc[-1]) if 'high' in data else self.current_price,
                    'low': float(data['low'].iloc[-1]) if 'low' in data else self.current_price,
                    'volume': float(data['volume'].iloc[-1]) if 'volume' in data else 1000,
                    'timestamp': datetime.now().isoformat()
                }
                await self.dashboard.send_price_data(price_data)
                self._last_price_update = current_time

        except Exception as e:
            logger.debug(f"Erreur update prix dashboard: {e}")

    async def _update_dashboard_metrics(self):
        """Mettre à jour les métriques dashboard (throttlé)"""
        try:
            current_time = time.time()
            if current_time - self._last_metrics_update > 300:  # Toutes les 5 minutes
                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                model_info = self.ai_model.get_ensemble_model_info()

                metrics = {
                    'bot_status': 'RUNNING',
                    'deriv_connected': getattr(self.deriv_api, 'connected', True),
                    'telegram_connected': True,
                    'signals_today': self.signals_today,
                    'mtf_rejections': self.mtf_rejections,
                    'ai_accuracy': model_info.get('validation_accuracy', 0),
                    'uptime_hours': uptime,
                    'premium_signals': self.premium_signals,
                    'high_quality_signals': self.high_quality_signals
                }

                await self.dashboard.send_system_metrics(metrics)
                self._last_metrics_update = current_time

        except Exception as e:
            logger.debug(f"Erreur update métriques dashboard: {e}")

    async def send_health_notification(self):
        """📱 Notification de santé MTF"""
        try:
            # Statistiques actuelles
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            model_info = self.ai_model.get_ensemble_model_info()

            # Calculs statistiques
            total_analyses = self.ai_predictions_today
            mtf_rejection_rate = (self.mtf_rejections / total_analyses * 100) if total_analyses > 0 else 0
            quality_signals = self.premium_signals + self.high_quality_signals
            quality_rate = (quality_signals / self.signals_today * 100) if self.signals_today > 0 else 0

            # Variation de prix
            data = await self.deriv_api.get_latest_data(count=12)  # ~1h de données
            price_change_1h = 0
            if data is not None and len(data) >= 12:
                price_1h_ago = float(data['price'].iloc[0])
                price_change_1h = ((self.current_price - price_1h_ago) / price_1h_ago) * 100

            bot_stats = {
                'connected': getattr(self.deriv_api, 'connected', True),
                'uptime_hours': uptime,
                'ai_accuracy': model_info.get('validation_accuracy', 0),
                'signals_today': self.signals_today,
                'premium_signals': self.premium_signals,
                'high_quality_signals': self.high_quality_signals,
                'mtf_rejections': self.mtf_rejections,
                'mtf_rejection_rate': mtf_rejection_rate,
                'quality_rate': quality_rate,
                'last_signal_time': self.last_signal_time_obj,
                'current_price': self.current_price,
                'price_change_1h': price_change_1h,
                'trading_mode': os.getenv('TRADING_MODE', 'demo'),
                'min_confluence': float(os.getenv('MIN_CONFLUENCE_SCORE', 0.55)) * 100
            }

            success = await self.telegram_bot.send_mtf_health_notification(bot_stats)

            if success:
                logger.info("📱 Notification de santé MTF envoyée")

        except Exception as e:
            logger.error(f"❌ Erreur notification santé: {e}")

    def _is_trading_hours(self) -> bool:
        """Vérifier les heures de trading"""
        current_hour = datetime.now(timezone.utc).hour
        # Éviter les heures de faible liquidité (22h-6h UTC)
        return not (22 <= current_hour or current_hour < 6)

    def _reset_daily_counter(self):
        """Reset compteur journalier"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            if self.signals_today > 0:
                logger.info(f"📅 Fin de journée - Stats MTF:")
                logger.info(f"   🎯 Signaux: {self.signals_today}")
                logger.info(f"   🏆 Premium: {self.premium_signals}")
                logger.info(f"   📈 Haute qualité: {self.high_quality_signals}")
                logger.info(f"   ❌ Rejets MTF: {self.mtf_rejections}")

            # Reset compteurs
            self.daily_trades = 0
            self.signals_today = 0
            self.premium_signals = 0
            self.high_quality_signals = 0
            self.mtf_rejections = 0
            self.ai_predictions_today = 0
            self.last_trade_date = today

    def _can_trade(self) -> bool:
        """Vérifier si on peut trader"""
        current_time = time.time()

        # Intervalle minimum entre signaux
        if current_time - self.last_signal_time < self.signal_interval:
            return False

        # Maximum de trades journaliers
        if self.daily_trades >= self.max_daily_trades:
            return False

        # Pertes consécutives
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False

        return True

    def _save_signal(self, signal: Dict):
        """Sauvegarder le signal en CSV"""
        try:
            import csv
            os.makedirs('data', exist_ok=True)

            csv_file = 'data/signals_mtf.csv'
            file_exists = os.path.exists(csv_file)

            # Aplatir les données pour CSV
            flat_signal = {
                'timestamp': signal['timestamp'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'tech_score': signal['tech_score'],
                'ai_confidence': signal['ai_confidence'],
                'combined_score': signal['combined_score'],
                'signal_quality': signal['signal_quality'],
                'confluence_score': signal.get('multi_timeframe', {}).get('confluence_score', 0),
                'mtf_strength': signal.get('multi_timeframe', {}).get('strength', 'unknown')
            }

            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flat_signal.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(flat_signal)

        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde signal: {e}")

    async def cleanup(self):
        """🧹 Nettoyage avant arrêt"""
        try:
            logger.info("🧹 Nettoyage avant arrêt...")

            # Statistiques finales
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600

            final_stats = {
                'uptime_hours': uptime,
                'signals_today': self.signals_today,
                'premium_signals': self.premium_signals,
                'mtf_rejections': self.mtf_rejections
            }

            # Notifications d'arrêt
            if self.signals_today > 0:
                await self.telegram_bot.send_mtf_daily_summary(final_stats)

            await self.telegram_bot.send_shutdown_message()

            # Dashboard
            if self.dashboard:
                await self.dashboard.send_shutdown_notification(final_stats)

            # Fermer connexions
            await self.deriv_api.disconnect()

            logger.info("✅ Nettoyage terminé")

        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")


def main():
    """🚀 Point d'entrée principal"""
    setup_logging()

    logger.info("=" * 80)
    logger.info("BOT TRADING VOL75 OPTIMISÉ MTF - DÉMARRAGE")
    logger.info("Version: 3.1 - RECONSTRUCTION COMPLÈTE")
    logger.info("🚀 Fonctionnalités:")
    logger.info("   • IA Ensemble XGBoost + LightGBM optimisée")
    logger.info("   • Analyse Multi-Timeframes M5/M15/H1")
    logger.info("   • Notifications Telegram détaillées")
    logger.info("   • Dashboard temps réel")
    logger.info("   • Confluence scoring avancé")
    logger.info("=" * 80)

    # Créer et lancer le bot
    bot = OptimizedTradingBotMTF()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Arrêt bot demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)
    finally:
        logger.info("=" * 80)
        logger.info("BOT TRADING VOL75 OPTIMISÉ MTF - ARRÊTÉ")
        logger.info("🚀 Session terminée")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()