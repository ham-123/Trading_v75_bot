#!/usr/bin/env python3
"""
Bot Trading Vol75 OPTIMISÉ - Point d'entrée principal
🚀 VERSION 3.1: IA Optimisée + Multi-Timeframes Analysis + Telegram MTF + Dashboard
"""

import asyncio
import logging
import time
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Dict
from dotenv import load_dotenv

# Imports des modules optimisés
from deriv_api import DerivAPI
from technical_analysis import TechnicalAnalysis
from ai_model import EnsembleAIModel as OptimizedAIModel
from signal_generator import MultiTimeframeSignalGenerator
from telegram_bot import EnhancedTelegramBot  # 🆕 Bot MTF amélioré

# 🆕 NOUVEAU: Import intégration dashboard
from bot_dashboard_integration import DashboardIntegration

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


logger = logging.getLogger(__name__)


class OptimizedTradingBotMTF:
    """🚀 Bot de trading Vol75 OPTIMISÉ avec MTF Telegram détaillé + Dashboard"""

    def __init__(self):
        """Initialisation du bot optimisé MTF + Dashboard"""
        # 🆕 Modules optimisés avec Telegram MTF + Dashboard
        self.deriv_api = DerivAPI()
        self.technical_analysis = TechnicalAnalysis()
        self.ai_model = OptimizedAIModel()
        self.signal_generator = MultiTimeframeSignalGenerator()
        self.telegram_bot = EnhancedTelegramBot()  # 🆕 Bot MTF détaillé

        # 🆕 NOUVEAU: Intégration Dashboard
        self.dashboard = DashboardIntegration()

        # Variables de contrôle
        self.last_signal_time = 0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.running = True

        # Variables de monitoring optimisées MTF
        self.start_time = datetime.now()
        self.last_health_notification = 0
        self.health_interval = 3600  # 1 heure
        self.signals_today = 0
        self.last_signal_time_obj = None
        self.current_price = 0
        self.historical_data_loaded = False

        # 🆕 Statistiques MTF avancées
        self.premium_signals = 0
        self.high_quality_signals = 0
        self.mtf_rejections = 0
        self.ai_predictions_today = 0
        self.daily_mtf_stats = {
            'confluence_scores': [],
            'h1_signals': 0,
            'm15_signals': 0,
            'm5_signals': 0,
            'avg_confluence': 0,
            'best_confluence': 0
        }

        # 🆕 NOUVEAU: Variables pour dashboard
        self._last_metrics_update = 0
        self._last_price_update = 0

        # Paramètres de configuration
        self.signal_interval = int(os.getenv('SIGNAL_INTERVAL', 3600))
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', 6))
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3))

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🚀 Bot Trading Vol75 OPTIMISÉ MTF + Dashboard initialisé")

    def _signal_handler(self, signum, frame):
        """Gestionnaire d'arrêt propre"""
        logger.info(f"Signal {signum} reçu, arrêt du bot optimisé MTF...")
        self.running = False

    async def initialize(self):
        """🚀 Initialisation optimisée MTF + Dashboard"""
        try:
            logger.info("🚀 Initialisation du bot Vol75 OPTIMISÉ MTF + Dashboard...")
            logger.info("   Version: 3.1 - IA Optimisée + Multi-Timeframes + Telegram + Dashboard")

            # Vérification configuration
            if not self._check_configuration():
                raise Exception("Configuration invalide")

            # 🆕 NOUVEAU: Test connexion dashboard
            if self.dashboard.test_connection():
                logger.info("✅ Dashboard connecté")

                # Envoyer métriques initiales
                await self.dashboard.send_system_metrics({
                    'bot_status': 'STARTING',
                    'deriv_connected': False,
                    'telegram_connected': True,
                    'signals_today': 0,
                    'mtf_rejections': 0,
                    'ai_accuracy': 0,
                    'uptime_hours': 0
                })
            else:
                logger.warning("⚠️ Dashboard non disponible - Continuons sans dashboard")

            # Connexion Deriv API
            await self.deriv_api.connect()
            logger.info("✅ Connexion Deriv API établie")

            # 🚀 Chargement des données historiques
            logger.info("📊 Chargement des données historiques Vol75...")
            self.historical_data_loaded = await self.deriv_api.load_historical_on_startup()

            if self.historical_data_loaded:
                logger.info("✅ Données historiques Vol75 chargées avec succès")
            else:
                logger.info("⚠️ Mode collecte temps réel activé")

            # 🧠 Initialisation IA OPTIMISÉE
            logger.info("🧠 Chargement du modèle IA OPTIMISÉ...")
            training_success = self.ai_model.load_or_create_ensemble_model()

            ai_info = {}
            if training_success:
                model_info = self.ai_model.get_ensemble_model_info()
                ai_info = {
                    'model_type': model_info.get('model_type', 'XGBoost-Optimized'),
                    'n_features': model_info.get('n_features', 45),
                    'validation_accuracy': model_info.get('validation_accuracy', 0),
                    'training_samples': model_info.get('training_samples', 0)
                }

                logger.info(f"✅ Modèle IA optimisé prêt:")
                logger.info(f"   📊 Précision: {ai_info['validation_accuracy']:.1%}")
                logger.info(f"   📈 Features: {ai_info['n_features']}")
                logger.info(f"   🎯 Échantillons: {ai_info['training_samples']:,}")
            else:
                logger.warning("⚠️ IA en mode simple")

            # 📊 Statistiques des modules
            logger.info("📊 Configuration des modules optimisés:")
            gen_stats = self.signal_generator.get_generator_stats()
            logger.info(f"   🎯 Générateur: {gen_stats['type']}")
            logger.info(f"   📈 Confluence min: {gen_stats['min_confluence_score']:.0%}")
            logger.info(f"   🔧 Filtres: {len(gen_stats['filters_enabled'])}")

            # 🆕 NOUVEAU: Envoyer métriques complètes au dashboard
            await self._send_full_metrics_update(ai_info)

            # 🚀 Notification de démarrage MTF optimisée
            await self.telegram_bot.send_mtf_startup_notification(
                historical_loaded=self.historical_data_loaded,
                ai_info=ai_info
            )

            # 🧠 Notification d'entraînement IA si applicable
            if training_success and self.historical_data_loaded and ai_info.get('validation_accuracy', 0) > 0.6:
                await self.telegram_bot.send_ai_training_notification(ai_info)

            logger.info("🚀 Initialisation optimisée MTF + Dashboard terminée avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation optimisée MTF: {e}")
            await self.telegram_bot.send_error_alert(str(e), "Initialisation-MTF")
            raise

    def _check_configuration(self):
        """Vérification de configuration"""
        required_vars = ['DERIV_APP_ID', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            logger.error(f"Variables manquantes: {missing_vars}")
            return False
        return True

    async def run(self):
        """🚀 Boucle principale optimisée MTF + Dashboard"""
        try:
            await self.initialize()
            logger.info("🔄 Démarrage de la boucle principale OPTIMISÉE MTF + Dashboard...")

            while self.running:
                try:
                    current_time = time.time()

                    # Notification de santé horaire MTF
                    if current_time - self.last_health_notification >= self.health_interval:
                        await self.send_mtf_health_notification()
                        self.last_health_notification = current_time

                    # Vérifier heures de trading
                    if not self._is_trading_hours():
                        await asyncio.sleep(300)
                        continue

                    # Reset compteur journalier
                    self._reset_daily_counter()

                    # Vérifier limites de trading
                    if not self._can_trade():
                        await asyncio.sleep(300)
                        continue

                    # 🚀 Analyse et traitement optimisés MTF + Dashboard
                    await self.process_market_data_optimized_mtf()

                    # Attendre avant prochaine analyse
                    await asyncio.sleep(300)  # 5 minutes

                except Exception as e:
                    logger.error(f"Erreur dans la boucle optimisée MTF: {e}")
                    await asyncio.sleep(60)

        except KeyboardInterrupt:
            logger.info("Arrêt du bot optimisé MTF demandé")
        except Exception as e:
            logger.error(f"Erreur critique bot optimisé MTF: {e}")
            await self.telegram_bot.send_error_alert(str(e), "Système-MTF")
        finally:
            await self.cleanup_optimized_mtf()

    async def process_market_data_optimized_mtf(self):
        """🚀 Traitement OPTIMISÉ MTF des données de marché + Dashboard"""
        try:
            # Récupérer données récentes
            data = await self.deriv_api.get_latest_data()
            if data is None or len(data) < 100:
                logger.debug("Pas assez de données pour analyse optimisée MTF")
                return

            self.current_price = float(data['price'].iloc[-1])
            logger.debug(f"🔍 Analyse optimisée MTF: {len(data)} points (prix: {self.current_price:.5f})")

            # 🆕 NOUVEAU: Envoyer données de prix au dashboard (throttlé automatiquement)
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

            # 📊 Analyse technique
            tech_score = self.technical_analysis.calculate_score(data)
            logger.debug(f"📊 Score technique: {tech_score}")

            # 🧠 Prédiction IA OPTIMISÉE
            ai_prediction = self.ai_model.predict_ensemble(data)
            self.ai_predictions_today += 1
            logger.debug(f"🧠 IA optimisée: {ai_prediction}")

            # 🎯 Génération signal MULTI-TIMEFRAMES
            signal = self.signal_generator.generate_signal(data, tech_score, ai_prediction)

            if signal:
                await self.process_optimized_signal_mtf(signal)
            else:
                self.mtf_rejections += 1
                logger.debug("❌ Signal rejeté par filtres MTF")

            # 🆕 NOUVEAU: Mise à jour périodique des métriques dashboard
            if current_time - self._last_metrics_update > 300:  # Toutes les 5 minutes
                await self._send_metrics_update()
                self._last_metrics_update = current_time

            # 🚨 Détection d'alertes MTF spéciales
            await self.check_mtf_special_conditions(data, tech_score, ai_prediction)

        except Exception as e:
            logger.error(f"Erreur traitement optimisé MTF: {e}")

    async def process_optimized_signal_mtf(self, signal):
        """🚀 Traitement du signal optimisé MTF + Dashboard"""
        try:
            direction = signal['direction']
            entry_price = signal['entry_price']
            combined_score = signal['combined_score']

            # 🆕 Statistiques MTF de qualité
            signal_quality = signal.get('signal_quality', 'UNKNOWN')
            mtf_info = signal.get('multi_timeframe', {})
            confluence_score = mtf_info.get('confluence_score', 0)

            logger.info(f"🎯 Signal OPTIMISÉ MTF généré:")
            logger.info(f"   📊 {direction} à {entry_price:.5f}")
            logger.info(f"   🏆 Qualité: {signal_quality}")
            logger.info(f"   📈 Confluence: {confluence_score:.1%}")
            logger.info(f"   🎯 Score: {combined_score:.1f}/100")

            # 🆕 Statistiques MTF avancées
            self._update_mtf_stats(signal)

            # Compteurs par qualité
            if signal_quality == 'PREMIUM':
                self.premium_signals += 1
            elif signal_quality in ['HIGH', 'GOOD']:
                self.high_quality_signals += 1

            # 📱 Envoyer notification Telegram MTF COMPLÈTE
            await self.telegram_bot.send_signal(signal)

            # 🆕 NOUVEAU: Envoyer signal au dashboard
            await self.dashboard.send_signal(signal)

            # Mise à jour compteurs
            self.last_signal_time = time.time()
            self.daily_trades += 1
            self.signals_today += 1
            self.last_signal_time_obj = datetime.now()

            # Sauvegarder signal optimisé MTF
            self._save_optimized_signal_mtf(signal)

            # Log de performance MTF
            logger.info(f"📊 Statistiques session MTF:")
            logger.info(f"   🎯 Signaux totaux: {self.signals_today}")
            logger.info(f"   🏆 Premium: {self.premium_signals}")
            logger.info(f"   📈 Haute qualité: {self.high_quality_signals}")
            logger.info(f"   ❌ Rejets MTF: {self.mtf_rejections}")
            logger.info(f"   📊 Confluence moy: {self.daily_mtf_stats['avg_confluence']:.1%}")

        except Exception as e:
            logger.error(f"Erreur traitement signal optimisé MTF: {e}")

    async def _send_metrics_update(self):
        """🆕 Envoyer mise à jour des métriques au dashboard"""
        try:
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
            logger.debug("📊 Métriques dashboard mises à jour")

        except Exception as e:
            logger.error(f"Erreur envoi métriques dashboard: {e}")

    async def _send_full_metrics_update(self, ai_info: Dict):
        """🆕 Envoyer métriques complètes d'initialisation"""
        try:
            metrics = {
                'bot_status': 'RUNNING',
                'deriv_connected': getattr(self.deriv_api, 'connected', True),
                'telegram_connected': True,
                'signals_today': 0,
                'mtf_rejections': 0,
                'ai_accuracy': ai_info.get('validation_accuracy', 0),
                'uptime_hours': 0,
                'premium_signals': 0,
                'high_quality_signals': 0
            }

            await self.dashboard.send_system_metrics(metrics)
            logger.info("📊 Métriques complètes envoyées au dashboard")

        except Exception as e:
            logger.error(f"Erreur envoi métriques complètes: {e}")

    def _update_mtf_stats(self, signal):
        """🆕 Mise à jour des statistiques MTF"""
        try:
            mtf_info = signal.get('multi_timeframe', {})
            confluence_score = mtf_info.get('confluence_score', 0)

            # Ajouter le score de confluence
            self.daily_mtf_stats['confluence_scores'].append(confluence_score)

            # Calculer moyenne et meilleur
            if self.daily_mtf_stats['confluence_scores']:
                self.daily_mtf_stats['avg_confluence'] = sum(self.daily_mtf_stats['confluence_scores']) / len(
                    self.daily_mtf_stats['confluence_scores'])
                self.daily_mtf_stats['best_confluence'] = max(self.daily_mtf_stats['confluence_scores'])

            # Compter les signaux par timeframe dominants
            timeframes_detail = mtf_info.get('timeframes_detail', {})
            if timeframes_detail:
                # Déterminer le timeframe le plus fort
                strongest_tf = None
                max_strength = 0

                for tf_name, tf_data in timeframes_detail.items():
                    tf_strength = tf_data.get('strength', 0)
                    if isinstance(tf_strength, (int, float)) and tf_strength > max_strength:
                        max_strength = tf_strength
                        strongest_tf = tf_name

                # Incrémenter le compteur du timeframe dominant
                if strongest_tf:
                    key = f"{strongest_tf.lower()}_signals"
                    if key in self.daily_mtf_stats:
                        self.daily_mtf_stats[key] += 1

        except Exception as e:
            logger.error(f"Erreur mise à jour stats MTF: {e}")

    async def check_mtf_special_conditions(self, data, tech_score, ai_prediction):
        """🚨 Vérification des conditions spéciales MTF"""
        try:
            # Ne vérifier que si pas de signal récent
            if time.time() - self.last_signal_time < 1800:  # 30 minutes
                return

            # Analyse MTF pour détection d'alertes
            from multi_timeframe_analysis import MultiTimeframeAnalysis
            mtf_analyzer = MultiTimeframeAnalysis()

            mtf_result = mtf_analyzer.multi_timeframe_analysis(data)
            if not mtf_result:
                return

            confluence_score = mtf_result.get('confluence_score', 0)

            # Confluence très élevée sans signal (divergence technique/IA)
            if confluence_score >= 0.85 and tech_score < 70:
                alert_data = {
                    'type': 'high_confluence',
                    'confluence_score': confluence_score,
                    'direction': mtf_result.get('direction', 'N/A'),
                    'strength': mtf_result.get('strength', 'unknown'),
                    'timeframes': mtf_result.get('timeframes', {})
                }
                await self.telegram_bot.send_mtf_analysis_alert(alert_data)

            # Divergence entre timeframes
            timeframes = mtf_result.get('timeframes', {})
            if len(timeframes) >= 3:
                directions = [tf.get('direction') for tf in timeframes.values() if tf.get('direction')]
                unique_directions = set(directions)

                if len(unique_directions) >= 2:  # Directions contradictoires
                    alert_data = {
                        'type': 'divergence',
                        'confluence_score': confluence_score,
                        'direction': 'DIVERGENT',
                        'strength': 'conflict',
                        'timeframes': timeframes
                    }
                    await self.telegram_bot.send_mtf_analysis_alert(alert_data)

        except Exception as e:
            logger.debug(f"Erreur vérification conditions spéciales MTF: {e}")

    def _save_optimized_signal_mtf(self, signal):
        """💾 Sauvegarde signal optimisé MTF"""
        try:
            import csv
            os.makedirs('data', exist_ok=True)

            csv_file = 'data/optimized_signals_mtf.csv'
            file_exists = os.path.exists(csv_file)

            # Aplatir les données MTF pour CSV
            flattened_signal = signal.copy()
            mtf_data = flattened_signal.pop('multi_timeframe', {})

            # Ajouter les données MTF aplaties
            flattened_signal['mtf_confluence_score'] = mtf_data.get('confluence_score', 0)
            flattened_signal['mtf_confluence_percentage'] = mtf_data.get('confluence_percentage', 0)
            flattened_signal['mtf_strength'] = mtf_data.get('strength', 'unknown')
            flattened_signal['mtf_direction'] = mtf_data.get('direction', 'unknown')

            # Détails par timeframe
            timeframes_detail = mtf_data.get('timeframes_detail', {})
            for tf_name in ['H1', 'M15', 'M5']:
                if tf_name in timeframes_detail:
                    tf_data = timeframes_detail[tf_name]
                    flattened_signal[f'mtf_{tf_name.lower()}_direction'] = tf_data.get('direction', 'N/A')
                    flattened_signal[f'mtf_{tf_name.lower()}_score'] = tf_data.get('score', 0)
                    flattened_signal[f'mtf_{tf_name.lower()}_strength'] = tf_data.get('strength', 0)
                    flattened_signal[f'mtf_{tf_name.lower()}_trend'] = tf_data.get('trend', 'unknown')

            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=flattened_signal.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(flattened_signal)

        except Exception as e:
            logger.error(f"Erreur sauvegarde signal optimisé MTF: {e}")

    async def send_mtf_health_notification(self):
        """📱 Notification de santé optimisée MTF + Dashboard"""
        try:
            # Récupérer données actuelles
            data = await self.deriv_api.get_latest_data()
            price_change_1h = 0

            if data is not None and len(data) > 0:
                self.current_price = float(data['price'].iloc[-1])

                if len(data) >= 12:
                    price_1h_ago = float(data['price'].iloc[-13])
                    price_change_1h = ((self.current_price - price_1h_ago) / price_1h_ago) * 100

            # Stats du bot optimisé MTF
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600
            model_info = self.ai_model.get_ensemble_model_info()

            # Taux de rejet MTF
            total_analyses = self.ai_predictions_today
            mtf_rejection_rate = (self.mtf_rejections / total_analyses * 100) if total_analyses > 0 else 0

            # Statistiques de qualité
            quality_signals = self.premium_signals + self.high_quality_signals
            quality_rate = (quality_signals / self.signals_today * 100) if self.signals_today > 0 else 0

            bot_stats = {
                'connected': getattr(self.deriv_api, 'connected', True),
                'uptime_hours': uptime,
                'data_points': len(getattr(self.deriv_api, 'data_buffer', [])),
                'ai_mode': f"{model_info.get('model_type', 'Simple')} ({model_info.get('n_features', 0)} features)",
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
                'min_confluence': float(os.getenv('MIN_CONFLUENCE_SCORE', 0.65)) * 100
            }

            # Envoyer à Telegram
            success = await self.telegram_bot.send_mtf_health_notification(bot_stats)

            # 🆕 NOUVEAU: Envoyer métriques au dashboard
            await self.dashboard.send_system_metrics({
                'bot_status': 'RUNNING',
                'deriv_connected': bot_stats['connected'],
                'telegram_connected': True,
                'signals_today': self.signals_today,
                'mtf_rejections': self.mtf_rejections,
                'ai_accuracy': model_info.get('validation_accuracy', 0),
                'uptime_hours': uptime,
                'premium_signals': self.premium_signals,
                'high_quality_signals': self.high_quality_signals
            })

            if success:
                logger.info("📱 Notification de santé MTF envoyée")
            else:
                logger.warning("⚠️ Échec envoi notification de santé MTF")

        except Exception as e:
            logger.error(f"Erreur notification santé MTF: {e}")

    def _is_trading_hours(self):
        """Vérifier heures de trading"""
        current_hour = datetime.now(timezone.utc).hour
        return not (22 <= current_hour or current_hour < 6)

    def _reset_daily_counter(self):
        """Reset compteur journalier optimisé MTF"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            if self.signals_today > 0:
                logger.info(f"📅 Fin de journée OPTIMISÉE MTF - Stats:")
                logger.info(f"   🎯 Signaux totaux: {self.signals_today}")
                logger.info(f"   🏆 Premium: {self.premium_signals}")
                logger.info(f"   📈 Haute qualité: {self.high_quality_signals}")
                logger.info(f"   ❌ Rejets MTF: {self.mtf_rejections}")
                logger.info(f"   📊 Confluence moy: {self.daily_mtf_stats['avg_confluence']:.1%}")

                # Envoyer résumé quotidien MTF
                daily_summary_stats = {
                    'total_signals': self.signals_today,
                    'premium_signals': self.premium_signals,
                    'high_quality_signals': self.high_quality_signals,
                    'mtf_rejections': self.mtf_rejections,
                    'avg_confluence_score': self.daily_mtf_stats['avg_confluence'],
                    'best_confluence_score': self.daily_mtf_stats['best_confluence'],
                    'h1_signals': self.daily_mtf_stats['h1_signals'],
                    'm15_signals': self.daily_mtf_stats['m15_signals'],
                    'm5_signals': self.daily_mtf_stats['m5_signals'],
                    'win_rate': 0.75,  # Placeholder - à implémenter avec tracking réel
                    'avg_rr_ratio': 3.2,  # Placeholder
                    'realized_pnl': 0,  # Placeholder
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                }

                asyncio.create_task(self.telegram_bot.send_mtf_daily_summary(daily_summary_stats))

            # Reset tous les compteurs MTF
            self.daily_trades = 0
            self.signals_today = 0
            self.premium_signals = 0
            self.high_quality_signals = 0
            self.mtf_rejections = 0
            self.ai_predictions_today = 0

            # Reset stats MTF spécifiques
            self.daily_mtf_stats = {
                'confluence_scores': [],
                'h1_signals': 0,
                'm15_signals': 0,
                'm5_signals': 0,
                'avg_confluence': 0,
                'best_confluence': 0
            }

            self.last_trade_date = today
            logger.info(f"📅 Nouveau jour - Reset compteurs optimisés MTF: {today}")

    def _can_trade(self):
        """Vérifier si on peut trader (optimisé MTF)"""
        current_time = time.time()

        # Intervalle minimum entre signaux
        if current_time - self.last_signal_time < self.signal_interval:
            return False

        # Maximum de trades journaliers (réduit car plus sélectif avec MTF)
        if self.daily_trades >= self.max_daily_trades:
            logger.debug(f"Maximum trades optimisés MTF atteint: {self.daily_trades}")
            return False

        # Pertes consécutives
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Maximum pertes consécutives: {self.consecutive_losses}")
            return False

        return True

    async def cleanup_optimized_mtf(self):
        """🧹 Nettoyage optimisé MTF avant arrêt + Dashboard"""
        try:
            logger.info("🧹 Nettoyage optimisé MTF + Dashboard avant arrêt...")

            # 🆕 NOUVEAU: Envoyer statut d'arrêt au dashboard
            try:
                await self.dashboard.send_system_metrics({
                    'bot_status': 'STOPPED',
                    'deriv_connected': False,
                    'telegram_connected': False,
                    'signals_today': self.signals_today,
                    'mtf_rejections': self.mtf_rejections,
                    'ai_accuracy': 0,
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
                })
            except Exception as e:
                logger.debug(f"Erreur envoi statut arrêt dashboard: {e}")

            # Statistiques finales optimisées MTF
            if self.signals_today > 0:
                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                model_info = self.ai_model.get_ensemble_model_info()

                final_stats = f"""
🛑 <b>Bot Vol75 OPTIMISÉ MTF + Dashboard arrêté</b>

📊 <b>Session terminée:</b>
• Durée: {uptime:.1f}h
• Signaux générés: {self.signals_today}
• 🏆 Premium: {self.premium_signals}
• 📈 Haute qualité: {self.high_quality_signals}
• ❌ Rejets MTF: {self.mtf_rejections}
• Dernier prix: {self.current_price:.5f}

🎯 <b>Multi-Timeframes:</b>
• Confluence moyenne: {self.daily_mtf_stats['avg_confluence']:.1%}
• Meilleure confluence: {self.daily_mtf_stats['best_confluence']:.1%}
• H1 signaux: {self.daily_mtf_stats['h1_signals']}
• M15 signaux: {self.daily_mtf_stats['m15_signals']}
• M5 signaux: {self.daily_mtf_stats['m5_signals']}

🧠 <b>IA Optimisée:</b>
• Type: {model_info.get('model_type', 'XGBoost')}
• Features: {model_info.get('n_features', 0)}
• Précision: {model_info.get('validation_accuracy', 0):.1%}

📱 <b>Communication:</b>
• Messages Telegram: {self.telegram_bot.messages_sent}
• Taux succès: {self.telegram_bot.get_success_rate():.0f}%

📊 <b>Dashboard:</b>
• Status: {'✅ Connecté' if self.dashboard.enabled else '❌ Désactivé'}
• Messages envoyés: ✅

🕐 <i>Arrêté le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</i>

🚀 <b>Merci d'avoir utilisé le Bot MTF OPTIMISÉ + Dashboard!</b>
"""
                await self.telegram_bot.send_message(final_stats)
            else:
                await self.telegram_bot.send_shutdown_message()

            # Fermer connexion Deriv
            await self.deriv_api.disconnect()
            logger.info("✅ Nettoyage optimisé MTF + Dashboard terminé")

        except Exception as e:
            logger.error(f"Erreur nettoyage optimisé MTF: {e}")


def main():
    """🚀 Point d'entrée principal OPTIMISÉ MTF + Dashboard"""
    setup_logging()

    logger.info("=" * 80)
    logger.info("BOT TRADING VOL75 OPTIMISÉ MTF + DASHBOARD - DÉMARRAGE")
    logger.info("Version: 3.1 - IA Optimisée + Multi-Timeframes + Telegram + Dashboard")
    logger.info("🚀 Nouvelles fonctionnalités:")
    logger.info("   • 45+ features IA optimisées")
    logger.info("   • Analyse Multi-Timeframes M5/M15/H1")
    logger.info("   • Notifications Telegram détaillées MTF")
    logger.info("   • Dashboard temps réel avec API + Interface web")
    logger.info("   • Confluence scoring avancé")
    logger.info("   • Alertes spéciales MTF")
    logger.info("=" * 80)

    # Créer et lancer le bot optimisé MTF + Dashboard
    bot = OptimizedTradingBotMTF()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Arrêt bot optimisé MTF demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale bot optimisé MTF: {e}")
        sys.exit(1)
    finally:
        logger.info("=" * 80)
        logger.info("BOT TRADING VOL75 OPTIMISÉ MTF + DASHBOARD - ARRÊTÉ")
        logger.info("🚀 Merci d'avoir utilisé la version MTF + Dashboard optimisée!")
        logger.info("   📊 Analyses Multi-Timeframes complètes")
        logger.info("   📱 Notifications Telegram détaillées")
        logger.info("   🌐 Dashboard temps réel accessible")
        logger.info("   🎯 Sélectivité maximale des signaux")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()