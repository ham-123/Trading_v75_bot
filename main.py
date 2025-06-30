#!/usr/bin/env python3
"""
Bot Trading Vol75 - Point d'entrée principal
Orchestration de tous les composants du système de trading automatisé
Avec chargement de données historiques et notifications de santé
"""

import asyncio
import logging
import time
import os
import signal
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Imports des modules du bot
from deriv_api import DerivAPI
from technical_analysis import TechnicalAnalysis
from ai_model import AIModel
from signal_generator import SignalGenerator
from telegram_bot import TelegramBot

# Charger les variables d'environnement
load_dotenv()


# Configuration des logs
def setup_logging():
    """Configuration du système de logs"""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    verbose = os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

    # Créer le dossier logs s'il n'existe pas
    os.makedirs('logs', exist_ok=True)

    # Format des logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configuration des handlers
    handlers = [
        logging.FileHandler('logs/trading_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]

    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Réduire le bruit des autres libs si pas en mode verbose
    if not verbose:
        logging.getLogger('websocket').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class TradingBot:
    """Classe principale du bot de trading Vol75"""

    def __init__(self):
        """Initialisation du bot"""
        self.deriv_api = DerivAPI()
        self.technical_analysis = TechnicalAnalysis()
        self.ai_model = AIModel()
        self.signal_generator = SignalGenerator()
        self.telegram_bot = TelegramBot()

        # Variables de contrôle trading
        self.last_signal_time = 0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.running = True

        # Variables de santé et monitoring
        self.start_time = datetime.now()
        self.last_health_notification = 0
        self.health_interval = 3600  # 1 heure en secondes
        self.signals_today = 0
        self.last_signal_time_obj = None
        self.current_price = 0
        self.historical_data_loaded = False

        # Paramètres de configuration
        self.signal_interval = int(os.getenv('SIGNAL_INTERVAL', 3600))  # 1h par défaut
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', 8))
        self.max_consecutive_losses = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3))

        # Setup signal handlers pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Gestionnaire pour arrêt propre du bot"""
        logger.info(f"Signal {signum} reçu, arrêt du bot...")
        self.running = False

    async def initialize(self):
        """Initialisation des composants avec chargement historique"""
        try:
            logger.info("🚀 Initialisation du bot Vol75...")

            # Vérifier la configuration
            if not self._check_configuration():
                raise Exception("Configuration invalide")

            # Connexion à Deriv API
            await self.deriv_api.connect()
            logger.info("✅ Connexion Deriv API établie")

            # 🆕 CHARGER LES DONNÉES HISTORIQUES RÉELLES
            logger.info("📊 Chargement des données historiques Vol75...")
            self.historical_data_loaded = await self.deriv_api.load_historical_on_startup()

            if self.historical_data_loaded:
                logger.info("✅ Données historiques Vol75 chargées avec succès")
            else:
                logger.info("⚠️ Mode collecte temps réel activé")

            # Initialiser le modèle IA
            logger.info("🧠 Chargement du modèle IA...")
            training_success = self.ai_model.load_or_create_model()
            logger.info("✅ Modèle IA prêt")

            # Notification de démarrage améliorée
            await self.telegram_bot.send_startup_notification(self.historical_data_loaded)

            # Si entraînement réussi avec données historiques, notifier
            if training_success and self.historical_data_loaded and hasattr(self.ai_model, 'validation_accuracy'):
                if self.ai_model.validation_accuracy > 0.5:  # Seulement si modèle vraiment entraîné
                    training_results = {
                        'accuracy': self.ai_model.validation_accuracy,
                        'samples': self.ai_model.training_samples,
                        'features': getattr(self.ai_model, 'n_features', 18)
                    }
                    await self.telegram_bot.send_ai_training_notification(training_results)

            logger.info("✅ Initialisation terminée avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            await self.telegram_bot.send_error_alert(str(e), "Initialisation")
            raise

    def _check_configuration(self):
        """Vérifier la configuration requise"""
        required_vars = ['DERIV_APP_ID', 'TELEGRAM_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Variables d'environnement manquantes: {missing_vars}")
            return False

        return True

    async def run(self):
        """Boucle principale du bot avec notifications de santé"""
        try:
            await self.initialize()

            logger.info("🔄 Démarrage de la boucle principale...")

            while self.running:
                try:
                    current_time = time.time()

                    # 🆕 NOTIFICATION DE SANTÉ HORAIRE
                    if current_time - self.last_health_notification >= self.health_interval:
                        await self.send_health_notification()
                        self.last_health_notification = current_time

                    # Vérifier les heures de trading (éviter 22h-6h UTC)
                    if not self._is_trading_hours():
                        await asyncio.sleep(300)  # Vérifier toutes les 5 minutes
                        continue

                    # Reset du compteur journalier si nouveau jour
                    self._reset_daily_counter()

                    # Vérifier les limites de trading
                    if not self._can_trade():
                        await asyncio.sleep(300)
                        continue

                    # Analyser et traiter les données de marché
                    await self.process_market_data()

                    # Attendre avant la prochaine analyse
                    await asyncio.sleep(300)  # 5 minutes

                except Exception as e:
                    logger.error(f"Erreur dans la boucle principale: {e}")
                    await asyncio.sleep(60)  # Attendre 1 minute avant de réessayer

        except KeyboardInterrupt:
            logger.info("Arrêt du bot demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur critique dans run(): {e}")
            await self.telegram_bot.send_error_alert(str(e), "Système")
        finally:
            await self.cleanup()

    def _is_trading_hours(self):
        """Vérifier si on est dans les heures de trading"""
        current_hour = datetime.now(timezone.utc).hour
        # Éviter 22h-6h UTC (heures de faible liquidité)
        return not (22 <= current_hour or current_hour < 6)

    def _reset_daily_counter(self):
        """Reset du compteur de trades journaliers"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            if self.signals_today > 0:  # Log seulement si il y a eu des signaux
                logger.info(f"📅 Fin de journée - {self.signals_today} signaux générés le {self.last_trade_date}")

            self.daily_trades = 0
            self.signals_today = 0
            self.last_trade_date = today
            logger.info(f"📅 Nouveau jour - Reset compteur trades: {today}")

    def _can_trade(self):
        """Vérifier si on peut trader"""
        current_time = time.time()

        # Vérifier l'intervalle minimum entre signaux
        if current_time - self.last_signal_time < self.signal_interval:
            return False

        # Vérifier le maximum de trades journaliers
        if self.daily_trades >= self.max_daily_trades:
            logger.debug(f"Maximum trades journaliers atteint: {self.daily_trades}")
            return False

        # Vérifier les pertes consécutives
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Maximum pertes consécutives atteint: {self.consecutive_losses}")
            return False

        return True

    async def process_market_data(self):
        """Traiter les données de marché et générer des signaux"""
        try:
            # Récupérer les données récentes
            data = await self.deriv_api.get_latest_data()
            if data is None or len(data) < 50:
                logger.debug("Pas assez de données pour l'analyse")
                return

            # Mettre à jour le prix actuel
            self.current_price = float(data['price'].iloc[-1])

            logger.debug(f"Analyse de {len(data)} points de données (prix actuel: {self.current_price:.5f})")

            # Analyse technique
            tech_score = self.technical_analysis.calculate_score(data)
            logger.debug(f"Score technique: {tech_score}")

            # Prédiction IA
            ai_prediction = self.ai_model.predict(data)
            logger.debug(f"Prédiction IA: {ai_prediction}")

            # Générer signal si conditions réunies
            signal = self.signal_generator.generate_signal(
                data, tech_score, ai_prediction
            )

            if signal:
                await self.process_signal(signal)

        except Exception as e:
            logger.error(f"Erreur traitement données marché: {e}")

    async def process_signal(self, signal):
        """Traiter et envoyer un signal avec comptage"""
        try:
            logger.info(f"🎯 Signal généré: {signal['direction']} à {signal['entry_price']}")

            # Envoyer notification Telegram
            await self.telegram_bot.send_signal(signal)

            # Mettre à jour les compteurs
            self.last_signal_time = time.time()
            self.daily_trades += 1
            self.signals_today += 1
            self.last_signal_time_obj = datetime.now()

            # Sauvegarder le signal
            self._save_signal(signal)

            # Log détaillé
            logger.info(
                f"📊 Signal #{self.signals_today} envoyé - Direction: {signal['direction']}, "
                f"Score: {signal['combined_score']}, "
                f"Trades aujourd'hui: {self.daily_trades}"
            )

        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")

    def _save_signal(self, signal):
        """Sauvegarder le signal dans un fichier CSV"""
        try:
            import csv
            os.makedirs('data', exist_ok=True)

            csv_file = 'data/signals.csv'
            file_exists = os.path.exists(csv_file)

            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=signal.keys())

                # Écrire l'en-tête si nouveau fichier
                if not file_exists:
                    writer.writeheader()

                writer.writerow(signal)

        except Exception as e:
            logger.error(f"Erreur sauvegarde signal: {e}")

    async def send_health_notification(self):
        """Envoyer la notification de santé horaire"""
        try:
            # Récupérer les données actuelles
            data = await self.deriv_api.get_latest_data()
            price_change_1h = 0

            if data is not None and len(data) > 0:
                self.current_price = float(data['price'].iloc[-1])

                # Calculer variation 1h (approximative)
                if len(data) >= 12:  # Au moins 1h de données (5min intervals)
                    price_1h_ago = float(data['price'].iloc[-13])  # 12*5min = 1h
                    price_change_1h = ((self.current_price - price_1h_ago) / price_1h_ago) * 100

            # Préparer les stats du bot
            uptime = (datetime.now() - self.start_time).total_seconds() / 3600

            # Déterminer le mode IA
            ai_mode = "Mode Simple"
            ai_accuracy = 0
            if hasattr(self.ai_model, 'model') and self.ai_model.model is not None:
                ai_mode = "XGBoost Actif"
                ai_accuracy = getattr(self.ai_model, 'validation_accuracy', 0)

            bot_stats = {
                'connected': self.deriv_api.connected,
                'uptime_hours': uptime,
                'data_points': len(self.deriv_api.data_buffer),
                'ai_mode': ai_mode,
                'ai_accuracy': ai_accuracy,
                'signals_today': self.signals_today,
                'last_signal_time': self.last_signal_time_obj,
                'current_price': self.current_price,
                'price_change_1h': price_change_1h,
                'trading_mode': os.getenv('TRADING_MODE', 'demo')
            }

            # Envoyer la notification
            success = await self.telegram_bot.send_health_notification(bot_stats)
            if success:
                logger.info("📱 Notification de santé envoyée")
            else:
                logger.warning("⚠️ Échec envoi notification de santé")

        except Exception as e:
            logger.error(f"Erreur notification santé: {e}")

    async def cleanup(self):
        """Nettoyage avant arrêt"""
        try:
            logger.info("🧹 Nettoyage avant arrêt...")

            # Envoyer statistiques finales
            if self.signals_today > 0:
                uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                final_stats = f"""
🛑 <b>Bot Vol75 arrêté</b>

📊 <b>Session terminée:</b>
• Durée: {uptime:.1f}h
• Signaux générés: {self.signals_today}
• Dernier prix: {self.current_price:.5f}
• Messages Telegram: {self.telegram_bot.messages_sent}

🕐 <i>Arrêté le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</i>
"""
                await self.telegram_bot.send_message(final_stats)
            else:
                await self.telegram_bot.send_shutdown_message()

            # Fermer la connexion Deriv
            await self.deriv_api.disconnect()

            logger.info("✅ Nettoyage terminé")

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


def main():
    """Point d'entrée principal"""
    # Configuration des logs
    setup_logging()

    logger.info("=" * 60)
    logger.info("BOT TRADING VOL75 - DÉMARRAGE")
    logger.info("Version: 2.0 avec IA XGBoost et données historiques")
    logger.info("=" * 60)

    # Créer et lancer le bot
    bot = TradingBot()

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        sys.exit(1)
    finally:
        logger.info("=" * 60)
        logger.info("BOT TRADING VOL75 - ARRÊTÉ")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()