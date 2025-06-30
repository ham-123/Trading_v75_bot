#!/usr/bin/env python3
"""
Bot Trading Vol75 - Point d'entrée principal
Orchestration de tous les composants du système de trading automatisé
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

        # Variables de contrôle
        self.last_signal_time = 0
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.last_trade_date = None
        self.running = True

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
        """Initialisation des composants"""
        try:
            logger.info("🚀 Initialisation du bot Vol75...")

            # Vérifier la configuration
            if not self._check_configuration():
                raise Exception("Configuration invalide")

            # Connexion à Deriv API
            await self.deriv_api.connect()
            logger.info("✅ Connexion Deriv API établie")

            # Initialiser le modèle IA
            logger.info("🧠 Chargement du modèle IA...")
            self.ai_model.load_or_create_model()
            logger.info("✅ Modèle IA prêt")

            # Test de notification Telegram
            await self.telegram_bot.send_message(
                "🤖 <b>Bot Vol75 démarré avec succès!</b>\n"
                f"📊 Mode: {os.getenv('TRADING_MODE', 'demo')}\n"
                f"💰 Capital: {os.getenv('CAPITAL', 1000)}$\n"
                f"⚠️ Risque par trade: {os.getenv('RISK_AMOUNT', 10)}$"
            )

            logger.info("✅ Initialisation terminée avec succès")

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
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
        """Boucle principale du bot"""
        try:
            await self.initialize()

            logger.info("🔄 Démarrage de la boucle principale...")

            while self.running:
                try:
                    # Vérifier l'heure de trading (éviter 22h-6h UTC)
                    if not self._is_trading_hours():
                        await asyncio.sleep(300)  # Vérifier toutes les 5 minutes
                        continue

                    # Reset du compteur journalier si nouveau jour
                    self._reset_daily_counter()

                    # Vérifier les limites de trading
                    if not self._can_trade():
                        await asyncio.sleep(300)
                        continue

                    # Récupérer et analyser les données
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
            await self.telegram_bot.send_message(f"❌ <b>Erreur critique:</b> {e}")
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
            self.daily_trades = 0
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

            logger.debug(f"Analyse de {len(data)} points de données")

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
        """Traiter et envoyer un signal"""
        try:
            logger.info(f"🎯 Signal généré: {signal['direction']} à {signal['entry_price']}")

            # Envoyer notification Telegram
            await self.telegram_bot.send_signal(signal)

            # Mettre à jour les compteurs
            self.last_signal_time = time.time()
            self.daily_trades += 1

            # Sauvegarder le signal
            self._save_signal(signal)

            # Log détaillé
            logger.info(
                f"📊 Signal envoyé - Direction: {signal['direction']}, "
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

    async def cleanup(self):
        """Nettoyage avant arrêt"""
        try:
            logger.info("🧹 Nettoyage avant arrêt...")

            # Fermer la connexion Deriv
            await self.deriv_api.disconnect()

            # Message d'arrêt
            await self.telegram_bot.send_message("🛑 <b>Bot Vol75 arrêté</b>")

            logger.info("✅ Nettoyage terminé")

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")


def main():
    """Point d'entrée principal"""
    # Configuration des logs
    setup_logging()

    logger.info("=" * 50)
    logger.info("BOT TRADING VOL75 - DÉMARRAGE")
    logger.info("=" * 50)

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
        logger.info("Bot arrêté")


if __name__ == "__main__":
    main()