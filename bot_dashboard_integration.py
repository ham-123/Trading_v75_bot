#!/usr/bin/env python3
"""
Intégration Dashboard - À ajouter dans votre main.py
🚀 Module pour envoyer les données du bot vers le dashboard
"""

import requests
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)


class DashboardIntegration:
    """Classe pour intégrer le bot avec le dashboard"""

    def __init__(self):
        """Initialisation de l'intégration dashboard"""
        self.api_url = os.getenv('DASHBOARD_API_URL', 'http://localhost:8000')
        self.enabled = os.getenv('DASHBOARD_ENABLED', 'true').lower() == 'true'

        if self.enabled:
            logger.info(f"🚀 Dashboard intégration activée: {self.api_url}")
        else:
            logger.info("📊 Dashboard intégration désactivée")

    async def send_signal(self, signal_data: Dict) -> bool:
        """Envoyer un nouveau signal au dashboard"""
        if not self.enabled:
            return True

        try:
            # Préparer les données pour l'API
            api_data = {
                'timestamp': signal_data.get('timestamp', datetime.now().isoformat()),
                'direction': signal_data.get('direction'),
                'entry_price': signal_data.get('entry_price'),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profit'),
                'tech_score': signal_data.get('tech_score'),
                'ai_confidence': signal_data.get('ai_confidence'),
                'combined_score': signal_data.get('combined_score'),
                'signal_quality': signal_data.get('signal_quality'),
                'multi_timeframe': signal_data.get('multi_timeframe', {})
            }

            # Envoyer à l'API
            response = requests.post(
                f"{self.api_url}/api/signals",
                json=api_data,
                timeout=5
            )

            if response.status_code == 200:
                logger.debug("📊 Signal envoyé au dashboard")
                return True
            else:
                logger.warning(f"⚠️ Erreur dashboard API: {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.debug("📊 Dashboard API non disponible")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur envoi signal dashboard: {e}")
            return False

    async def send_system_metrics(self, bot_stats: Dict) -> bool:
        """Envoyer les métriques système"""
        if not self.enabled:
            return True

        try:
            # Préparer métriques
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'bot_status': 'RUNNING' if bot_stats.get('connected', False) else 'STOPPED',
                'deriv_connected': bot_stats.get('connected', False),
                'telegram_connected': True,  # Assumé si on peut envoyer
                'signals_today': bot_stats.get('signals_today', 0),
                'mtf_rejections': bot_stats.get('mtf_rejections', 0),
                'ai_accuracy': bot_stats.get('ai_accuracy', 0),
                'uptime_hours': bot_stats.get('uptime_hours', 0),
                'premium_signals': bot_stats.get('premium_signals', 0),
                'high_quality_signals': bot_stats.get('high_quality_signals', 0)
            }

            response = requests.post(
                f"{self.api_url}/api/system/metrics",
                json=metrics,
                timeout=5
            )

            if response.status_code == 200:
                logger.debug("📊 Métriques envoyées au dashboard")
                return True
            else:
                logger.warning(f"⚠️ Erreur metrics dashboard: {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.debug("📊 Dashboard API non disponible")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur envoi metrics dashboard: {e}")
            return False

    async def send_price_data(self, price_data: Dict) -> bool:
        """Envoyer données de prix temps réel"""
        if not self.enabled:
            return True

        try:
            # Envoyer seulement toutes les 30 secondes pour éviter le spam
            current_time = datetime.now()
            if hasattr(self, '_last_price_send'):
                if (current_time - self._last_price_send).total_seconds() < 30:
                    return True

            self._last_price_send = current_time

            # Préparer données prix
            api_data = {
                'timestamp': price_data.get('timestamp', current_time.isoformat()),
                'price': price_data.get('price'),
                'high': price_data.get('high', price_data.get('price')),
                'low': price_data.get('low', price_data.get('price')),
                'volume': price_data.get('volume', 1000)
            }

            response = requests.post(
                f"{self.api_url}/api/price",
                json=api_data,
                timeout=5
            )

            if response.status_code == 200:
                logger.debug("📊 Prix envoyé au dashboard")
                return True
            else:
                logger.warning(f"⚠️ Erreur price dashboard: {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.debug("📊 Dashboard API non disponible")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur envoi prix dashboard: {e}")
            return False

    def test_connection(self) -> bool:
        """Tester la connexion au dashboard"""
        if not self.enabled:
            return True

        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False


# =============================================================================
# MODIFICATIONS À AJOUTER DANS VOTRE MAIN.PY
# =============================================================================

"""
INSTRUCTIONS D'INTÉGRATION:

1. Ajoutez cet import en haut de votre main.py:
   from bot_dashboard_integration import DashboardIntegration

2. Dans la classe OptimizedTradingBotMTF.__init__(), ajoutez:
   self.dashboard = DashboardIntegration()

3. Dans initialize(), après l'initialisation réussie, ajoutez:
   # Test connexion dashboard
   if self.dashboard.test_connection():
       logger.info("✅ Dashboard connecté")
   else:
       logger.warning("⚠️ Dashboard non disponible")

4. Dans process_optimized_signal_mtf(), après l'envoi Telegram, ajoutez:
   # Envoyer au dashboard
   await self.dashboard.send_signal(signal)

5. Dans send_mtf_health_notification(), ajoutez:
   # Envoyer métriques au dashboard
   await self.dashboard.send_system_metrics(bot_stats)

6. Dans process_market_data_optimized_mtf(), ajoutez périodiquement:
   # Envoyer prix au dashboard (toutes les 30s automatiquement)
   if data is not None and len(data) > 0:
       latest_data = {
           'price': float(data['price'].iloc[-1]),
           'high': float(data['high'].iloc[-1]) if 'high' in data else float(data['price'].iloc[-1]),
           'low': float(data['low'].iloc[-1]) if 'low' in data else float(data['price'].iloc[-1]),
           'volume': float(data['volume'].iloc[-1]) if 'volume' in data else 1000,
           'timestamp': datetime.now().isoformat()
       }
       await self.dashboard.send_price_data(latest_data)

7. Dans votre .env, ajoutez:
   DASHBOARD_ENABLED=true
   DASHBOARD_API_URL=http://localhost:8000
"""


# =============================================================================
# CODE COMPLET D'INTÉGRATION POUR MAIN.PY
# =============================================================================

class OptimizedTradingBotMTFWithDashboard:
    """Version de votre bot avec intégration dashboard"""

    def __init__(self):
        """Initialisation avec dashboard"""
        # Votre code existant...

        # 🆕 AJOUTER CETTE LIGNE
        self.dashboard = DashboardIntegration()

        logger.info("🚀 Bot Trading Vol75 OPTIMISÉ MTF + Dashboard initialisé")

    async def initialize(self):
        """Initialisation avec test dashboard"""
        try:
            # Votre code d'initialisation existant...

            # 🆕 AJOUTER APRÈS L'INITIALISATION RÉUSSIE
            # Test connexion dashboard
            if self.dashboard.test_connection():
                logger.info("✅ Dashboard connecté")
            else:
                logger.warning("⚠️ Dashboard non disponible")

            # Votre code existant...

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation optimisée MTF: {e}")
            raise

    async def process_optimized_signal_mtf(self, signal):
        """Traitement signal avec envoi dashboard"""
        try:
            # Votre code existant...

            # 📱 Envoyer notification Telegram MTF COMPLÈTE
            await self.telegram_bot.send_signal(signal)

            # 🆕 AJOUTER CETTE LIGNE
            # 📊 Envoyer au dashboard
            await self.dashboard.send_signal(signal)

            # Votre code existant...

        except Exception as e:
            logger.error(f"Erreur traitement signal optimisé MTF: {e}")

    async def send_mtf_health_notification(self):
        """Notification santé avec envoi dashboard"""
        try:
            # Votre code existant pour créer bot_stats...

            success = await self.telegram_bot.send_mtf_health_notification(bot_stats)

            # 🆕 AJOUTER CETTE LIGNE
            # 📊 Envoyer métriques au dashboard
            await self.dashboard.send_system_metrics(bot_stats)

            if success:
                logger.info("📱 Notification de santé MTF envoyée")

        except Exception as e:
            logger.error(f"Erreur notification santé MTF: {e}")

    async def process_market_data_optimized_mtf(self):
        """Traitement marché avec envoi prix dashboard"""
        try:
            # Votre code existant...
            data = await self.deriv_api.get_latest_data()

            if data is None or len(data) < 100:
                return

            self.current_price = float(data['price'].iloc[-1])

            # 🆕 AJOUTER CES LIGNES
            # 📊 Envoyer prix au dashboard (throttlé automatiquement)
            latest_data = {
                'price': self.current_price,
                'high': float(data['high'].iloc[-1]) if 'high' in data else self.current_price,
                'low': float(data['low'].iloc[-1]) if 'low' in data else self.current_price,
                'volume': float(data['volume'].iloc[-1]) if 'volume' in data else 1000,
                'timestamp': datetime.now().isoformat()
            }
            await self.dashboard.send_price_data(latest_data)

            # Votre code existant...

        except Exception as e:
            logger.error(f"Erreur traitement optimisé MTF: {e}")


# =============================================================================
# VARIABLES D'ENVIRONNEMENT À AJOUTER
# =============================================================================

"""
Ajoutez ces lignes dans votre fichier .env:

# Dashboard Configuration
DASHBOARD_ENABLED=true
DASHBOARD_API_URL=http://localhost:8000

# Redis Configuration (optionnel)
REDIS_HOST=localhost
REDIS_PORT=6379
"""

# =============================================================================
# STRUCTURE FINALE DES FICHIERS
# =============================================================================

"""
Votre projet final aura cette structure:

votre-projet/
├── main.py                          # Votre bot modifié
├── ai_model.py                      # IA Ensemble
├── telegram_bot.py                  # Telegram MTF
├── deriv_api.py                     # API Deriv
├── technical_analysis.py            # Analyse technique
├── signal_generator.py              # Générateur MTF
├── multi_timeframe_analysis.py      # Analyse MTF
├── bot_dashboard_integration.py     # 🆕 Ce fichier
├── docker-compose.yml               # 🆕 Avec dashboard
├── Dockerfile                       # Bot existant
├── Dockerfile.dashboard             # 🆕 API Dashboard
├── Dockerfile.streamlit             # 🆕 Interface
├── requirements.txt                 # Bot existant
├── dashboard/
│   ├── requirements.txt             # 🆕 Dashboard deps
│   ├── api.py                       # 🆕 API FastAPI
│   └── app.py                       # 🆕 Interface Streamlit
├── data/                            # Données partagées
├── logs/                            # Logs partagés
└── .env                             # Variables d'environnement
"""

if __name__ == "__main__":
    # Code de test
    import asyncio


    async def test_dashboard_integration():
        """Test de l'intégration dashboard"""
        dashboard = DashboardIntegration()

        print("🧪 Test de l'intégration dashboard...")

        # Test connexion
        connected = dashboard.test_connection()
        print(f"Connexion: {'✅' if connected else '❌'}")

        if connected:
            # Test signal
            test_signal = {
                'direction': 'BUY',
                'entry_price': 1050.75,
                'stop_loss': 1048.50,
                'take_profit': 1057.25,
                'tech_score': 85,
                'ai_confidence': 0.87,
                'combined_score': 88.5,
                'signal_quality': 'PREMIUM',
                'multi_timeframe': {
                    'confluence_score': 0.82,
                    'strength': 'very_strong'
                }
            }

            success = await dashboard.send_signal(test_signal)
            print(f"Test signal: {'✅' if success else '❌'}")

            # Test métriques
            test_metrics = {
                'connected': True,
                'signals_today': 5,
                'mtf_rejections': 12,
                'ai_accuracy': 0.876,
                'uptime_hours': 24.5
            }

            success = await dashboard.send_system_metrics(test_metrics)
            print(f"Test métriques: {'✅' if success else '❌'}")


    # Lancer le test si le fichier est exécuté directement
    asyncio.run(test_dashboard_integration())