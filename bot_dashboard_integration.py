#!/usr/bin/env python3
"""
Intégration Dashboard - Module pour connecter le bot au dashboard
🚀 Module pour envoyer les données du bot vers le dashboard temps réel
Version 3.1 - Intégration complète avec gestion d'erreurs robuste
"""

import requests
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import os
import time

logger = logging.getLogger(__name__)


class DashboardIntegration:
    """Classe pour intégrer le bot avec le dashboard temps réel"""

    def __init__(self):
        """Initialisation de l'intégration dashboard"""
        self.api_url = os.getenv('DASHBOARD_API_URL', 'http://dashboard-api:8000')
        self.enabled = os.getenv('DASHBOARD_ENABLED', 'true').lower() == 'true'

        # Configuration de retry et timeout
        self.max_retries = 3
        self.timeout = 5
        self.retry_delay = 1

        # Throttling pour éviter le spam
        self._last_price_send = 0
        self._last_metrics_send = 0
        self._price_throttle = 30  # 30 secondes entre envois de prix
        self._metrics_throttle = 60  # 60 secondes entre envois de métriques

        # Statistiques d'envoi
        self.signals_sent = 0
        self.metrics_sent = 0
        self.prices_sent = 0
        self.errors_count = 0
        self.last_connection_test = 0
        self.connection_status = False

        if self.enabled:
            logger.info(f"🚀 Dashboard intégration activée: {self.api_url}")
            # Test initial de connexion
            self._test_connection_async()
        else:
            logger.info("📊 Dashboard intégration désactivée")

    def _test_connection_async(self):
        """Test de connexion asynchrone non-bloquant"""
        try:
            current_time = time.time()
            # Tester seulement toutes les 5 minutes
            if current_time - self.last_connection_test < 300:
                return self.connection_status

            self.last_connection_test = current_time
            response = requests.get(f"{self.api_url}/", timeout=2)
            self.connection_status = response.status_code == 200

            if self.connection_status:
                logger.debug("✅ Dashboard API accessible")
            else:
                logger.warning(f"⚠️ Dashboard API erreur: {response.status_code}")

        except Exception as e:
            self.connection_status = False
            logger.debug(f"📊 Dashboard API non accessible: {e}")

        return self.connection_status

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

            # Nettoyer les données (supprimer les valeurs None)
            api_data = {k: v for k, v in api_data.items() if v is not None}

            # Envoyer avec retry
            success = await self._send_with_retry(
                endpoint="/api/signals",
                data=api_data,
                operation="signal"
            )

            if success:
                self.signals_sent += 1
                logger.debug(f"📊 Signal envoyé au dashboard (Total: {self.signals_sent})")

            return success

        except Exception as e:
            self.errors_count += 1
            logger.error(f"❌ Erreur envoi signal dashboard: {e}")
            return False

    async def send_system_metrics(self, bot_stats: Dict) -> bool:
        """Envoyer les métriques système avec throttling"""
        if not self.enabled:
            return True

        try:
            # Throttling - éviter le spam
            current_time = time.time()
            if current_time - self._last_metrics_send < self._metrics_throttle:
                return True

            self._last_metrics_send = current_time

            # Préparer métriques
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'bot_status': 'RUNNING' if bot_stats.get('connected', False) else 'STOPPED',
                'deriv_connected': bool(bot_stats.get('connected', False)),
                'telegram_connected': True,  # Assumé si on peut envoyer
                'signals_today': int(bot_stats.get('signals_today', 0)),
                'mtf_rejections': int(bot_stats.get('mtf_rejections', 0)),
                'ai_accuracy': float(bot_stats.get('ai_accuracy', 0)),
                'uptime_hours': float(bot_stats.get('uptime_hours', 0)),
                'premium_signals': int(bot_stats.get('premium_signals', 0)),
                'high_quality_signals': int(bot_stats.get('high_quality_signals', 0))
            }

            # Envoyer avec retry
            success = await self._send_with_retry(
                endpoint="/api/system/metrics",
                data=metrics,
                operation="metrics"
            )

            if success:
                self.metrics_sent += 1
                logger.debug(f"📊 Métriques envoyées au dashboard (Total: {self.metrics_sent})")

            return success

        except Exception as e:
            self.errors_count += 1
            logger.error(f"❌ Erreur envoi metrics dashboard: {e}")
            return False

    async def send_price_data(self, price_data: Dict) -> bool:
        """Envoyer données de prix temps réel avec throttling intelligent"""
        if not self.enabled:
            return True

        try:
            # Throttling intelligent - envoyer toutes les 30 secondes
            current_time = time.time()
            if current_time - self._last_price_send < self._price_throttle:
                return True

            self._last_price_send = current_time

            # Préparer données prix
            api_data = {
                'timestamp': price_data.get('timestamp', datetime.now().isoformat()),
                'price': float(price_data.get('price', 0)),
                'high': float(price_data.get('high', price_data.get('price', 0))),
                'low': float(price_data.get('low', price_data.get('price', 0))),
                'volume': float(price_data.get('volume', 1000))
            }

            # Validation des données
            if api_data['price'] <= 0:
                logger.warning("⚠️ Prix invalide, envoi ignoré")
                return False

            # Envoyer avec retry
            success = await self._send_with_retry(
                endpoint="/api/price",
                data=api_data,
                operation="price"
            )

            if success:
                self.prices_sent += 1
                logger.debug(f"📊 Prix envoyé au dashboard: {api_data['price']:.5f} (Total: {self.prices_sent})")

            return success

        except Exception as e:
            self.errors_count += 1
            logger.error(f"❌ Erreur envoi prix dashboard: {e}")
            return False

    async def _send_with_retry(self, endpoint: str, data: Dict, operation: str) -> bool:
        """Méthode générique d'envoi avec retry et gestion d'erreurs"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}{endpoint}",
                    json=data,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return True
                elif response.status_code == 422:
                    # Erreur de validation - ne pas retry
                    logger.warning(f"⚠️ Erreur validation {operation}: {response.text}")
                    return False
                else:
                    logger.warning(f"⚠️ Erreur {operation} (tentative {attempt + 1}): {response.status_code}")

            except requests.exceptions.ConnectionError:
                if attempt == 0:  # Log seulement à la première tentative
                    logger.debug(f"📊 Dashboard API non accessible pour {operation}")

            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ Timeout {operation} (tentative {attempt + 1})")

            except requests.exceptions.RequestException as e:
                logger.warning(f"🌐 Erreur réseau {operation}: {e}")

            except Exception as e:
                logger.error(f"❌ Erreur inattendue {operation}: {e}")
                return False

            # Attendre avant retry (sauf pour la dernière tentative)
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        # Toutes les tentatives ont échoué
        self.errors_count += 1
        return False

    def test_connection(self) -> bool:
        """Tester la connexion au dashboard de manière synchrone"""
        if not self.enabled:
            return True

        return self._test_connection_async()

    async def send_startup_notification(self, bot_info: Dict) -> bool:
        """Envoyer notification de démarrage du bot"""
        try:
            startup_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'bot_startup',
                'bot_version': bot_info.get('version', '3.1'),
                'features_enabled': bot_info.get('features', []),
                'ai_model': bot_info.get('ai_model', {}),
                'configuration': {
                    'trading_mode': bot_info.get('trading_mode', 'demo'),
                    'capital': bot_info.get('capital', 1000),
                    'risk_amount': bot_info.get('risk_amount', 10),
                    'mtf_enabled': bot_info.get('mtf_enabled', True)
                }
            }

            # Utiliser l'endpoint générique pour les événements
            success = await self._send_with_retry(
                endpoint="/api/system/metrics",
                data=startup_data,
                operation="startup"
            )

            if success:
                logger.info("📊 Notification démarrage envoyée au dashboard")

            return success

        except Exception as e:
            logger.error(f"❌ Erreur notification démarrage: {e}")
            return False

    async def send_shutdown_notification(self, shutdown_stats: Dict) -> bool:
        """Envoyer notification d'arrêt du bot"""
        try:
            shutdown_data = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'bot_shutdown',
                'session_stats': shutdown_stats,
                'bot_status': 'STOPPED'
            }

            success = await self._send_with_retry(
                endpoint="/api/system/metrics",
                data=shutdown_data,
                operation="shutdown"
            )

            if success:
                logger.info("📊 Notification arrêt envoyée au dashboard")

            return success

        except Exception as e:
            logger.error(f"❌ Erreur notification arrêt: {e}")
            return False

    def get_integration_stats(self) -> Dict:
        """Obtenir les statistiques de l'intégration dashboard"""
        return {
            'enabled': self.enabled,
            'api_url': self.api_url,
            'connection_status': self.connection_status,
            'signals_sent': self.signals_sent,
            'metrics_sent': self.metrics_sent,
            'prices_sent': self.prices_sent,
            'errors_count': self.errors_count,
            'success_rate': self._calculate_success_rate(),
            'last_connection_test': datetime.fromtimestamp(
                self.last_connection_test).isoformat() if self.last_connection_test else None,
            'throttling': {
                'price_interval': self._price_throttle,
                'metrics_interval': self._metrics_throttle
            }
        }

    def _calculate_success_rate(self) -> float:
        """Calculer le taux de succès des envois"""
        total_attempts = self.signals_sent + self.metrics_sent + self.prices_sent + self.errors_count
        if total_attempts == 0:
            return 100.0

        successful = self.signals_sent + self.metrics_sent + self.prices_sent
        return (successful / total_attempts) * 100

    async def health_check(self) -> Dict:
        """Vérification de santé de l'intégration dashboard"""
        health_status = {
            'dashboard_enabled': self.enabled,
            'api_reachable': False,
            'response_time_ms': None,
            'last_error': None
        }

        if not self.enabled:
            return health_status

        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/", timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000

            health_status['api_reachable'] = response.status_code == 200
            health_status['response_time_ms'] = round(response_time, 2)

        except Exception as e:
            health_status['last_error'] = str(e)

        return health_status

    def reset_stats(self):
        """Réinitialiser les statistiques"""
        self.signals_sent = 0
        self.metrics_sent = 0
        self.prices_sent = 0
        self.errors_count = 0
        logger.info("📊 Statistiques dashboard réinitialisées")

    def configure_throttling(self, price_interval: int = 30, metrics_interval: int = 60):
        """Configurer les intervalles de throttling"""
        self._price_throttle = price_interval
        self._metrics_throttle = metrics_interval
        logger.info(f"📊 Throttling configuré: Prix={price_interval}s, Métriques={metrics_interval}s")


# =============================================================================
# FONCTIONS UTILITAIRES POUR L'INTÉGRATION
# =============================================================================

async def test_dashboard_integration():
    """Test complet de l'intégration dashboard"""
    print("🧪 Test de l'intégration dashboard...")

    dashboard = DashboardIntegration()

    # Test 1: Connexion
    connected = dashboard.test_connection()
    print(f"Connexion: {'✅' if connected else '❌'}")

    if not connected:
        print("❌ Dashboard API non disponible")
        return

    # Test 2: Signal de test
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

    signal_success = await dashboard.send_signal(test_signal)
    print(f"Test signal: {'✅' if signal_success else '❌'}")

    # Test 3: Métriques de test
    test_metrics = {
        'connected': True,
        'signals_today': 5,
        'mtf_rejections': 12,
        'ai_accuracy': 0.876,
        'uptime_hours': 24.5,
        'premium_signals': 3,
        'high_quality_signals': 2
    }

    metrics_success = await dashboard.send_system_metrics(test_metrics)
    print(f"Test métriques: {'✅' if metrics_success else '❌'}")

    # Test 4: Prix de test
    test_price = {
        'price': 1052.34567,
        'high': 1053.12345,
        'low': 1051.23456,
        'volume': 1500
    }

    price_success = await dashboard.send_price_data(test_price)
    print(f"Test prix: {'✅' if price_success else '❌'}")

    # Test 5: Health check
    health = await dashboard.health_check()
    print(f"Health check: {health}")

    # Statistiques finales
    stats = dashboard.get_integration_stats()
    print(f"\n📊 Statistiques:")
    print(f"   Signaux envoyés: {stats['signals_sent']}")
    print(f"   Métriques envoyées: {stats['metrics_sent']}")
    print(f"   Prix envoyés: {stats['prices_sent']}")
    print(f"   Erreurs: {stats['errors_count']}")
    print(f"   Taux de succès: {stats['success_rate']:.1f}%")


# =============================================================================
# EXEMPLE D'UTILISATION DANS VOTRE MAIN.PY
# =============================================================================

def get_integration_example():
    """Exemple d'intégration dans votre main.py"""
    return """
    # Dans votre classe OptimizedTradingBotMTF:

    def __init__(self):
        # Vos imports existants...
        self.dashboard = DashboardIntegration()

    async def initialize(self):
        # Test connexion dashboard
        if self.dashboard.test_connection():
            logger.info("✅ Dashboard connecté")
            # Envoyer notification de démarrage
            await self.dashboard.send_startup_notification({
                'version': '3.1',
                'trading_mode': os.getenv('TRADING_MODE', 'demo'),
                'capital': os.getenv('CAPITAL', 1000),
                'mtf_enabled': True
            })
        else:
            logger.warning("⚠️ Dashboard non disponible")

    async def process_optimized_signal_mtf(self, signal):
        # Votre code existant...
        await self.telegram_bot.send_signal(signal)
        # Envoyer au dashboard
        await self.dashboard.send_signal(signal)

    async def cleanup_optimized_mtf(self):
        # Notification d'arrêt
        await self.dashboard.send_shutdown_notification({
            'signals_today': self.signals_today,
            'uptime_hours': uptime,
            'premium_signals': self.premium_signals
        })
    """


if __name__ == "__main__":
    # Lancer le test si le fichier est exécuté directement
    import asyncio

    asyncio.run(test_dashboard_integration())