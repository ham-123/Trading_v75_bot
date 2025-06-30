#!/usr/bin/env python3
"""
Deriv API - Gestion de la connexion WebSocket et collecte de données Vol75
Connexion temps réel, auto-reconnexion, sauvegarde CSV, données historiques
"""

import websocket
import json
import pandas as pd
import asyncio
import logging
import threading
import time
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class DerivAPI:
    """Classe pour gérer la connexion WebSocket avec Deriv API"""

    def __init__(self):
        """Initialisation de l'API Deriv"""
        self.app_id = os.getenv('DERIV_APP_ID')
        self.token = os.getenv('DERIV_TOKEN')  # Optionnel pour mode démo

        if not self.app_id:
            raise ValueError("DERIV_APP_ID requis dans les variables d'environnement")

        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = None
        self.ws_thread = None

        # Buffer de données
        self.data_buffer: List[Dict] = []
        self.max_buffer_size = 1000

        # État de connexion
        self.connected = False
        self.authenticated = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Statistiques
        self.messages_received = 0
        self.last_tick_time = None

        # Variables pour données historiques
        self.historical_data = []
        self.historical_complete = False
        self.historical_error = None

        # Fichier de sauvegarde
        self.csv_file = 'data/vol75_data.csv'
        os.makedirs('data', exist_ok=True)

    async def connect(self):
        """Établir la connexion WebSocket"""
        try:
            logger.info(f"🔌 Connexion à Deriv API: {self.ws_url}")

            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            # Démarrer la connexion WebSocket dans un thread séparé
            self.ws_thread = threading.Thread(target=self._run_websocket)
            self.ws_thread.daemon = True
            self.ws_thread.start()

            # Attendre la connexion (max 10 secondes)
            for _ in range(100):  # 10 secondes avec sleep 0.1
                if self.connected:
                    break
                await asyncio.sleep(0.1)

            if not self.connected:
                raise Exception("Échec de connexion WebSocket")

            logger.info("✅ Connexion Deriv API établie")

        except Exception as e:
            logger.error(f"❌ Erreur connexion Deriv: {e}")
            raise

    def _run_websocket(self):
        """Exécuter la connexion WebSocket"""
        try:
            self.ws.run_forever(
                ping_interval=30,  # Ping toutes les 30 secondes
                ping_timeout=10  # Timeout de 10 secondes
            )
        except Exception as e:
            logger.error(f"Erreur WebSocket thread: {e}")

    def _on_open(self, ws):
        """Callback à l'ouverture de connexion"""
        logger.info("📡 Connexion WebSocket ouverte")
        self.connected = True
        self.reconnect_attempts = 0

        # S'authentifier si token fourni
        if self.token:
            self._authenticate()

        # S'abonner aux ticks Vol75
        self._subscribe_to_vol75()

    def _authenticate(self):
        """Authentification avec token (optionnel)"""
        try:
            auth_message = {
                "authorize": self.token,
                "req_id": int(time.time())
            }
            self.ws.send(json.dumps(auth_message))
            logger.info("🔐 Demande d'authentification envoyée")

        except Exception as e:
            logger.error(f"Erreur authentification: {e}")

    def _subscribe_to_vol75(self):
        """S'abonner aux ticks de l'indice Vol75"""
        try:
            subscribe_message = {
                "ticks": "R_75",  # Volatility 75 Index
                "subscribe": 1,
                "req_id": int(time.time())
            }
            self.ws.send(json.dumps(subscribe_message))
            logger.info("📊 Abonnement aux ticks Vol75 demandé")

        except Exception as e:
            logger.error(f"Erreur abonnement Vol75: {e}")

    def _on_message(self, ws, message):
        """Traitement des messages reçus"""
        try:
            data = json.loads(message)

            # Traiter les différents types de messages
            if 'authorize' in data:
                self._handle_auth_response(data)
            elif 'tick' in data:
                self._handle_tick_data(data)
            elif 'subscription' in data:
                self._handle_subscription_response(data)
            elif 'history' in data:
                self._handle_historical_response(data)
            elif 'error' in data:
                self._handle_error_response(data)

        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON: {e}")
        except Exception as e:
            logger.error(f"Erreur traitement message: {e}")

    def _handle_auth_response(self, data):
        """Traiter la réponse d'authentification"""
        if 'authorize' in data and data['authorize']:
            self.authenticated = True
            logger.info("✅ Authentification réussie")
        else:
            logger.warning("❌ Échec d'authentification")

    def _handle_tick_data(self, data):
        """Traiter les données de tick reçues"""
        try:
            tick = data['tick']

            # Créer l'objet tick
            tick_data = {
                'timestamp': datetime.fromtimestamp(tick['epoch'], tz=timezone.utc),
                'price': float(tick['quote']),
                'symbol': tick['symbol'],
                'pip_size': tick.get('pip_size', 0.00001),
                'epoch': tick['epoch']
            }

            # Ajouter au buffer
            self.data_buffer.append(tick_data)
            self.messages_received += 1
            self.last_tick_time = datetime.now(timezone.utc)

            # Maintenir la taille du buffer
            if len(self.data_buffer) > self.max_buffer_size:
                self.data_buffer = self.data_buffer[-self.max_buffer_size:]

            # Log périodique
            if self.messages_received % 100 == 0:
                logger.debug(f"📈 {self.messages_received} ticks reçus, prix actuel: {tick_data['price']}")

            # Sauvegarder périodiquement (toutes les 50 ticks)
            if len(self.data_buffer) % 50 == 0:
                self._save_to_csv()

        except Exception as e:
            logger.error(f"Erreur traitement tick: {e}")

    def _handle_subscription_response(self, data):
        """Traiter la réponse d'abonnement"""
        subscription = data.get('subscription', {})
        if subscription.get('id'):
            logger.info(f"✅ Abonnement confirmé: {subscription['id']}")
        else:
            logger.warning("❌ Échec d'abonnement")

    def _handle_historical_response(self, data):
        """Traiter la réponse de données historiques"""
        try:
            if 'history' in data:
                history = data['history']
                prices = history.get('prices', [])
                times = history.get('times', [])

                logger.info(f"📈 Traitement de {len(prices)} points historiques...")

                # Convertir en format standard
                for timestamp, price in zip(times, prices):
                    tick_data = {
                        'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        'price': float(price),
                        'symbol': 'R_75',
                        'pip_size': 0.00001,
                        'epoch': timestamp
                    }
                    self.historical_data.append(tick_data)

                logger.info(f"✅ {len(self.historical_data)} points historiques Vol75 récupérés")
                self.historical_complete = True

        except Exception as e:
            logger.error(f"Erreur traitement réponse historique: {e}")
            self.historical_error = str(e)
            self.historical_complete = True

    def _handle_error_response(self, data):
        """Traiter les messages d'erreur"""
        error = data.get('error', {})
        error_code = error.get('code', 'Inconnu')
        error_message = error.get('message', 'Erreur inconnue')

        # Ignorer certaines erreurs temporaires
        if error_code == 'WrongResponse':
            logger.debug(f"Erreur API temporaire: {error_code} - {error_message}")
        else:
            logger.error(f"❌ Erreur API: {error_code} - {error_message}")

    def _on_error(self, ws, error):
        """Callback en cas d'erreur WebSocket"""
        logger.error(f"❌ Erreur WebSocket: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback à la fermeture de connexion"""
        self.connected = False
        self.authenticated = False

        logger.warning(f"🔌 Connexion fermée: {close_status_code} - {close_msg}")

        # Tenter une reconnexion automatique
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self._schedule_reconnect()
        else:
            logger.error("❌ Maximum de tentatives de reconnexion atteint")

    def _schedule_reconnect(self):
        """Programmer une reconnexion"""
        self.reconnect_attempts += 1
        delay = min(2 ** self.reconnect_attempts, 60)  # Backoff exponentiel, max 60s

        logger.info(f"🔄 Reconnexion #{self.reconnect_attempts} dans {delay}s...")

        def reconnect():
            import time
            time.sleep(delay)
            # Utiliser asyncio.run au lieu de create_task
            try:
                import asyncio
                asyncio.run(self.connect())
            except Exception as e:
                logger.error(f"Erreur reconnexion: {e}")

        threading.Thread(target=reconnect, daemon=True).start()

    def get_historical_data(self, days_back=30):
        """Récupérer les données historiques Vol75 réelles via API"""
        try:
            logger.info(f"📊 Récupération données historiques Vol75 ({days_back} jours)...")

            # Calculer les timestamps
            end_time = int(time.time())
            start_time = int(end_time - (days_back * 24 * 60 * 60))

            # Préparer la requête historique
            history_request = {
                "ticks_history": "R_75",
                "start": start_time,
                "end": end_time,
                "style": "ticks",
                "count": 5000,  # Maximum Deriv API
                "req_id": int(time.time() * 1000)  # ID unique
            }

            # Réinitialiser les variables
            self.historical_data = []
            self.historical_complete = False
            self.historical_error = None

            # Envoyer la requête
            self.ws.send(json.dumps(history_request))
            logger.info("📡 Requête données historiques envoyée...")

            # Attendre la réponse (max 60 secondes)
            wait_time = 0
            while not self.historical_complete and wait_time < 60:
                time.sleep(0.5)
                wait_time += 0.5

            # Traitement des résultats
            if self.historical_error:
                logger.error(f"❌ Échec récupération historique: {self.historical_error}")
                return None

            if not self.historical_data:
                logger.warning("⚠️ Aucune donnée historique reçue")
                return None

            # Sauvegarder en CSV
            df_historical = pd.DataFrame(self.historical_data)
            df_historical = df_historical.sort_values('timestamp').reset_index(drop=True)

            # Sauvegarder dans le fichier principal
            df_historical.to_csv(self.csv_file, index=False)
            logger.info(f"💾 {len(df_historical)} points sauvegardés dans {self.csv_file}")

            return df_historical

        except Exception as e:
            logger.error(f"❌ Erreur critique récupération historique: {e}")
            return None

    async def load_historical_on_startup(self):
        """Charger les données historiques au démarrage"""
        try:
            # Vérifier si on a déjà des données récentes
            if os.path.exists(self.csv_file):
                try:
                    df_existing = pd.read_csv(self.csv_file)
                    if len(df_existing) > 1000:
                        df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                        last_date = df_existing['timestamp'].max()

                        # Calculer l'âge des données
                        if hasattr(last_date, 'tz_localize'):
                            last_date = last_date.tz_localize('UTC')
                        elif last_date.tzinfo is None:
                            last_date = last_date.replace(tzinfo=timezone.utc)

                        hours_ago = (datetime.now(timezone.utc) - last_date).total_seconds() / 3600

                        if hours_ago < 2:  # Données de moins de 2h
                            logger.info(
                                f"✅ Données récentes trouvées: {len(df_existing)} points (dernière: {hours_ago:.1f}h)")
                            # Charger dans le buffer
                            for _, row in df_existing.tail(self.max_buffer_size).iterrows():
                                tick_data = {
                                    'timestamp': pd.to_datetime(row['timestamp']),
                                    'price': float(row['price']),
                                    'symbol': row.get('symbol', 'R_75'),
                                    'pip_size': row.get('pip_size', 0.00001),
                                    'epoch': row.get('epoch', 0)
                                }
                                self.data_buffer.append(tick_data)
                            return True
                except Exception as e:
                    logger.warning(f"Erreur lecture fichier existant: {e}")

            # Attendre que la connexion soit stable
            await asyncio.sleep(2)

            if not self.connected:
                logger.warning("⚠️ Connexion non établie, impossible de récupérer l'historique")
                return False

            # Récupérer de nouvelles données historiques
            logger.info("🔄 Récupération de nouvelles données historiques Vol75...")
            historical_df = self.get_historical_data(days_back=30)  # 1 mois de données

            if historical_df is not None and len(historical_df) > 100:
                # Charger dans le buffer
                for _, row in historical_df.tail(self.max_buffer_size).iterrows():
                    tick_data = {
                        'timestamp': row['timestamp'],
                        'price': float(row['price']),
                        'symbol': row.get('symbol', 'R_75'),
                        'pip_size': row.get('pip_size', 0.00001),
                        'epoch': row.get('epoch', 0)
                    }
                    self.data_buffer.append(tick_data)

                logger.info(f"✅ {len(historical_df)} points historiques chargés et prêts pour l'IA")
                return True
            else:
                logger.warning("⚠️ Échec récupération historique, passage en mode temps réel")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur chargement historique: {e}")
            return False

    async def get_latest_data(self, count: int = 200) -> Optional[pd.DataFrame]:
        """Récupérer les dernières données sous forme de DataFrame"""
        try:
            if len(self.data_buffer) < 10:
                logger.debug("Pas assez de données dans le buffer")
                return None

            # Prendre les derniers points
            recent_data = self.data_buffer[-count:] if len(self.data_buffer) >= count else self.data_buffer

            # Convertir en DataFrame
            df = pd.DataFrame(recent_data)

            # S'assurer que les colonnes sont dans le bon type
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['price'] = df['price'].astype(float)

            # Trier par timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            return None

    def _save_to_csv(self):
        """Sauvegarder les données en CSV"""
        try:
            if not self.data_buffer:
                return

            # Créer DataFrame temporaire
            df = pd.DataFrame(self.data_buffer)

            # Vérifier si le fichier existe
            if os.path.exists(self.csv_file):
                # Lire les dernières données pour éviter les doublons
                try:
                    existing_df = pd.read_csv(self.csv_file, parse_dates=['timestamp'])
                    last_timestamp = existing_df['timestamp'].max()

                    # Filtrer seulement les nouvelles données
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    new_data = df[df['timestamp'] > last_timestamp]

                    if len(new_data) > 0:
                        new_data.to_csv(self.csv_file, mode='a', header=False, index=False)
                        logger.debug(f"💾 {len(new_data)} nouveaux points sauvegardés")

                except Exception as e:
                    # Si erreur de lecture, sauvegarder tout
                    df.to_csv(self.csv_file, index=False)
                    logger.warning(f"Sauvegarde complète due à erreur: {e}")
            else:
                # Première sauvegarde
                df.to_csv(self.csv_file, index=False)
                logger.info(f"💾 Fichier CSV créé: {len(df)} points")

        except Exception as e:
            logger.error(f"Erreur sauvegarde CSV: {e}")

    def get_connection_status(self) -> Dict:
        """Obtenir le statut de connexion"""
        return {
            'connected': self.connected,
            'authenticated': self.authenticated,
            'messages_received': self.messages_received,
            'buffer_size': len(self.data_buffer),
            'last_tick': self.last_tick_time.isoformat() if self.last_tick_time else None,
            'reconnect_attempts': self.reconnect_attempts
        }

    async def disconnect(self):
        """Fermer la connexion proprement"""
        try:
            logger.info("🔌 Fermeture de la connexion Deriv...")

            # Sauvegarder les données restantes
            if self.data_buffer:
                self._save_to_csv()

            # Fermer WebSocket
            if self.ws:
                self.ws.close()

            # Attendre que le thread se termine
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)

            self.connected = False
            logger.info("✅ Connexion fermée proprement")

        except Exception as e:
            logger.error(f"Erreur fermeture connexion: {e}")


# Test de la classe si exécuté directement
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(level=logging.INFO)


    async def test_deriv_api():
        """Test de l'API Deriv"""
        api = DerivAPI()

        try:
            await api.connect()

            # Test chargement historique
            historical_loaded = await api.load_historical_on_startup()
            print(f"Données historiques chargées: {historical_loaded}")

            # Attendre quelques données
            for i in range(10):
                await asyncio.sleep(1)
                data = await api.get_latest_data()
                if data is not None:
                    print(f"Données reçues: {len(data)} points, dernier prix: {data['price'].iloc[-1]}")

        finally:
            await api.disconnect()


    asyncio.run(test_deriv_api())