#!/usr/bin/env python3
"""
Deriv API - CORRIGÉ - Gestion WebSocket compatible
🔧 CORRECTION : Suppression paramètre close_timeout incompatible
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
    """Classe pour gérer la connexion WebSocket avec Deriv API - CORRIGÉE"""

    def __init__(self):
        """Initialisation de l'API Deriv"""
        self.app_id = os.getenv('DERIV_APP_ID')
        self.token = os.getenv('DERIV_TOKEN')

        if not self.app_id:
            raise ValueError("DERIV_APP_ID requis dans les variables d'environnement")

        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = None
        self.ws_thread = None

        # Buffer de données
        self.data_buffer: List[Dict] = []
        self.max_buffer_size = 1000

        # Buffer pour calculer high/low
        self.tick_buffer: List[Dict] = []
        self.candle_interval = 300

        # État de connexion
        self.connected = False
        self.authenticated = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Statistiques
        self.messages_received = 0
        self.last_tick_time = None
        self.last_ping_time = time.time()

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
            for _ in range(100):
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
        """🔧 CORRIGÉ : WebSocket sans close_timeout"""
        try:
            # 🔧 CORRECTION : Suppression du paramètre close_timeout non supporté
            self.ws.run_forever(
                ping_interval=15,
                ping_timeout=6,
                ping_payload=b"vol75_keepalive"
                # 🔧 SUPPRIMÉ : close_timeout=10 (non supporté par certaines versions)
            )
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    def _on_open(self, ws):
        """Callback à l'ouverture de connexion"""
        logger.info("📡 Connexion WebSocket ouverte")
        self.connected = True
        self.reconnect_attempts = 0
        self.last_ping_time = time.time()

        # Délai de stabilisation
        time.sleep(0.5)

        # S'authentifier si token fourni
        if self.token:
            self._authenticate()
            time.sleep(0.3)

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
                "ticks": "R_75",
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

    def _handle_subscription_response(self, data):
        """Traiter la réponse d'abonnement"""
        subscription = data.get('subscription', {})
        if subscription.get('id'):
            logger.info(f"✅ Abonnement confirmé: {subscription['id']}")
        else:
            logger.warning("❌ Échec d'abonnement")

    def _handle_tick_data(self, data):
        """Traitement tick handling optimisé"""
        try:
            tick = data['tick']
            current_time = datetime.fromtimestamp(tick['epoch'], tz=timezone.utc)
            price = float(tick['quote'])

            # Update ping time
            self.last_ping_time = time.time()

            # Ajouter au buffer de ticks bruts
            tick_raw = {
                'timestamp': current_time,
                'price': price,
                'epoch': tick['epoch']
            }
            self.tick_buffer.append(tick_raw)

            # Créer données OHLCV
            tick_data = self._create_ohlcv_data(current_time, price, tick)
            if tick_data:
                self.data_buffer.append(tick_data)
                self.messages_received += 1
                self.last_tick_time = current_time

                # Sauvegarde périodique
                if len(self.data_buffer) % 25 == 0:
                    self._save_to_csv()

                # Buffer management
                if len(self.data_buffer) > self.max_buffer_size:
                    self.data_buffer = self.data_buffer[-self.max_buffer_size:]

                # Log moins fréquent
                if self.messages_received % 500 == 0:
                    logger.info(f"📈 {self.messages_received} ticks, prix: {price:.5f}")

        except Exception as e:
            logger.error(f"Erreur tick: {e}")

    def _create_ohlcv_data(self, timestamp: datetime, price: float, tick_raw: dict) -> Optional[Dict]:
        """Créer les données OHLCV à partir des ticks"""
        try:
            # Nettoyer le buffer des ticks trop anciens
            cutoff_time = timestamp.timestamp() - 3600
            self.tick_buffer = [t for t in self.tick_buffer if t['timestamp'].timestamp() > cutoff_time]

            # Calculer high/low sur les derniers ticks
            recent_ticks = self.tick_buffer[-15:] if len(self.tick_buffer) >= 15 else self.tick_buffer

            if not recent_ticks:
                recent_prices = [price]
            else:
                recent_prices = [t['price'] for t in recent_ticks]

            # Créer les données OHLCV
            tick_data = {
                'timestamp': timestamp,
                'price': price,
                'open': recent_prices[0] if recent_prices else price,
                'high': max(recent_prices) if recent_prices else price,
                'low': min(recent_prices) if recent_prices else price,
                'close': price,
                'volume': 1000 + (len(recent_prices) * 50),
                'symbol': tick_raw.get('symbol', 'R_75'),
                'pip_size': tick_raw.get('pip_size', 0.00001),
                'epoch': tick_raw.get('epoch', 0)
            }

            return tick_data

        except Exception as e:
            logger.error(f"Erreur création OHLCV: {e}")
            return None

    def _handle_historical_response(self, data):
        """Traiter la réponse de données historiques"""
        try:
            if 'history' in data:
                history = data['history']
                prices = history.get('prices', [])
                times = history.get('times', [])

                logger.info(f"📈 Traitement de {len(prices)} points historiques...")

                for i, (timestamp, price) in enumerate(zip(times, prices)):
                    price_float = float(price)
                    variation = price_float * 0.001

                    tick_data = {
                        'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc),
                        'price': price_float,
                        'open': price_float,
                        'high': price_float + variation,
                        'low': price_float - variation,
                        'close': price_float,
                        'volume': 1000,
                        'symbol': 'R_75',
                        'pip_size': 0.00001,
                        'epoch': timestamp
                    }
                    self.historical_data.append(tick_data)

                logger.info(f"✅ {len(self.historical_data)} points historiques récupérés")
                self.historical_complete = True

        except Exception as e:
            logger.error(f"Erreur traitement historique: {e}")
            self.historical_error = str(e)
            self.historical_complete = True

    def _handle_error_response(self, data):
        """Traiter les messages d'erreur"""
        error = data.get('error', {})
        error_code = error.get('code', 'Inconnu')
        error_message = error.get('message', 'Erreur inconnue')

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
        """Reconnexion automatique"""
        self.reconnect_attempts += 1

        if self.reconnect_attempts <= 3:
            delay = 2
        elif self.reconnect_attempts <= 6:
            delay = 5
        else:
            delay = min(10 + (self.reconnect_attempts - 6) * 2, 30)

        logger.warning(f"🔄 Reconnexion #{self.reconnect_attempts} dans {delay}s")

        def reconnect():
            time.sleep(delay)
            try:
                if self.ws:
                    try:
                        self.ws.close()
                        self.ws = None
                    except:
                        pass

                self.connected = False
                self.authenticated = False

                asyncio.run(self.connect())

            except Exception as e:
                logger.error(f"Reconnexion failed: {e}")

        threading.Thread(target=reconnect, daemon=True).start()

    def get_connection_health(self) -> Dict:
        """Vérifier la santé de la connexion"""
        current_time = time.time()
        last_activity = getattr(self, 'last_ping_time', current_time)
        time_since_activity = current_time - last_activity

        health_status = {
            'connected': self.connected,
            'authenticated': self.authenticated,
            'messages_received': self.messages_received,
            'buffer_size': len(self.data_buffer),
            'time_since_last_activity': time_since_activity,
            'connection_stable': time_since_activity < 30,
            'reconnect_attempts': self.reconnect_attempts,
            'status': 'HEALTHY' if (self.connected and time_since_activity < 30) else 'UNSTABLE'
        }

        return health_status

    async def load_historical_on_startup(self):
        """Charger les données historiques au démarrage"""
        try:
            # 🔥 FORCER la récupération de 50K données historiques
            logger.info("🔄 Récupération de 50K données historiques Vol75...")

            # Attendre connexion stable
            await asyncio.sleep(2)

            if not self.connected:
                logger.warning("⚠️ Connexion non établie, impossible de récupérer l'historique")
                return False

            # 🔥 RÉCUPÉRER 50K DONNÉES (365 jours au lieu de 30)
            historical_df = self.get_multiple_historical_data(target_points=50000)

            if historical_df is not None and len(historical_df) > 1000:
                logger.info(f"✅ Données historiques récupérées: {len(historical_df):,} points")

                # 🔥 CHARGER TOUTES LES DONNÉES (pas seulement max_buffer_size)
                self.data_buffer = []  # Vider le buffer

                for _, row in historical_df.iterrows():  # TOUTES les données
                    tick_data = {
                        'timestamp': row['timestamp'],
                        'price': float(row['price']),
                        'open': float(row.get('open', row['price'])),
                        'high': float(row.get('high', row['price'])),
                        'low': float(row.get('low', row['price'])),
                        'close': float(row.get('close', row['price'])),
                        'volume': float(row.get('volume', 1000)),
                        'symbol': row.get('symbol', 'R_75'),
                        'pip_size': row.get('pip_size', 0.00001),
                        'epoch': row.get('epoch', 0)
                    }
                    self.data_buffer.append(tick_data)

                logger.info(f"✅ {len(self.data_buffer):,} points chargés dans le buffer")
                return True
            else:
                logger.warning("⚠️ Échec récupération historique")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur chargement historique: {e}")
            return False

    def get_historical_data(self, days_back=365):  # 🔥 CHANGÉ: 365 jours au lieu de 30
        """Récupérer les données historiques Vol75"""
        try:
            logger.info(f"📊 Récupération données historiques Vol75 ({days_back} jours)...")

            end_time = int(time.time())
            start_time = int(end_time - (days_back * 24 * 60 * 60))

            history_request = {
                "ticks_history": "R_75",
                "start": start_time,
                "end": end_time,
                "style": "ticks",
                "count": 50000,  # 🔥 Déjà bon - demander 50K
                "req_id": int(time.time() * 1000)
            }

            self.historical_data = []
            self.historical_complete = False
            self.historical_error = None

            self.ws.send(json.dumps(history_request))
            logger.info("📡 Requête données historiques envoyée...")

            wait_time = 0
            while not self.historical_complete and wait_time < 120:  # 🔥 CHANGÉ: 120s au lieu de 60s
                time.sleep(0.5)
                wait_time += 0.5

            if self.historical_error:
                logger.error(f"❌ Échec récupération historique: {self.historical_error}")
                return None

            if not self.historical_data:
                logger.warning("⚠️ Aucune donnée historique reçue")
                return None

            df_historical = pd.DataFrame(self.historical_data)
            df_historical = df_historical.sort_values('timestamp').reset_index(drop=True)

            df_historical.to_csv(self.csv_file, index=False)
            logger.info(f"💾 {len(df_historical)} points sauvegardés dans {self.csv_file}")

            return df_historical

        except Exception as e:
            logger.error(f"❌ Erreur critique récupération historique: {e}")
            return None


    async def get_latest_data(self, count: int = 2000) -> Optional[pd.DataFrame]:
        """Récupérer les dernières données sous forme de DataFrame"""
        try:
            if len(self.data_buffer) < 10:
                logger.debug("Pas assez de données dans le buffer")
                return None

            recent_data = self.data_buffer[-count:] if len(self.data_buffer) >= count else self.data_buffer

            df = pd.DataFrame(recent_data)

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['price'] = df['price'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)

            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            return None

    def get_multiple_historical_data(self, target_points=50000):
        """Récupérer plusieurs périodes pour atteindre 50K points"""
        try:
            logger.info(f"📊 Collecte {target_points:,} points par périodes...")

            all_data = []
            periods = 10  # 10 périodes de ~36 jours chacune

            for period in range(periods):
                days_end = (period + 1) * 36  # 36 jours par période
                days_start = period * 36

                logger.info(f"   📅 Période {period + 1}/{periods}: {days_start}-{days_end} jours...")

                # Calculer les timestamps
                end_time = int(time.time() - (days_start * 24 * 60 * 60))
                start_time = int(end_time - (36 * 24 * 60 * 60))

                history_request = {
                    "ticks_history": "R_75",
                    "start": start_time,
                    "end": end_time,
                    "style": "ticks",
                    "count": 5000,
                    "req_id": int(time.time() * 1000) + period
                }

                self.historical_data = []
                self.historical_complete = False
                self.historical_error = None

                self.ws.send(json.dumps(history_request))

                # Attendre la réponse
                wait_time = 0
                while not self.historical_complete and wait_time < 30:
                    time.sleep(0.5)
                    wait_time += 0.5

                if self.historical_data and not self.historical_error:
                    all_data.extend(self.historical_data)
                    logger.info(f"   ✅ Collecté: {len(self.historical_data):,} points")
                else:
                    logger.warning(f"   ⚠️ Échec période {period + 1}")

                time.sleep(2)  # Pause entre requêtes

                if len(all_data) >= target_points:
                    break

            if all_data:
                df_combined = pd.DataFrame(all_data)
                df_combined = df_combined.drop_duplicates().sort_values('timestamp')
                logger.info(f"✅ Total collecté: {len(df_combined):,} points")

                # Sauvegarder
                df_combined.to_csv(self.csv_file, index=False)
                return df_combined

            return None

        except Exception as e:
            logger.error(f"❌ Erreur collecte multiple: {e}")
            return None

    def _save_to_csv(self):
        """Sauvegarder les données en CSV"""
        try:
            if not self.data_buffer:
                return

            df = pd.DataFrame(self.data_buffer)

            required_columns = ['timestamp', 'price', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'pip_size',
                                'epoch']

            for col in required_columns:
                if col not in df.columns:
                    if col in ['open', 'close']:
                        df[col] = df['price']
                    elif col in ['high', 'low']:
                        df[col] = df['price']
                    elif col == 'volume':
                        df[col] = 1000
                    elif col == 'symbol':
                        df[col] = 'R_75'
                    elif col == 'pip_size':
                        df[col] = 0.00001
                    elif col == 'epoch':
                        df[col] = 0

            df = df[required_columns]

            if os.path.exists(self.csv_file):
                try:
                    existing_df = pd.read_csv(self.csv_file, nrows=1)
                    existing_columns = list(existing_df.columns)

                    if existing_columns != required_columns:
                        logger.info(f"🔄 Format CSV incompatible, recréation")
                        df.to_csv(self.csv_file, index=False)
                        return

                    existing_df = pd.read_csv(self.csv_file, parse_dates=['timestamp'])
                    last_timestamp = existing_df['timestamp'].max()

                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    new_data = df[df['timestamp'] > last_timestamp]

                    if len(new_data) > 0:
                        new_data.to_csv(self.csv_file, mode='a', header=False, index=False)
                        logger.debug(f"💾 {len(new_data)} nouveaux points sauvegardés")

                except Exception as e:
                    logger.warning(f"Erreur lecture CSV: {e}")
                    df.to_csv(self.csv_file, index=False)
            else:
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
            'tick_buffer_size': len(self.tick_buffer),
            'last_tick': self.last_tick_time.isoformat() if self.last_tick_time else None,
            'reconnect_attempts': self.reconnect_attempts
        }

    async def disconnect(self):
        """Fermer la connexion proprement"""
        try:
            logger.info("🔌 Fermeture de la connexion Deriv...")

            if self.data_buffer:
                self._save_to_csv()

            if self.ws:
                self.ws.close()

            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)

            self.connected = False
            logger.info("✅ Connexion fermée proprement")

        except Exception as e:
            logger.error(f"Erreur fermeture connexion: {e}")