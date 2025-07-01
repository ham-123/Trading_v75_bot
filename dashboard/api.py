#!/usr/bin/env python3
"""
Dashboard API - FastAPI Backend pour Bot Vol75
🚀 API pour exposer toutes les données du bot en temps réel
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import json
import sqlite3
import pandas as pd
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import logging
from contextlib import asynccontextmanager

# Configuration
logger = logging.getLogger(__name__)


class DashboardAPI:
    def __init__(self):
        """Initialisation de l'API Dashboard"""
        self.redis_client = None
        self.db_path = "data/dashboard.db"
        self.init_database()
        self.init_redis()

        # WebSocket connections pour temps réel
        self.active_connections: List[WebSocket] = []

        logger.info("🚀 Dashboard API initialisée")

    def init_database(self):
        """Initialiser la base de données SQLite"""
        try:
            os.makedirs("data", exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Table des signaux
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    tech_score INTEGER NOT NULL,
                    ai_confidence REAL NOT NULL,
                    combined_score REAL NOT NULL,
                    signal_quality TEXT NOT NULL,
                    confluence_score REAL,
                    mtf_strength TEXT,
                    status TEXT DEFAULT 'PENDING',
                    pnl REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table des métriques système
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    bot_status TEXT,
                    deriv_connected BOOLEAN,
                    telegram_connected BOOLEAN,
                    signals_today INTEGER,
                    mtf_rejections INTEGER,
                    ai_accuracy REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table des prix
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    volume REAL NOT NULL,
                    timeframe TEXT DEFAULT 'M5',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()
            conn.close()

            logger.info("✅ Base de données initialisée")

        except Exception as e:
            logger.error(f"❌ Erreur init database: {e}")

    def init_redis(self):
        """Initialiser Redis pour le cache temps réel"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )
            # Test de connexion
            self.redis_client.ping()
            logger.info("✅ Redis connecté")

        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible: {e}")
            self.redis_client = None

    async def add_signal(self, signal_data: Dict):
        """Ajouter un nouveau signal à la base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            mtf_info = signal_data.get('multi_timeframe', {})

            cursor.execute("""
                INSERT INTO signals (
                    timestamp, direction, entry_price, stop_loss, take_profit,
                    tech_score, ai_confidence, combined_score, signal_quality,
                    confluence_score, mtf_strength
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('timestamp'),
                signal_data.get('direction'),
                signal_data.get('entry_price'),
                signal_data.get('stop_loss'),
                signal_data.get('take_profit'),
                signal_data.get('tech_score'),
                signal_data.get('ai_confidence'),
                signal_data.get('combined_score'),
                signal_data.get('signal_quality'),
                mtf_info.get('confluence_score'),
                mtf_info.get('strength')
            ))

            conn.commit()
            conn.close()

            # Broadcast aux WebSockets
            await self.broadcast_signal(signal_data)

            logger.info(f"✅ Signal ajouté: {signal_data.get('direction')} à {signal_data.get('entry_price')}")

        except Exception as e:
            logger.error(f"❌ Erreur ajout signal: {e}")

    async def update_system_metrics(self, metrics: Dict):
        """Mettre à jour les métriques système"""
        try:
            # Base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO system_metrics (
                    timestamp, bot_status, deriv_connected, telegram_connected,
                    signals_today, mtf_rejections, ai_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metrics.get('bot_status', 'UNKNOWN'),
                metrics.get('deriv_connected', False),
                metrics.get('telegram_connected', False),
                metrics.get('signals_today', 0),
                metrics.get('mtf_rejections', 0),
                metrics.get('ai_accuracy', 0)
            ))

            conn.commit()
            conn.close()

            # Cache Redis
            if self.redis_client:
                self.redis_client.setex(
                    "system_metrics",
                    300,  # 5 minutes
                    json.dumps(metrics)
                )

            # Broadcast
            await self.broadcast_metrics(metrics)

        except Exception as e:
            logger.error(f"❌ Erreur update metrics: {e}")

    async def add_price_data(self, price_data: Dict):
        """Ajouter données de prix"""
        try:
            # Cache Redis pour temps réel
            if self.redis_client:
                self.redis_client.setex(
                    "current_price",
                    60,  # 1 minute
                    json.dumps(price_data)
                )

                # Historique court terme (dernières 1000 valeurs)
                self.redis_client.lpush("price_history", json.dumps(price_data))
                self.redis_client.ltrim("price_history", 0, 999)

            # Base de données (échantillonné)
            if datetime.now().minute % 5 == 0:  # Sauver toutes les 5 minutes
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO price_data (timestamp, price, high, low, volume)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    price_data.get('timestamp'),
                    price_data.get('price'),
                    price_data.get('high'),
                    price_data.get('low'),
                    price_data.get('volume')
                ))

                conn.commit()
                conn.close()

            # Broadcast prix temps réel
            await self.broadcast_price(price_data)

        except Exception as e:
            logger.error(f"❌ Erreur ajout prix: {e}")

    async def get_dashboard_data(self) -> Dict:
        """Récupérer toutes les données pour le dashboard"""
        try:
            data = {
                'system_status': await self.get_system_status(),
                'current_price': await self.get_current_price(),
                'recent_signals': await self.get_recent_signals(),
                'technical_indicators': await self.get_technical_indicators(),
                'performance_stats': await self.get_performance_stats(),
                'ai_model_info': await self.get_ai_model_info()
            }
            return data

        except Exception as e:
            logger.error(f"❌ Erreur get dashboard data: {e}")
            return {}

    async def get_system_status(self) -> Dict:
        """Statut système actuel"""
        try:
            # Redis d'abord
            if self.redis_client:
                cached = self.redis_client.get("system_metrics")
                if cached:
                    return json.loads(cached)

            # Sinon base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM system_metrics 
                ORDER BY created_at DESC LIMIT 1
            """)

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'bot_status': row[3],
                    'deriv_connected': bool(row[4]),
                    'telegram_connected': bool(row[5]),
                    'signals_today': row[6],
                    'mtf_rejections': row[7],
                    'ai_accuracy': row[8],
                    'timestamp': row[1]
                }

            return {
                'bot_status': 'UNKNOWN',
                'deriv_connected': False,
                'telegram_connected': False,
                'signals_today': 0,
                'mtf_rejections': 0,
                'ai_accuracy': 0,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Erreur get system status: {e}")
            return {}

    async def get_current_price(self) -> Dict:
        """Prix actuel Vol75"""
        try:
            if self.redis_client:
                cached = self.redis_client.get("current_price")
                if cached:
                    return json.loads(cached)

            # Lire depuis CSV si pas de cache
            if os.path.exists("data/vol75_data.csv"):
                df = pd.read_csv("data/vol75_data.csv")
                if len(df) > 0:
                    latest = df.iloc[-1]
                    return {
                        'price': float(latest['price']),
                        'high': float(latest.get('high', latest['price'])),
                        'low': float(latest.get('low', latest['price'])),
                        'volume': float(latest.get('volume', 1000)),
                        'timestamp': latest.get('timestamp', datetime.now().isoformat())
                    }

            return {'price': 0, 'timestamp': datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"❌ Erreur get current price: {e}")
            return {}

    async def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Signaux récents"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM signals 
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()
            conn.close()

            signals = []
            for row in rows:
                signals.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'direction': row[2],
                    'entry_price': row[3],
                    'stop_loss': row[4],
                    'take_profit': row[5],
                    'tech_score': row[6],
                    'ai_confidence': row[7],
                    'combined_score': row[8],
                    'signal_quality': row[9],
                    'confluence_score': row[10],
                    'mtf_strength': row[11],
                    'status': row[12],
                    'pnl': row[13]
                })

            return signals

        except Exception as e:
            logger.error(f"❌ Erreur get recent signals: {e}")
            return []

    async def get_technical_indicators(self) -> Dict:
        """Indicateurs techniques actuels"""
        try:
            # Lire les indicateurs depuis le fichier de données
            if os.path.exists("data/vol75_data.csv"):
                df = pd.read_csv("data/vol75_data.csv")
                if len(df) >= 50:  # Assez de données pour calculer
                    # Calculer quelques indicateurs de base
                    import ta

                    # RSI
                    rsi = ta.momentum.rsi(df['price'], window=14)

                    # MACD
                    macd_data = ta.trend.MACD(df['price'])

                    # EMA
                    ema_21 = ta.trend.ema_indicator(df['price'], window=21)
                    ema_50 = ta.trend.ema_indicator(df['price'], window=50)

                    # Bollinger
                    bb_data = ta.volatility.BollingerBands(df['price'])

                    return {
                        'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                        'macd': float(macd_data.macd().iloc[-1]) if not pd.isna(macd_data.macd().iloc[-1]) else 0,
                        'macd_signal': float(macd_data.macd_signal().iloc[-1]) if not pd.isna(
                            macd_data.macd_signal().iloc[-1]) else 0,
                        'ema_21': float(ema_21.iloc[-1]) if not pd.isna(ema_21.iloc[-1]) else df['price'].iloc[-1],
                        'ema_50': float(ema_50.iloc[-1]) if not pd.isna(ema_50.iloc[-1]) else df['price'].iloc[-1],
                        'bb_upper': float(bb_data.bollinger_hband().iloc[-1]) if not pd.isna(
                            bb_data.bollinger_hband().iloc[-1]) else df['price'].iloc[-1] * 1.02,
                        'bb_lower': float(bb_data.bollinger_lband().iloc[-1]) if not pd.isna(
                            bb_data.bollinger_lband().iloc[-1]) else df['price'].iloc[-1] * 0.98,
                        'current_price': float(df['price'].iloc[-1]),
                        'timestamp': datetime.now().isoformat()
                    }

            return {}

        except Exception as e:
            logger.error(f"❌ Erreur get technical indicators: {e}")
            return {}

    async def get_performance_stats(self) -> Dict:
        """Statistiques de performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Statistiques générales
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'WIN'")
            winning_signals = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(combined_score) FROM signals")
            avg_score = cursor.fetchone()[0] or 0

            cursor.execute("SELECT SUM(pnl) FROM signals")
            total_pnl = cursor.fetchone()[0] or 0

            # Signaux par qualité
            cursor.execute("""
                SELECT signal_quality, COUNT(*) 
                FROM signals 
                GROUP BY signal_quality
            """)
            quality_stats = dict(cursor.fetchall())

            conn.close()

            win_rate = (winning_signals / total_signals * 100) if total_signals > 0 else 0

            return {
                'total_signals': total_signals,
                'winning_signals': winning_signals,
                'win_rate': round(win_rate, 2),
                'avg_score': round(avg_score, 2),
                'total_pnl': round(total_pnl, 2),
                'quality_distribution': quality_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Erreur get performance stats: {e}")
            return {}

    async def get_ai_model_info(self) -> Dict:
        """Informations du modèle IA"""
        try:
            # Lire les infos du modèle depuis le fichier
            model_files = [
                "data/ensemble_model_info.json",
                "data/optimized_model_info.json",
                "data/model_info.json"
            ]

            for file_path in model_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)

            return {
                'model_type': 'Unknown',
                'validation_accuracy': 0,
                'n_features': 0,
                'training_samples': 0,
                'last_training': None
            }

        except Exception as e:
            logger.error(f"❌ Erreur get AI model info: {e}")
            return {}

    # WebSocket management
    async def connect_websocket(self, websocket: WebSocket):
        """Connecter un WebSocket"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📡 WebSocket connecté: {len(self.active_connections)} clients")

    def disconnect_websocket(self, websocket: WebSocket):
        """Déconnecter un WebSocket"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"📡 WebSocket déconnecté: {len(self.active_connections)} clients")

    async def broadcast_signal(self, signal_data: Dict):
        """Diffuser un nouveau signal"""
        if self.active_connections:
            message = {
                'type': 'new_signal',
                'data': signal_data,
                'timestamp': datetime.now().isoformat()
            }
            await self._broadcast(message)

    async def broadcast_price(self, price_data: Dict):
        """Diffuser nouveau prix"""
        if self.active_connections:
            message = {
                'type': 'price_update',
                'data': price_data,
                'timestamp': datetime.now().isoformat()
            }
            await self._broadcast(message)

    async def broadcast_metrics(self, metrics: Dict):
        """Diffuser métriques système"""
        if self.active_connections:
            message = {
                'type': 'system_metrics',
                'data': metrics,
                'timestamp': datetime.now().isoformat()
            }
            await self._broadcast(message)

    async def _broadcast(self, message: Dict):
        """Diffuser message à tous les clients connectés"""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)

        # Nettoyer les connexions fermées
        for connection in disconnected:
            self.disconnect_websocket(connection)


# Initialisation de l'API
dashboard_api = DashboardAPI()

# Application FastAPI
app = FastAPI(
    title="Vol75 Trading Bot Dashboard API",
    description="API pour le dashboard du bot de trading Vol75",
    version="1.0.0"
)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes API
@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {"message": "Vol75 Trading Bot Dashboard API", "status": "active"}


@app.get("/api/dashboard")
async def get_dashboard():
    """Données complètes du dashboard"""
    return await dashboard_api.get_dashboard_data()


@app.get("/api/system/status")
async def get_system_status():
    """Statut système"""
    return await dashboard_api.get_system_status()


@app.get("/api/price/current")
async def get_current_price():
    """Prix actuel"""
    return await dashboard_api.get_current_price()


@app.get("/api/signals/recent")
async def get_recent_signals(limit: int = 10):
    """Signaux récents"""
    return await dashboard_api.get_recent_signals(limit)


@app.get("/api/indicators")
async def get_technical_indicators():
    """Indicateurs techniques"""
    return await dashboard_api.get_technical_indicators()


@app.get("/api/performance")
async def get_performance():
    """Statistiques de performance"""
    return await dashboard_api.get_performance_stats()


@app.get("/api/ai/model")
async def get_ai_model():
    """Informations modèle IA"""
    return await dashboard_api.get_ai_model_info()


@app.post("/api/signals")
async def add_signal(signal_data: dict):
    """Ajouter un nouveau signal"""
    await dashboard_api.add_signal(signal_data)
    return {"status": "success"}


@app.post("/api/system/metrics")
async def update_metrics(metrics: dict):
    """Mettre à jour les métriques système"""
    await dashboard_api.update_system_metrics(metrics)
    return {"status": "success"}


@app.post("/api/price")
async def add_price(price_data: dict):
    """Ajouter données de prix"""
    await dashboard_api.add_price_data(price_data)
    return {"status": "success"}


# WebSocket pour temps réel
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket pour données temps réel"""
    await dashboard_api.connect_websocket(websocket)
    try:
        while True:
            # Envoyer données périodiquement
            data = await dashboard_api.get_dashboard_data()
            await websocket.send_text(json.dumps({
                'type': 'dashboard_update',
                'data': data,
                'timestamp': datetime.now().isoformat()
            }))
            await asyncio.sleep(30)  # Toutes les 30 secondes

    except WebSocketDisconnect:
        dashboard_api.disconnect_websocket(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)