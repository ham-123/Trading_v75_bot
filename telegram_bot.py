#!/usr/bin/env python3
"""
Telegram Bot - Notifications pour signaux Vol75
Envoie des alertes formatées avec emojis et informations détaillées
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
from telegram import Bot
from telegram.error import TelegramError, TimedOut, NetworkError

logger = logging.getLogger(__name__)


class TelegramBot:
    """Classe pour gérer les notifications Telegram"""

    def __init__(self):
        """Initialisation du bot Telegram"""
        self.token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not self.token or not self.chat_id:
            logger.warning("⚠️ Telegram non configuré (TOKEN ou CHAT_ID manquant)")
            self.bot = None
            self.enabled = False
        else:
            self.bot = Bot(token=self.token)
            self.enabled = True
            logger.info("✅ Bot Telegram initialisé")

        # Statistiques
        self.messages_sent = 0
        self.errors_count = 0
        self.last_message_time = None

        # Paramètres de retry
        self.max_retries = 3
        self.retry_delay = 5  # secondes

    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Envoyer un message Telegram simple"""
        if not self.enabled:
            logger.debug("Telegram désactivé, message non envoyé")
            return False

        for attempt in range(self.max_retries):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True
                )

                self.messages_sent += 1
                self.last_message_time = datetime.now()

                logger.debug("✅ Message Telegram envoyé")
                return True

            except TimedOut:
                logger.warning(f"⏱️ Timeout Telegram (tentative {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

            except NetworkError as e:
                logger.warning(f"🌐 Erreur réseau Telegram: {e} (tentative {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

            except TelegramError as e:
                logger.error(f"❌ Erreur Telegram: {e}")
                self.errors_count += 1
                return False

            except Exception as e:
                logger.error(f"❌ Erreur inattendue Telegram: {e}")
                self.errors_count += 1
                return False

        logger.error("❌ Échec d'envoi après toutes les tentatives")
        self.errors_count += 1
        return False

    async def send_signal(self, signal: Dict) -> bool:
        """Envoyer un signal de trading formaté"""
        try:
            if not self.enabled:
                logger.debug("Telegram désactivé, signal non envoyé")
                return False

            # Formatter le message du signal
            message = self._format_signal_message(signal)

            # Envoyer le message
            success = await self.send_message(message)

            if success:
                logger.info(f"📤 Signal envoyé: {signal['direction']} à {signal['entry_price']}")
            else:
                logger.error("❌ Échec envoi signal")

            return success

        except Exception as e:
            logger.error(f"Erreur envoi signal: {e}")
            return False

    def _format_signal_message(self, signal: Dict) -> str:
        """Formatter le message du signal avec emojis et informations"""
        try:
            # Emojis et textes selon la direction
            if signal['direction'] == 'BUY':
                direction_emoji = "🚀"
                direction_text = "ACHAT"
                direction_flag = "🟢"
            else:
                direction_emoji = "📉"
                direction_text = "VENTE"
                direction_flag = "🔴"

            # Calculer les montants en dollars approximatifs
            risk_amount = float(os.getenv('RISK_AMOUNT', 10))
            reward_amount = risk_amount * signal.get('actual_ratio', 3)

            # Timestamp formaté
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Message principal
            message = f"""
{direction_emoji} <b>SIGNAL Vol75</b> {direction_flag}

📊 <b>{direction_text}</b> à <code>{signal['entry_price']}</code>
🛑 SL: <code>{signal['stop_loss']}</code> (-{risk_amount}$)
🎯 TP: <code>{signal['take_profit']}</code> (+{reward_amount:.0f}$)

📈 <b>Analyse:</b>
• IA: {signal['ai_confidence'] * 100:.0f}% ({signal.get('ai_direction', 'N/A')})
• Technique: {signal['tech_score']}/100
• Score combiné: {signal['combined_score']:.0f}/100
• Ratio R:R: 1:{signal.get('actual_ratio', 3):.1f}

📊 <b>Niveaux:</b>
• Stop Loss: {signal.get('stop_loss_pct', 0):.2f}%
• Take Profit: {signal.get('take_profit_pct', 0):.2f}%

🕐 {timestamp}
"""

            # Ajouter le contexte de marché si disponible
            market_context = signal.get('market_conditions', {})
            if market_context:
                trend = market_context.get('trend', 'unknown')
                volatility = market_context.get('volatility', 'normal')

                trend_emoji = self._get_trend_emoji(trend)
                vol_emoji = self._get_volatility_emoji(volatility)

                message += f"""
📋 <b>Marché:</b>
{trend_emoji} Tendance: {trend.capitalize()}
{vol_emoji} Volatilité: {volatility.capitalize()}
"""

                # Ajouter les variations de prix si disponibles
                if 'price_change_5min' in market_context:
                    change_5m = market_context['price_change_5min']
                    change_1h = market_context.get('price_change_1h', 0)

                    emoji_5m = "📈" if change_5m > 0 else "📉" if change_5m < 0 else "➡️"
                    emoji_1h = "📈" if change_1h > 0 else "📉" if change_1h < 0 else "➡️"

                    message += f"""
📊 <b>Variations:</b>
{emoji_5m} 5min: {change_5m:+.2f}%
{emoji_1h} 1h: {change_1h:+.2f}%
"""

            # Footer avec disclaimer
            message += f"""
⚠️ <i>Trading automatique - Risquez seulement ce que vous pouvez perdre</i>
"""

            return message.strip()

        except Exception as e:
            logger.error(f"Erreur formatage message: {e}")
            # Message de fallback simple
            return f"""
🚀 SIGNAL Vol75
{signal['direction']} à {signal['entry_price']}
SL: {signal['stop_loss']} | TP: {signal['take_profit']}
Score: {signal.get('combined_score', 0):.0f}/100
"""

    def _get_trend_emoji(self, trend: str) -> str:
        """Obtenir l'emoji pour la tendance"""
        trend_emojis = {
            'uptrend': '📈',
            'downtrend': '📉',
            'sideways': '➡️',
            'unknown': '❓'
        }
        return trend_emojis.get(trend, '❓')

    def _get_volatility_emoji(self, volatility: str) -> str:
        """Obtenir l'emoji pour la volatilité"""
        vol_emojis = {
            'low': '😴',
            'normal': '📊',
            'elevated': '⚡',
            'high': '🌪️'
        }
        return vol_emojis.get(volatility, '📊')

    async def send_startup_message(self) -> bool:
        """Envoyer message de démarrage du bot"""
        startup_msg = f"""
🤖 <b>Bot Vol75 Trading - DÉMARRÉ</b>

🔧 <b>Configuration:</b>
• Mode: {os.getenv('TRADING_MODE', 'demo').upper()}
• Capital: {os.getenv('CAPITAL', 1000)}$
• Risque/trade: {os.getenv('RISK_AMOUNT', 10)}$
• Ratio R:R: 1:{os.getenv('RISK_REWARD_RATIO', 3)}

📊 <b>Critères signaux:</b>
• Score technique min: {os.getenv('MIN_TECH_SCORE', 70)}/100
• Confiance IA min: {float(os.getenv('MIN_AI_CONFIDENCE', 0.75)) * 100:.0f}%

🕐 <i>Démarré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</i>

✅ <b>Bot actif - Surveillance en cours...</b>
"""
        return await self.send_message(startup_msg)

    async def send_shutdown_message(self) -> bool:
        """Envoyer message d'arrêt du bot"""
        shutdown_msg = f"""
🛑 <b>Bot Vol75 Trading - ARRÊTÉ</b>

📊 <b>Session terminée:</b>
• Messages envoyés: {self.messages_sent}
• Erreurs: {self.errors_count}
• Dernier message: {self.last_message_time.strftime('%H:%M:%S') if self.last_message_time else 'Aucun'}

🕐 <i>Arrêté le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</i>

⚠️ <b>Surveillance interrompue</b>
"""
        return await self.send_message(shutdown_msg)

    async def send_error_alert(self, error_msg: str, component: str = "Bot") -> bool:
        """Envoyer une alerte d'erreur"""
        alert_msg = f"""
⚠️ <b>ALERTE - {component.upper()}</b>

❌ <b>Erreur détectée:</b>
<code>{error_msg}</code>

🕐 {datetime.now().strftime('%H:%M:%S')}

🔧 <i>Vérifiez les logs pour plus de détails</i>
"""
        return await self.send_message(alert_msg)

    async def send_daily_summary(self, stats: Dict) -> bool:
        """Envoyer un résumé quotidien"""
        try:
            summary_msg = f"""
📊 <b>RÉSUMÉ QUOTIDIEN - Vol75</b>

🎯 <b>Signaux aujourd'hui:</b>
• Total signaux: {stats.get('total_signals', 0)}
• Signaux BUY: {stats.get('buy_signals', 0)}
• Signaux SELL: {stats.get('sell_signals', 0)}

📈 <b>Performance:</b>
• Score moyen: {stats.get('avg_score', 0):.1f}/100
• Confiance IA moy: {stats.get('avg_ai_confidence', 0) * 100:.0f}%
• Win rate: {stats.get('win_rate', 0) * 100:.0f}%

💰 <b>Financier:</b>
• Risque total: {stats.get('total_risk', 0):.0f}$
• Profit potentiel: {stats.get('potential_profit', 0):.0f}$
• P&L réalisé: {stats.get('realized_pnl', 0):+.0f}$

🔧 <b>Technique:</b>
• Messages envoyés: {self.messages_sent}
• Uptime: {stats.get('uptime_hours', 0):.1f}h
• Dernière mise à jour: {datetime.now().strftime('%H:%M:%S')}

🕐 <i>{datetime.now().strftime('%d/%m/%Y')}</i>
"""
            return await self.send_message(summary_msg)

        except Exception as e:
            logger.error(f"Erreur envoi résumé quotidien: {e}")
            return False

    async def send_health_check(self) -> bool:
        """Envoyer un message de vérification de santé"""
        health_msg = f"""
💚 <b>HEALTH CHECK - Bot Vol75</b>

✅ <b>Statut:</b> Opérationnel
🔄 <b>Dernière vérification:</b> {datetime.now().strftime('%H:%M:%S')}
📊 <b>Messages envoyés:</b> {self.messages_sent}
❌ <b>Erreurs:</b> {self.errors_count}

🤖 <i>Bot en fonctionnement normal</i>
"""
        return await self.send_message(health_msg)

    async def test_connection(self) -> bool:
        """Tester la connexion Telegram"""
        try:
            if not self.enabled:
                return False

            test_msg = "🧪 Test de connexion Telegram - Bot Vol75"
            success = await self.send_message(test_msg)

            if success:
                logger.info("✅ Test connexion Telegram réussi")
            else:
                logger.error("❌ Test connexion Telegram échoué")

            return success

        except Exception as e:
            logger.error(f"Erreur test connexion: {e}")
            return False

    def get_stats(self) -> Dict:
        """Obtenir les statistiques du bot Telegram"""
        return {
            'enabled': self.enabled,
            'messages_sent': self.messages_sent,
            'errors_count': self.errors_count,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'success_rate': (self.messages_sent / (self.messages_sent + self.errors_count)) * 100 if (
                                                                                                                 self.messages_sent + self.errors_count) > 0 else 0
        }

    def reset_stats(self):
        """Remettre à zéro les statistiques"""
        self.messages_sent = 0
        self.errors_count = 0
        self.last_message_time = None
        logger.info("📊 Statistiques Telegram remises à zéro")


# Test de la classe si exécuté directement
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)


    async def test_telegram_bot():
        """Test du bot Telegram"""
        bot = TelegramBot()

        if not bot.enabled:
            print("❌ Bot Telegram non configuré")
            return

        print("🧪 Test du bot Telegram...")

        # Test de connexion
        success = await bot.test_connection()
        print(f"Test connexion: {'✅' if success else '❌'}")

        # Test signal
        test_signal = {
            'direction': 'BUY',
            'entry_price': 1234.56,
            'stop_loss': 1224.56,
            'take_profit': 1264.56,
            'tech_score': 78,
            'ai_confidence': 0.85,
            'ai_direction': 'UP',
            'combined_score': 82.5,
            'actual_ratio': 3.0,
            'stop_loss_pct': 0.81,
            'take_profit_pct': 2.43,
            'market_conditions': {
                'trend': 'uptrend',
                'volatility': 'normal',
                'price_change_5min': 0.12,
                'price_change_1h': 0.45
            }
        }

        print("📤 Envoi d'un signal test...")
        signal_success = await bot.send_signal(test_signal)
        print(f"Signal envoyé: {'✅' if signal_success else '❌'}")

        # Test résumé quotidien
        test_stats = {
            'total_signals': 5,
            'buy_signals': 3,
            'sell_signals': 2,
            'avg_score': 76.4,
            'avg_ai_confidence': 0.78,
            'win_rate': 0.6,
            'total_risk': 50,
            'potential_profit': 150,
            'realized_pnl': 25,
            'uptime_hours': 8.5
        }

        print("📊 Envoi d'un résumé test...")
        summary_success = await bot.send_daily_summary(test_stats)
        print(f"Résumé envoyé: {'✅' if summary_success else '❌'}")

        # Statistiques
        stats = bot.get_stats()
        print(f"📈 Statistiques: {stats}")


    asyncio.run(test_telegram_bot())