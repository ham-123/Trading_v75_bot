#!/usr/bin/env python3
"""
Telegram Bot AMÉLIORÉ - Notifications Multi-Timeframes COMPLÈTES
🚀 NOUVEAU: Détails complets M5/M15/H1 + Confluence + Analyse par timeframe
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Optional
from telegram import Bot
from telegram.error import TelegramError, TimedOut, NetworkError

logger = logging.getLogger(__name__)


class EnhancedTelegramBot:
    """Bot Telegram avec notifications Multi-Timeframes COMPLÈTES"""

    def __init__(self):
        """Initialisation du bot Telegram amélioré"""
        self.token = os.getenv('TELEGRAM_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not self.token or not self.chat_id:
            logger.warning("⚠️ Telegram non configuré (TOKEN ou CHAT_ID manquant)")
            self.bot = None
            self.enabled = False
        else:
            self.bot = Bot(token=self.token)
            self.enabled = True
            logger.info("✅ Bot Telegram AMÉLIORÉ initialisé")

        # Statistiques
        self.messages_sent = 0
        self.errors_count = 0
        self.last_message_time = None
        self.max_retries = 3
        self.retry_delay = 5

    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Envoyer un message Telegram avec retry"""
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
        """🚀 NOUVEAU: Envoyer signal avec analyse Multi-Timeframes COMPLÈTE"""
        try:
            if not self.enabled:
                logger.debug("Telegram désactivé, signal non envoyé")
                return False

            # 📊 Message principal avec MTF complet
            main_message = self._format_complete_signal_message(signal)
            success1 = await self.send_message(main_message)

            # 🎯 Message détaillé par timeframe (si confluence élevée)
            mtf_info = signal.get('multi_timeframe', {})
            confluence_score = mtf_info.get('confluence_score', 0)

            if confluence_score >= 0.70:  # Signal fort
                detailed_message = self._format_detailed_mtf_analysis(signal)
                success2 = await self.send_message(detailed_message)
            else:
                success2 = True

            # 📈 Message de contexte de marché avancé
            context_message = self._format_advanced_market_context(signal)
            success3 = await self.send_message(context_message)

            overall_success = success1 and success2 and success3

            if overall_success:
                logger.info(f"📤 Signal MTF COMPLET envoyé: {signal['direction']} à {signal['entry_price']}")
            else:
                logger.error("❌ Échec envoi signal MTF complet")

            return overall_success

        except Exception as e:
            logger.error(f"Erreur envoi signal MTF: {e}")
            return False

    def _format_complete_signal_message(self, signal: Dict) -> str:
        """🚀 NOUVEAU: Message principal avec MTF complet"""
        try:
            # Emojis selon direction
            if signal['direction'] == 'BUY':
                direction_emoji = "🚀"
                direction_text = "ACHAT"
                direction_flag = "🟢"
            else:
                direction_emoji = "📉"
                direction_text = "VENTE"
                direction_flag = "🔴"

            # Données de base
            risk_amount = float(os.getenv('RISK_AMOUNT', 10))
            reward_amount = risk_amount * signal.get('actual_ratio', 3)
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Qualité du signal
            signal_quality = signal.get('signal_quality', 'UNKNOWN')
            quality_emoji = self._get_quality_emoji(signal_quality)

            message = f"""
{direction_emoji} <b>SIGNAL Vol75 {signal_quality}</b> {direction_flag}

💎 <b>{direction_text}</b> à <code>{signal['entry_price']}</code>
🛑 SL: <code>{signal['stop_loss']}</code> (-{risk_amount}$)
🎯 TP: <code>{signal['take_profit']}</code> (+{reward_amount:.0f}$)
⚡ R:R: 1:{signal.get('actual_ratio', 3):.1f}

{quality_emoji} <b>Qualité Signal: {signal_quality}</b>

📊 <b>Scores Détaillés:</b>
🧠 IA: {signal['ai_confidence'] * 100:.0f}% ({signal.get('ai_direction', 'N/A')})
📈 Technique: {signal['tech_score']}/100 (orig: {signal.get('original_tech_score', 'N/A')})
🎯 Score Final: {signal['combined_score']:.0f}/100

🕐 {timestamp}
"""

            # 🆕 SECTION MULTI-TIMEFRAMES COMPLÈTE
            mtf_info = signal.get('multi_timeframe', {})
            if mtf_info:
                message += self._format_mtf_summary_section(mtf_info)

            return message.strip()

        except Exception as e:
            logger.error(f"Erreur formatage message principal: {e}")
            return f"🚀 SIGNAL Vol75: {signal['direction']} à {signal['entry_price']}"

    def _format_mtf_summary_section(self, mtf_info: Dict) -> str:
        """🎯 Section résumé Multi-Timeframes"""
        try:
            conf_score = mtf_info.get('confluence_score', 0)
            conf_percentage = mtf_info.get('confluence_percentage', conf_score * 100)
            strength = mtf_info.get('strength', 'unknown')
            mtf_direction = mtf_info.get('direction', 'N/A')

            # Emoji selon la force
            if strength == 'very_strong':
                strength_emoji = "🟢🟢🟢"
                strength_text = "TRÈS FORT"
            elif strength == 'strong':
                strength_emoji = "🟢🟢"
                strength_text = "FORT"
            elif strength == 'moderate':
                strength_emoji = "🟡🟡"
                strength_text = "MODÉRÉ"
            else:
                strength_emoji = "🔴"
                strength_text = "FAIBLE"

            mtf_section = f"""

🎯 <b>ANALYSE MULTI-TIMEFRAMES</b>
{strength_emoji} <b>Confluence: {conf_percentage:.0f}% ({strength_text})</b>
🔄 Direction MTF: <b>{mtf_direction}</b>
"""

            # Détail rapide par timeframe
            timeframes_detail = mtf_info.get('timeframes_detail', {}) or mtf_info.get('timeframes', {})
            if timeframes_detail:
                mtf_section += "\n📊 <b>Vue d'ensemble:</b>\n"

                for tf_name in ['M5', 'M15', 'H1']:
                    if tf_name in timeframes_detail:
                        tf_data = timeframes_detail[tf_name]
                        tf_direction = tf_data.get('direction', 'N/A')
                        tf_score = tf_data.get('score', 0)
                        tf_strength = tf_data.get('strength', 0)

                        # Emoji selon direction
                        if tf_direction == 'BUY':
                            tf_emoji = "🟢"
                        elif tf_direction == 'SELL':
                            tf_emoji = "🔴"
                        else:
                            tf_emoji = "⚪"

                        # Force en pourcentage
                        strength_pct = tf_strength * 100 if isinstance(tf_strength, float) else tf_strength

                        mtf_section += f"{tf_emoji} <b>{tf_name}:</b> {tf_direction} (Score: {tf_score}, Force: {strength_pct:.0f}%)\n"

            return mtf_section

        except Exception as e:
            logger.error(f"Erreur section MTF: {e}")
            return "\n🎯 Multi-Timeframes: Analyse en cours..."

    def _format_detailed_mtf_analysis(self, signal: Dict) -> str:
        """📊 NOUVEAU: Message détaillé par timeframe (signaux forts uniquement)"""
        try:
            mtf_info = signal.get('multi_timeframe', {})
            timeframes_detail = mtf_info.get('timeframes_detail', {}) or mtf_info.get('timeframes', {})

            if not timeframes_detail:
                return ""

            conf_score = mtf_info.get('confluence_score', 0)
            signal_quality = signal.get('signal_quality', 'UNKNOWN')

            message = f"""
📊 <b>ANALYSE DÉTAILLÉE MULTI-TIMEFRAMES</b>
🎯 Signal {signal_quality} - Confluence {conf_score * 100:.0f}%

"""

            # Analyser chaque timeframe en détail
            for tf_name in ['H1', 'M15', 'M5']:  # Ordre d'importance
                if tf_name in timeframes_detail:
                    tf_data = timeframes_detail[tf_name]
                    message += self._format_single_timeframe_analysis(tf_name, tf_data)

            # Résumé de convergence
            message += self._format_convergence_summary(mtf_info, timeframes_detail)

            return message.strip()

        except Exception as e:
            logger.error(f"Erreur analyse détaillée MTF: {e}")
            return ""

    def _format_single_timeframe_analysis(self, tf_name: str, tf_data: Dict) -> str:
        """Analyse d'un timeframe spécifique"""
        try:
            direction = tf_data.get('direction', 'N/A')
            score = tf_data.get('score', 0)
            strength = tf_data.get('strength', 0)
            trend = tf_data.get('trend', 'unknown')
            volatility = tf_data.get('volatility', 'normal')
            rsi = tf_data.get('rsi', 50)
            macd_bullish = tf_data.get('macd_bullish', False)
            price_vs_ema21 = tf_data.get('price_vs_ema21', 0)

            # Emojis
            if direction == 'BUY':
                dir_emoji = "🟢"
            elif direction == 'SELL':
                dir_emoji = "🔴"
            else:
                dir_emoji = "⚪"

            trend_emoji = "📈" if trend in ['uptrend', 'bullish'] else "📉" if trend in ['downtrend', 'bearish'] else "➡️"
            vol_emoji = "🌪️" if volatility == 'high' else "⚡" if volatility == 'elevated' else "📊"
            macd_emoji = "✅" if macd_bullish else "❌"

            # Importance selon le timeframe
            if tf_name == 'H1':
                importance = "🔥 MAJEUR"
                weight = "50%"
            elif tf_name == 'M15':
                importance = "⚡ IMPORTANT"
                weight = "30%"
            else:
                importance = "📊 COURT TERME"
                weight = "20%"

            tf_section = f"""
{dir_emoji} <b>{tf_name} ({weight}) - {importance}</b>
├─ Direction: <b>{direction}</b> (Score: {score})
├─ {trend_emoji} Tendance: {trend.capitalize()}
├─ {vol_emoji} Volatilité: {volatility.capitalize()}
├─ 📊 RSI: {rsi:.0f} {"(Survente)" if rsi < 30 else "(Surachat)" if rsi > 70 else "(Normal)"}
├─ {macd_emoji} MACD: {"Haussier" if macd_bullish else "Baissier"}
├─ 📈 Prix vs EMA21: {price_vs_ema21:+.2f}%
└─ 💪 Force: {strength * 100 if isinstance(strength, float) else strength:.0f}%

"""
            return tf_section

        except Exception as e:
            logger.error(f"Erreur analyse {tf_name}: {e}")
            return f"{tf_name}: Erreur d'analyse\n"

    def _format_convergence_summary(self, mtf_info: Dict, timeframes_detail: Dict) -> str:
        """Résumé de convergence entre timeframes"""
        try:
            summary = mtf_info.get('summary', '')
            direction_votes = mtf_info.get('direction_votes', {})

            conv_section = f"""
🎯 <b>CONVERGENCE MULTI-TIMEFRAMES</b>

📊 <b>Votes par Direction:</b>
"""

            # Afficher les votes
            if direction_votes:
                for direction, votes in direction_votes.items():
                    if direction and votes > 0:
                        vote_pct = votes * 100
                        if direction == 'BUY':
                            emoji = "🟢"
                        elif direction == 'SELL':
                            emoji = "🔴"
                        else:
                            emoji = "⚪"

                        conv_section += f"{emoji} {direction}: {vote_pct:.0f}%\n"

            # Résumé textuel si disponible
            if summary:
                conv_section += f"\n📋 <b>Résumé:</b>\n<i>{summary}</i>\n"

            # Recommandation finale
            conf_score = mtf_info.get('confluence_score', 0)
            if conf_score >= 0.8:
                recommendation = "🔥 SIGNAL PREMIUM - Tous timeframes alignés"
            elif conf_score >= 0.65:
                recommendation = "✅ SIGNAL FORT - Bonne convergence"
            else:
                recommendation = "⚠️ Signal modéré - Confluence limitée"

            conv_section += f"\n🎯 <b>{recommendation}</b>"

            return conv_section

        except Exception as e:
            logger.error(f"Erreur résumé convergence: {e}")
            return "\n🎯 Convergence: Analyse en cours..."

    def _format_advanced_market_context(self, signal: Dict) -> str:
        """📈 NOUVEAU: Contexte de marché avancé"""
        try:
            market_context = signal.get('market_conditions', {})
            current_price = market_context.get('current_price', signal.get('entry_price', 0))

            message = f"""
📈 <b>CONTEXTE DE MARCHÉ AVANCÉ</b>
💰 Prix actuel: <code>{current_price:.5f}</code>

🔍 <b>Conditions Actuelles:</b>
"""

            # Tendance générale
            trend = market_context.get('trend', 'unknown')
            volatility = market_context.get('volatility', 'normal')
            momentum = market_context.get('momentum', 'neutral')

            trend_emoji = self._get_trend_emoji(trend)
            vol_emoji = self._get_volatility_emoji(volatility)
            momentum_emoji = "⚡" if momentum in ['strong', 'bullish'] else "⚫" if momentum == 'bearish' else "🔵"

            message += f"{trend_emoji} Tendance: <b>{trend.capitalize()}</b>\n"
            message += f"{vol_emoji} Volatilité: <b>{volatility.capitalize()}</b>\n"
            message += f"{momentum_emoji} Momentum: <b>{momentum.capitalize()}</b>\n"

            # Variations de prix
            if 'price_change_5min' in market_context:
                change_5m = market_context['price_change_5min']
                emoji_5m = "📈" if change_5m > 0 else "📉" if change_5m < 0 else "➡️"
                message += f"\n📊 <b>Variations Récentes:</b>\n"
                message += f"{emoji_5m} 5min: <b>{change_5m:+.3f}%</b>\n"

            if 'price_change_1h' in market_context:
                change_1h = market_context['price_change_1h']
                emoji_1h = "📈" if change_1h > 0 else "📉" if change_1h < 0 else "➡️"
                message += f"{emoji_1h} 1h: <b>{change_1h:+.3f}%</b>\n"

            # Niveaux de prix
            message += f"\n🎯 <b>Niveaux de Trading:</b>\n"
            message += f"🛑 Stop Loss: {signal.get('stop_loss_pct', 0):.3f}%\n"
            message += f"🎯 Take Profit: {signal.get('take_profit_pct', 0):.3f}%\n"

            # Multiplier TP utilisé
            tp_multiplier = signal.get('tp_multiplier_used', signal.get('actual_ratio', 3))
            message += f"⚡ Multiplicateur TP: x{tp_multiplier:.1f}\n"

            # Horodatage
            message += f"\n🕐 <i>Analyse à {datetime.now().strftime('%H:%M:%S')}</i>"

            return message

        except Exception as e:
            logger.error(f"Erreur contexte marché: {e}")
            return "📈 Contexte de marché: Analyse en cours..."

    def _get_quality_emoji(self, quality: str) -> str:
        """Emoji selon la qualité du signal"""
        quality_emojis = {
            'PREMIUM': '💎',
            'HIGH': '🔥',
            'GOOD': '✅',
            'AVERAGE': '🟡',
            'LOW': '🔴',
            'UNKNOWN': '❓'
        }
        return quality_emojis.get(quality, '❓')

    def _get_trend_emoji(self, trend: str) -> str:
        """Emoji pour la tendance"""
        trend_emojis = {
            'uptrend': '📈',
            'bullish': '📈',
            'downtrend': '📉',
            'bearish': '📉',
            'sideways': '➡️',
            'ranging': '↔️',
            'unknown': '❓'
        }
        return trend_emojis.get(trend, '❓')

    def _get_volatility_emoji(self, volatility: str) -> str:
        """Emoji pour la volatilité"""
        vol_emojis = {
            'low': '😴',
            'normal': '📊',
            'elevated': '⚡',
            'high': '🌪️'
        }
        return vol_emojis.get(volatility, '📊')

    async def send_mtf_health_notification(self, bot_stats: Dict) -> bool:
        """🚀 NOUVEAU: Notification de santé avec détails MTF"""
        try:
            current_time = datetime.now().strftime('%H:%M:%S')
            current_date = datetime.now().strftime('%d/%m/%Y')

            # Stats de base
            uptime_hours = bot_stats.get('uptime_hours', 0)
            uptime_str = f"{uptime_hours:.1f}h" if uptime_hours < 24 else f"{uptime_hours / 24:.1f}j"
            connection_status = "🟢 Connecté" if bot_stats.get('connected', False) else "🔴 Déconnecté"

            # Stats MTF spécifiques
            signals_today = bot_stats.get('signals_today', 0)
            premium_signals = bot_stats.get('premium_signals', 0)
            high_quality_signals = bot_stats.get('high_quality_signals', 0)
            mtf_rejections = bot_stats.get('mtf_rejections', 0)
            mtf_rejection_rate = bot_stats.get('mtf_rejection_rate', 0)
            quality_rate = bot_stats.get('quality_rate', 0)

            # Prix et variation
            current_price = bot_stats.get('current_price', 0)
            price_change = bot_stats.get('price_change_1h', 0)
            price_emoji = "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️"

            message = f"""
💚 <b>BOT VOL75 MULTI-TIMEFRAMES - RAPPORT</b>

🕐 <b>{current_time}</b> • {current_date}

📊 <b>Statut Système:</b>
{connection_status} • Uptime: {uptime_str}
🧠 IA XGBoost: ✅ Opérationnelle
🎯 MTF M5/M15/H1: ✅ Active

📈 <b>Vol75 Actuel:</b>
{price_emoji} Prix: <code>{current_price:.5f}</code>
{price_emoji} Variation 1h: {price_change:+.3f}%

🎯 <b>Signaux Multi-Timeframes:</b>
• 📊 Total: {signals_today}
• 💎 Premium: {premium_signals}
• 🔥 Haute qualité: {high_quality_signals}
• ❌ Rejets MTF: {mtf_rejections} ({mtf_rejection_rate:.0f}%)
• 🏆 Taux qualité: {quality_rate:.0f}%

🔧 <b>Performance MTF:</b>
• Sélectivité: {mtf_rejection_rate:.0f}% rejeté 🎯
• Confluence min: {bot_stats.get('min_confluence', 65):.0f}%
• Messages envoyés: {self.messages_sent}

<i>🚀 Bot MTF actif - Signaux haute confluence uniquement!</i>
"""

            return await self.send_message(message)

        except Exception as e:
            logger.error(f"Erreur notification santé MTF: {e}")
            return False

    async def send_mtf_startup_notification(self, historical_loaded: bool = False, ai_info: Dict = None) -> bool:
        """🚀 NOUVEAU: Notification de démarrage MTF complète"""
        try:
            ai_info = ai_info or {}

            startup_msg = f"""
🚀 <b>BOT VOL75 MULTI-TIMEFRAMES - DÉMARRÉ</b>

🔧 <b>Configuration Avancée:</b>
• Mode: {os.getenv('TRADING_MODE', 'demo').upper()}
• Capital: {os.getenv('CAPITAL', 1000)}$
• Risque/trade: {os.getenv('RISK_AMOUNT', 10)}$
• Ratio R:R: 1:{os.getenv('RISK_REWARD_RATIO', 3)}

🧠 <b>IA XGBoost Optimisée:</b>
• Modèle: {ai_info.get('model_type', 'XGBoost')}
• Features: {ai_info.get('n_features', 45)}+
• Précision: {ai_info.get('validation_accuracy', 0) * 100:.0f}%
• Échantillons: {ai_info.get('training_samples', 0):,}

🎯 <b>Multi-Timeframes M5/M15/H1:</b>
• Confluence minimum: {os.getenv('MIN_CONFLUENCE_SCORE', 65)}%
• Confluence forte: {os.getenv('STRONG_CONFLUENCE_SCORE', 80)}%
• Poids H1: 50% | M15: 30% | M5: 20%
• Filtres qualité: 6 actifs

📈 <b>Sources de Données:</b>
{'✅ Historiques Vol75 chargées (30j)' if historical_loaded else '🔄 Collecte temps réel active'}
✅ WebSocket Deriv temps réel
✅ Indicateurs techniques (19)

📊 <b>Seuils de Signaux:</b>
• Score technique: ≥{os.getenv('MIN_TECH_SCORE', 70)}/100
• Confiance IA: ≥{float(os.getenv('MIN_AI_CONFIDENCE', 0.75)) * 100:.0f}%
• Confluence MTF: ≥{os.getenv('MIN_CONFLUENCE_SCORE', 65)}%

🕐 <i>Démarré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</i>

🚀 <b>Bot MTF ACTIF - Signaux haute confluence uniquement!</b>
📱 <i>Messages détaillés avec analyse complète par timeframe</i>
🔔 <i>Rapport automatique toutes les heures avec stats MTF</i>
"""
            return await self.send_message(startup_msg)

        except Exception as e:
            logger.error(f"Erreur notification démarrage MTF: {e}")
            return False

    async def send_mtf_daily_summary(self, daily_stats: Dict) -> bool:
        """📊 NOUVEAU: Résumé quotidien avec stats MTF"""
        try:
            current_date = datetime.now().strftime('%d/%m/%Y')

            # Stats générales
            total_signals = daily_stats.get('total_signals', 0)
            premium_signals = daily_stats.get('premium_signals', 0)
            high_quality_signals = daily_stats.get('high_quality_signals', 0)
            mtf_rejections = daily_stats.get('mtf_rejections', 0)

            # Stats MTF
            avg_confluence = daily_stats.get('avg_confluence_score', 0) * 100
            best_confluence = daily_stats.get('best_confluence_score', 0) * 100
            h1_signals = daily_stats.get('h1_signals', 0)
            m15_signals = daily_stats.get('m15_signals', 0)
            m5_signals = daily_stats.get('m5_signals', 0)

            # Performance
            win_rate = daily_stats.get('win_rate', 0) * 100
            avg_rr_ratio = daily_stats.get('avg_rr_ratio', 3)

            summary_msg = f"""
📊 <b>RÉSUMÉ QUOTIDIEN MTF - {current_date}</b>

🎯 <b>Signaux Multi-Timeframes:</b>
• 📊 Total générés: {total_signals}
• 💎 Premium (≥80%): {premium_signals}
• 🔥 Haute qualité (≥75%): {high_quality_signals}
• ❌ Rejets MTF: {mtf_rejections}
• 🎯 Taux sélection: {100 - (mtf_rejections / max(1, total_signals + mtf_rejections) * 100):.0f}%

📈 <b>Confluence Multi-Timeframes:</b>
• 📊 Moyenne: {avg_confluence:.0f}%
• 🏆 Meilleure: {best_confluence:.0f}%
• 🎯 Seuil minimum: {os.getenv('MIN_CONFLUENCE_SCORE', 65)}%

⏰ <b>Répartition par Timeframe:</b>
• 🕐 H1 (50%): {h1_signals} signaux
• 🕒 M15 (30%): {m15_signals} signaux  
• 🕕 M5 (20%): {m5_signals} signaux

💰 <b>Performance:</b>
• 🎯 Win Rate: {win_rate:.0f}%
• ⚡ R:R Moyen: 1:{avg_rr_ratio:.1f}
• 💵 P&L: {daily_stats.get('realized_pnl', 0):+.0f}$

🔧 <b>Technique:</b>
• 📱 Messages: {self.messages_sent}
• ⏱️ Uptime: {daily_stats.get('uptime_hours', 0):.1f}h
• 🔄 Taux succès: {self.get_success_rate():.0f}%

🚀 <b>Points Clés MTF:</b>
• Analyse simultanée M5/M15/H1 ✅
• Filtrage intelligent des faux signaux ✅  
• Confluence minimum respectée ✅
• Qualité premium privilégiée ✅

🕐 <i>Rapport du {current_date}</i>

🎯 <b>MTF Bot - Performance optimisée par confluence!</b>
"""
            return await self.send_message(summary_msg)

        except Exception as e:
            logger.error(f"Erreur résumé quotidien MTF: {e}")
            return False

    async def send_mtf_analysis_alert(self, analysis_result: Dict) -> bool:
        """🚨 NOUVEAU: Alerte d'analyse MTF spéciale"""
        try:
            alert_type = analysis_result.get('type', 'general')
            confluence_score = analysis_result.get('confluence_score', 0) * 100
            direction = analysis_result.get('direction', 'N/A')
            strength = analysis_result.get('strength', 'unknown')

            if alert_type == 'high_confluence':
                emoji = "🚨"
                title = "CONFLUENCE ÉLEVÉE DÉTECTÉE"
            elif alert_type == 'divergence':
                emoji = "⚠️"
                title = "DIVERGENCE MULTI-TIMEFRAMES"
            elif alert_type == 'breakthrough':
                emoji = "🔥"
                title = "PERCÉE TECHNIQUE MTF"
            else:
                emoji = "📊"
                title = "ANALYSE MTF SPÉCIALE"

            alert_msg = f"""
{emoji} <b>{title}</b>

🎯 <b>Détection MTF:</b>
• Confluence: {confluence_score:.0f}%
• Direction: <b>{direction}</b>
• Force: <b>{strength.upper()}</b>

📊 <b>Timeframes Impliqués:</b>
"""

            # Détail par timeframe
            timeframes = analysis_result.get('timeframes', {})
            for tf_name, tf_data in timeframes.items():
                tf_direction = tf_data.get('direction', 'N/A')
                tf_score = tf_data.get('score', 0)

                if tf_direction == 'BUY':
                    tf_emoji = "🟢"
                elif tf_direction == 'SELL':
                    tf_emoji = "🔴"
                else:
                    tf_emoji = "⚪"

                alert_msg += f"{tf_emoji} {tf_name}: {tf_direction} (Score: {tf_score})\n"

            alert_msg += f"""
🕐 <i>Détection à {datetime.now().strftime('%H:%M:%S')}</i>

💡 <i>Surveillez l'évolution pour signal potentiel</i>
"""

            return await self.send_message(alert_msg)

        except Exception as e:
            logger.error(f"Erreur alerte analyse MTF: {e}")
            return False

    def get_success_rate(self) -> float:
        """Calculer le taux de succès des messages"""
        total = self.messages_sent + self.errors_count
        if total == 0:
            return 100.0
        return (self.messages_sent / total) * 100

    async def test_mtf_notification(self) -> bool:
        """🧪 Test des notifications MTF"""
        try:
            if not self.enabled:
                return False

            # Signal de test avec MTF complet
            test_signal = {
                'direction': 'BUY',
                'entry_price': 1050.75432,
                'stop_loss': 1048.50123,
                'take_profit': 1057.25891,
                'actual_ratio': 3.2,
                'tech_score': 85,
                'original_tech_score': 78,
                'ai_confidence': 0.87,
                'ai_direction': 'UP',
                'combined_score': 88.5,
                'signal_quality': 'PREMIUM',
                'stop_loss_pct': 0.214,
                'take_profit_pct': 0.685,
                'tp_multiplier_used': 3.2,
                'multi_timeframe': {
                    'confluence_score': 0.82,
                    'confluence_percentage': 82,
                    'strength': 'very_strong',
                    'direction': 'BUY',
                    'timeframes_detail': {
                        'H1': {
                            'direction': 'BUY',
                            'score': 88,
                            'strength': 0.85,
                            'trend': 'uptrend',
                            'volatility': 'normal',
                            'rsi': 58,
                            'macd_bullish': True,
                            'price_vs_ema21': 1.8
                        },
                        'M15': {
                            'direction': 'BUY',
                            'score': 82,
                            'strength': 0.78,
                            'trend': 'uptrend',
                            'volatility': 'elevated',
                            'rsi': 62,
                            'macd_bullish': True,
                            'price_vs_ema21': 1.2
                        },
                        'M5': {
                            'direction': 'BUY',
                            'score': 75,
                            'strength': 0.72,
                            'trend': 'sideways',
                            'volatility': 'normal',
                            'rsi': 55,
                            'macd_bullish': False,
                            'price_vs_ema21': 0.5
                        }
                    },
                    'direction_votes': {'BUY': 0.8, 'SELL': 0.2},
                    'summary': '🟢 H1: BUY (Score: 88, Tendance: uptrend)\n🟢 M15: BUY (Score: 82, Tendance: uptrend)\n🟢 M5: BUY (Score: 75, Tendance: sideways)'
                },
                'market_conditions': {
                    'trend': 'uptrend',
                    'volatility': 'normal',
                    'momentum': 'bullish',
                    'current_price': 1050.75432,
                    'price_change_5min': 0.125,
                    'price_change_1h': 0.458
                }
            }

            test_msg = "🧪 <b>TEST NOTIFICATIONS MTF</b>\n\nEnvoi d'un signal de test complet..."
            await self.send_message(test_msg)

            # Test du signal complet
            success = await self.send_signal(test_signal)

            result_msg = f"✅ Test MTF terminé: {'Succès' if success else 'Échec'}"
            await self.send_message(result_msg)

            return success

        except Exception as e:
            logger.error(f"Erreur test MTF: {e}")
            return False

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

    async def send_ai_training_notification(self, training_results: Dict) -> bool:
        """Notification d'entraînement IA"""
        try:
            accuracy = training_results.get('accuracy', training_results.get('validation_accuracy', 0))
            if isinstance(accuracy, float) and accuracy < 1:
                accuracy = accuracy * 100

            samples = training_results.get('samples', training_results.get('training_samples', 0))
            features = training_results.get('features', training_results.get('n_features', 0))
            model_type = training_results.get('model_type', 'XGBoost')

            training_msg = f"""
🧠 <b>IA VOL75 - ENTRAÎNEMENT TERMINÉ</b>

✅ <b>Modèle {model_type} prêt!</b>

📊 <b>Performance:</b>
• Précision: {accuracy:.1f}%
• Échantillons: {samples:,}
• Features: {features}

🎯 <b>Capacités:</b>
• Prédiction UP/DOWN haute précision
• Score de confiance affiné
• Analyse multi-indicateurs

🚀 <b>Le bot peut maintenant générer des signaux de qualité!</b>
"""
            return await self.send_message(training_msg)

        except Exception as e:
            logger.error(f"Erreur notification IA: {e}")
            return False

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

    def get_mtf_stats(self) -> Dict:
        """Statistiques MTF du bot Telegram"""
        return {
            'enabled': self.enabled,
            'messages_sent': self.messages_sent,
            'errors_count': self.errors_count,
            'last_message_time': self.last_message_time.isoformat() if self.last_message_time else None,
            'success_rate': self.get_success_rate(),
            'mtf_features': [
                'detailed_timeframe_analysis',
                'confluence_scoring',
                'signal_quality_assessment',
                'market_context_analysis',
                'advanced_health_reporting'
            ]
        }


# Wrapper pour compatibilité avec l'ancien code
class TelegramBot(EnhancedTelegramBot):
    """Wrapper pour compatibilité avec le code existant"""
    pass


# Test complet du bot MTF
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)


    async def test_enhanced_telegram():
        """Test complet du bot Telegram MTF"""
        bot = EnhancedTelegramBot()

        if not bot.enabled:
            print("❌ Bot Telegram non configuré")
            print("Configurez TELEGRAM_TOKEN et TELEGRAM_CHAT_ID dans .env")
            return

        print("🧪 Test du Bot Telegram Multi-Timeframes...")

        # Test de base
        basic_test = await bot.test_mtf_notification()
        print(f"Test notifications MTF: {'✅' if basic_test else '❌'}")

        # Test notification de démarrage MTF
        ai_info = {
            'model_type': 'XGBoost-Optimized',
            'n_features': 45,
            'validation_accuracy': 0.87,
            'training_samples': 15000
        }

        startup_test = await bot.send_mtf_startup_notification(
            historical_loaded=True,
            ai_info=ai_info
        )
        print(f"Notification démarrage MTF: {'✅' if startup_test else '❌'}")

        # Test notification de santé MTF
        health_stats = {
            'uptime_hours': 12.5,
            'connected': True,
            'signals_today': 8,
            'premium_signals': 3,
            'high_quality_signals': 2,
            'mtf_rejections': 15,
            'mtf_rejection_rate': 65,
            'quality_rate': 87,
            'current_price': 1052.34567,
            'price_change_1h': 0.234,
            'min_confluence': 65,
            'trading_mode': 'demo'
        }

        health_test = await bot.send_mtf_health_notification(health_stats)
        print(f"Notification santé MTF: {'✅' if health_test else '❌'}")

        # Statistiques finales
        stats = bot.get_mtf_stats()
        print(f"\n📊 Statistiques finales:")
        print(f"   Messages envoyés: {stats['messages_sent']}")
        print(f"   Taux de succès: {stats['success_rate']:.0f}%")
        print(f"   Features MTF: {len(stats['mtf_features'])}")


    # Lancer le test
    asyncio.run(test_enhanced_telegram())