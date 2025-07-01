#!/usr/bin/env python3
"""
Dashboard Vol75 - Interface Streamlit
🚀 Interface web complète pour monitorer le bot de trading
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import numpy as np

# Configuration Streamlit
st.set_page_config(
    page_title="Vol75 Trading Bot Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"


class DashboardApp:
    def __init__(self):
        """Initialisation du dashboard"""
        self.api_url = API_BASE_URL

        # Cache pour éviter trop de requêtes
        if 'last_update' not in st.session_state:
            st.session_state.last_update = 0
            st.session_state.dashboard_data = {}

    def fetch_data(self, endpoint: str, cache_seconds: int = 30):
        """Récupérer données depuis l'API avec cache"""
        try:
            current_time = time.time()
            cache_key = f"cache_{endpoint}"

            # Vérifier cache
            if (cache_key in st.session_state and
                    current_time - st.session_state.get(f"{cache_key}_time", 0) < cache_seconds):
                return st.session_state[cache_key]

            # Récupérer nouvelles données
            response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state[cache_key] = data
                st.session_state[f"{cache_key}_time"] = current_time
                return data
            else:
                st.error(f"Erreur API: {response.status_code}")
                return {}

        except requests.exceptions.ConnectionError:
            st.error("🔴 API non disponible - Démarrez le backend FastAPI")
            return {}
        except Exception as e:
            st.error(f"Erreur: {e}")
            return {}

    def main_dashboard(self):
        """Dashboard principal"""
        st.title("🚀 Vol75 Trading Bot Dashboard")

        # Auto-refresh
        if st.button("🔄 Actualiser", key="refresh_main"):
            st.session_state.clear()
            st.experimental_rerun()

        # Récupérer données principales
        dashboard_data = self.fetch_data("/api/dashboard", cache_seconds=10)

        if not dashboard_data:
            st.warning("⚠️ Aucune donnée disponible")
            return

        # === MÉTRIQUES PRINCIPALES ===
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            status_data = dashboard_data.get('system_status', {})
            bot_status = status_data.get('bot_status', 'UNKNOWN')

            if bot_status == 'RUNNING':
                st.metric("🤖 Bot Status", "🟢 En ligne")
            else:
                st.metric("🤖 Bot Status", "🔴 Hors ligne")

        with col2:
            price_data = dashboard_data.get('current_price', {})
            current_price = price_data.get('price', 0)
            st.metric("💰 Prix Vol75", f"{current_price:.5f}")

        with col3:
            signals_today = status_data.get('signals_today', 0)
            st.metric("📊 Signaux Aujourd'hui", signals_today)

        with col4:
            ai_info = dashboard_data.get('ai_model_info', {})
            ai_accuracy = ai_info.get('validation_accuracy', 0) * 100
            st.metric("🧠 IA Précision", f"{ai_accuracy:.1f}%")

        # === GRAPHIQUE PRIX PRINCIPAL ===
        st.subheader("📈 Prix Vol75 - Temps Réel")

        # Récupérer historique prix
        price_history = self.get_price_history()
        if not price_history.empty:
            fig_price = self.create_price_chart(price_history)
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("📊 Collecte des données de prix en cours...")

        # === DERNIERS SIGNAUX ===
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("🎯 Signaux Récents")
            recent_signals = dashboard_data.get('recent_signals', [])

            if recent_signals:
                signals_df = pd.DataFrame(recent_signals)

                # Formater le DataFrame
                display_df = signals_df[['timestamp', 'direction', 'entry_price',
                                         'signal_quality', 'confluence_score', 'status']].copy()

                # Colorer selon direction
                def color_direction(val):
                    if val == 'BUY':
                        return 'background-color: #90EE90'
                    elif val == 'SELL':
                        return 'background-color: #FFB6C1'
                    return ''

                styled_df = display_df.style.applymap(color_direction, subset=['direction'])
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("🔍 Aucun signal récent")

        with col_right:
            st.subheader("⚙️ Statut Système")

            # Connexions
            deriv_status = "🟢 Connecté" if status_data.get('deriv_connected', False) else "🔴 Déconnecté"
            telegram_status = "🟢 Connecté" if status_data.get('telegram_connected', False) else "🔴 Déconnecté"

            st.write(f"**Deriv API:** {deriv_status}")
            st.write(f"**Telegram:** {telegram_status}")

            # MTF Stats
            mtf_rejections = status_data.get('mtf_rejections', 0)
            st.write(f"**Rejets MTF:** {mtf_rejections}")

            # Modèle IA
            model_type = ai_info.get('model_type', 'Unknown')
            n_features = ai_info.get('n_features', 0)
            st.write(f"**Modèle IA:** {model_type}")
            st.write(f"**Features:** {n_features}")

    def indicators_page(self):
        """Page des indicateurs techniques"""
        st.title("📊 Indicateurs Techniques")

        # Récupérer indicateurs
        indicators = self.fetch_data("/api/indicators", cache_seconds=30)

        if not indicators:
            st.warning("⚠️ Indicateurs non disponibles")
            return

        current_price = indicators.get('current_price', 0)

        # === JAUGES PRINCIPALES ===
        col1, col2, col3 = st.columns(3)

        with col1:
            # RSI Gauge
            rsi = indicators.get('rsi', 50)
            fig_rsi = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=rsi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RSI (14)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightgray"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_rsi.update_layout(height=300)
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_diff = macd - macd_signal

            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(
                x=['MACD', 'Signal', 'Divergence'],
                y=[macd, macd_signal, macd_diff],
                marker_color=['blue', 'red', 'green' if macd_diff > 0 else 'red']
            ))
            fig_macd.update_layout(title="MACD", height=300)
            st.plotly_chart(fig_macd, use_container_width=True)

        with col3:
            # Prix vs EMA
            ema_21 = indicators.get('ema_21', current_price)
            ema_50 = indicators.get('ema_50', current_price)

            fig_ema = go.Figure()
            fig_ema.add_trace(go.Bar(
                x=['Prix', 'EMA 21', 'EMA 50'],
                y=[current_price, ema_21, ema_50],
                marker_color=['blue', 'orange', 'red']
            ))
            fig_ema.update_layout(title="Prix vs EMA", height=300)
            st.plotly_chart(fig_ema, use_container_width=True)

        # === BOLLINGER BANDS ===
        st.subheader("📈 Bollinger Bands")
        bb_upper = indicators.get('bb_upper', current_price * 1.02)
        bb_lower = indicators.get('bb_lower', current_price * 0.98)

        # Position dans les bandes
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔴 Bande Supérieure", f"{bb_upper:.5f}")
            st.metric("🔵 Prix Actuel", f"{current_price:.5f}")
            st.metric("🟢 Bande Inférieure", f"{bb_lower:.5f}")

        with col2:
            # Graphique position BB
            fig_bb = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bb_position,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Position BB (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 20], 'color': "lightgreen"},
                        {'range': [20, 80], 'color': "lightgray"},
                        {'range': [80, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            fig_bb.update_layout(height=300)
            st.plotly_chart(fig_bb, use_container_width=True)

    def signals_page(self):
        """Page des signaux et trading"""
        st.title("🎯 Signaux & Trading")

        # Performance stats
        performance = self.fetch_data("/api/performance", cache_seconds=60)

        if performance:
            # === MÉTRIQUES PERFORMANCE ===
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Total Signaux", performance.get('total_signals', 0))
            with col2:
                st.metric("✅ Signaux Gagnants", performance.get('winning_signals', 0))
            with col3:
                st.metric("📈 Win Rate", f"{performance.get('win_rate', 0):.1f}%")
            with col4:
                st.metric("💰 P&L Total", f"{performance.get('total_pnl', 0):.2f}$")

            # === DISTRIBUTION QUALITÉ ===
            quality_dist = performance.get('quality_distribution', {})
            if quality_dist:
                st.subheader("🏆 Distribution par Qualité")

                fig_quality = px.pie(
                    values=list(quality_dist.values()),
                    names=list(quality_dist.keys()),
                    title="Répartition des Signaux par Qualité"
                )
                st.plotly_chart(fig_quality, use_container_width=True)

        # === HISTORIQUE SIGNAUX ===
        st.subheader("📋 Historique Complet")
        recent_signals = self.fetch_data("/api/signals/recent?limit=50", cache_seconds=30)

        if recent_signals:
            signals_df = pd.DataFrame(recent_signals)

            # Filtres
            col1, col2, col3 = st.columns(3)
            with col1:
                direction_filter = st.selectbox("Direction", ["Tous", "BUY", "SELL"])
            with col2:
                quality_filter = st.selectbox("Qualité", ["Tous", "PREMIUM", "HIGH", "GOOD", "AVERAGE"])
            with col3:
                status_filter = st.selectbox("Statut", ["Tous", "PENDING", "WIN", "LOSS"])

            # Appliquer filtres
            filtered_df = signals_df.copy()
            if direction_filter != "Tous":
                filtered_df = filtered_df[filtered_df['direction'] == direction_filter]
            if quality_filter != "Tous":
                filtered_df = filtered_df[filtered_df['signal_quality'] == quality_filter]
            if status_filter != "Tous":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]

            # Affichage
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)

                # Graphique P&L dans le temps
                if 'pnl' in filtered_df.columns:
                    st.subheader("💹 P&L dans le Temps")
                    filtered_df['cumulative_pnl'] = filtered_df['pnl'].cumsum()

                    fig_pnl = px.line(
                        filtered_df,
                        x='timestamp',
                        y='cumulative_pnl',
                        title="P&L Cumulé"
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("Aucun signal correspondant aux filtres")

    def ai_model_page(self):
        """Page du modèle IA"""
        st.title("🧠 Modèle IA & Analyse")

        # Infos modèle
        ai_info = self.fetch_data("/api/ai/model", cache_seconds=60)

        if ai_info:
            # === INFORMATIONS MODÈLE ===
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Informations Générales")
                st.write(f"**Type:** {ai_info.get('model_type', 'Unknown')}")
                st.write(f"**Version:** {ai_info.get('version', 'N/A')}")
                st.write(f"**Features:** {ai_info.get('n_features', 0)}")
                st.write(f"**Échantillons:** {ai_info.get('training_samples', 0):,}")

                last_training = ai_info.get('last_training')
                if last_training:
                    st.write(f"**Dernier entraînement:** {last_training}")

            with col2:
                st.subheader("🎯 Performance")
                accuracy = ai_info.get('validation_accuracy', 0) * 100

                # Gauge de précision
                fig_accuracy = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=accuracy,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Précision (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightcoral"},
                            {'range': [50, 70], 'color': "lightyellow"},
                            {'range': [70, 85], 'color': "lightgreen"},
                            {'range': [85, 100], 'color': "darkgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_accuracy.update_layout(height=400)
                st.plotly_chart(fig_accuracy, use_container_width=True)

            # === FEATURE IMPORTANCE ===
            feature_importance = ai_info.get('feature_importance', {})
            if feature_importance:
                st.subheader("🔍 Importance des Features")

                # Top 15 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]

                if sorted_features:
                    features_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])

                    fig_features = px.bar(
                        features_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Features les Plus Importantes"
                    )
                    fig_features.update_layout(height=600)
                    st.plotly_chart(fig_features, use_container_width=True)

            # === PARAMÈTRES MODÈLE ===
            if st.expander("🔧 Paramètres Modèle"):
                xgb_params = ai_info.get('xgb_params', {})
                lgb_params = ai_info.get('lgb_params', {})

                if xgb_params:
                    st.subheader("XGBoost Paramètres")
                    st.json(xgb_params)

                if lgb_params:
                    st.subheader("LightGBM Paramètres")
                    st.json(lgb_params)

    def logs_page(self):
        """Page des logs et debug"""
        st.title("🔧 Logs & Debug")

        # Lecture des logs
        log_file = "logs/trading_bot_mtf_optimized.log"

        if st.button("🔄 Actualiser Logs"):
            st.session_state.logs_refreshed = time.time()

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = f.readlines()

            # Filtres logs
            col1, col2 = st.columns(2)
            with col1:
                log_level = st.selectbox("Niveau", ["TOUS", "INFO", "WARNING", "ERROR", "DEBUG"])
            with col2:
                search_term = st.text_input("Rechercher", "")

            # Afficher logs récents
            recent_logs = logs[-500:]  # 500 dernières lignes

            # Filtrer
            if log_level != "TOUS":
                recent_logs = [log for log in recent_logs if log_level in log]

            if search_term:
                recent_logs = [log for log in recent_logs if search_term.lower() in log.lower()]

            # Affichage avec couleurs
            for log_line in reversed(recent_logs[-100:]):  # 100 plus récentes
                if "ERROR" in log_line:
                    st.error(log_line.strip())
                elif "WARNING" in log_line:
                    st.warning(log_line.strip())
                elif "INFO" in log_line:
                    st.info(log_line.strip())
                else:
                    st.text(log_line.strip())

        except FileNotFoundError:
            st.error("📁 Fichier de logs non trouvé")
        except Exception as e:
            st.error(f"Erreur lecture logs: {e}")

    def get_price_history(self):
        """Récupérer historique des prix"""
        try:
            # Essayer de lire depuis le CSV
            if os.path.exists("data/vol75_data.csv"):
                df = pd.read_csv("data/vol75_data.csv")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df.tail(200)  # 200 derniers points
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Erreur lecture historique: {e}")
            return pd.DataFrame()

    def create_price_chart(self, df):
        """Créer graphique de prix avec indicateurs"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Prix Vol75', 'Volume'),
            row_width=[0.7, 0.3]
        )

        # Prix principal
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name='Prix Vol75',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # EMA si calculable
        if len(df) >= 21:
            df['ema_21'] = df['price'].ewm(span=21).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['ema_21'],
                    mode='lines',
                    name='EMA 21',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        # Volume
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['volume'],
                    name='Volume',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )

        fig.update_layout(
            title="Graphique Prix Vol75 - Temps Réel",
            xaxis_title="Temps",
            yaxis_title="Prix",
            height=500,
            showlegend=True
        )

        return fig

    def run(self):
        """Lancer l'application"""
        # Sidebar navigation
        st.sidebar.title("🚀 Navigation")

        pages = {
            "🏠 Dashboard": self.main_dashboard,
            "📊 Indicateurs": self.indicators_page,
            "🎯 Signaux": self.signals_page,
            "🧠 Modèle IA": self.ai_model_page,
            "🔧 Logs": self.logs_page
        }

        selected_page = st.sidebar.selectbox("Choisir une page", list(pages.keys()))

        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)")

        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()

        # Afficher la page sélectionnée
        pages[selected_page]()

        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Vol75 Trading Bot Dashboard**")
        st.sidebar.markdown(f"*Dernière MAJ: {datetime.now().strftime('%H:%M:%S')}*")


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    app = DashboardApp()
    app.run()