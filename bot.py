import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import schedule
import time
import asyncio
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.constants import ParseMode
from telegram.error import TelegramError, BadRequest
from dotenv import load_dotenv
import json
import sqlite3
import hashlib
import aiohttp
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Load environment variables first
load_dotenv()

# Import config after loading env
import config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.config.LOG_LEVEL),
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinaryTradingBot:
    def __init__(self):
        self.config = config.config
        self.application = Application.builder().token(self.config.BOT_TOKEN).build()
        self.user_cooldown = {}
        self.setup_handlers()
        self.setup_database()
        self.setup_advanced_features()
        
        # Validate configuration
        try:
            self.config.validate_config()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            raise
    
    def setup_advanced_features(self):
        """Setup advanced features"""
        self.session = None
        self.last_signal_time = {}
        self.user_analytics = {}
        self.premium_users = set()
    
    def setup_database(self):
        """Initialize SQLite database with advanced schema"""
        self.conn = sqlite3.connect(self.config.DATABASE_NAME, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Users table with enhanced fields
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                risk_level TEXT DEFAULT 'medium',
                preferred_pairs TEXT DEFAULT 'EUR/USD,GBP/USD',
                notification_enabled INTEGER DEFAULT 1,
                is_premium INTEGER DEFAULT 0,
                premium_until TIMESTAMP NULL,
                joined_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_signals INTEGER DEFAULT 0,
                successful_signals INTEGER DEFAULT 0,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                language_code TEXT DEFAULT 'en'
            )
        ''')
        
        # Enhanced signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                direction TEXT,
                expiry_time TIMESTAMP,
                confidence REAL,
                price REAL,
                stop_loss REAL,
                take_profit REAL,
                signal_type TEXT DEFAULT 'regular',
                strategy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT NULL,
                actual_result TEXT DEFAULT NULL,
                profit_loss REAL DEFAULT NULL
            )
        ''')
        
        # User signals tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_signals (
                user_id INTEGER,
                signal_id INTEGER,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action_taken TEXT DEFAULT 'viewed',
                result_noted INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
        ''')
        
        # Channel subscription tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channel_subs (
                user_id INTEGER PRIMARY KEY,
                channel_username TEXT,
                subscribed INTEGER DEFAULT 0,
                last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        self.conn.commit()
        logger.info("Database initialized successfully")
    
    def setup_handlers(self):
        """Setup all command and callback handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("signal", self.send_signal))
        self.application.add_handler(CommandHandler("settings", self.settings))
        self.application.add_handler(CommandHandler("stats", self.user_stats))
        self.application.add_handler(CommandHandler("history", self.signal_history))
        self.application.add_handler(CommandHandler("analysis", self.market_analysis))
        self.application.add_handler(CommandHandler("premium", self.premium_info))
        self.application.add_handler(CommandHandler("admin", self.admin_panel))
        self.application.add_handler(CommandHandler("otc", self.otc_market))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        logger.info("Handlers setup completed")
    
    async def check_channel_subscription(self, user_id: int) -> bool:
        """Check if user is subscribed to required channel"""
        if not self.config.CHANNEL_REQUIRED:
            return True
        
        try:
            chat_member = await self.application.bot.get_chat_member(
                self.config.CHANNEL_USERNAME, user_id
            )
            is_subscribed = chat_member.status in ['member', 'administrator', 'creator']
            
            # Update database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO channel_subs (user_id, channel_username, subscribed, last_checked)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (user_id, self.config.CHANNEL_USERNAME, 1 if is_subscribed else 0))
            self.conn.commit()
            
            logger.info(f"User {user_id} subscription status: {is_subscribed}")
            return is_subscribed
            
        except BadRequest as e:
            logger.warning(f"User {user_id} not subscribed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking channel subscription: {e}")
            return False
    
    async def require_subscription(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show subscription required message"""
        keyboard = [
            [InlineKeyboardButton("📢 Join Channel", url=self.config.CHANNEL_LINK)],
            [InlineKeyboardButton("✅ Check Subscription", callback_data="check_subscription")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = self.config.MESSAGES['channel_required']
        
        if update.message:
            await update.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
        else:
            await update.callback_query.message.reply_text(
                message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message after checking subscription"""
        user = update.effective_user
        user_id = user.id
        
        logger.info(f"User {user_id} started the bot")
        
        # Check channel subscription
        if self.config.CHANNEL_REQUIRED:
            is_subscribed = await self.check_channel_subscription(user_id)
            if not is_subscribed:
                await self.require_subscription(update, context)
                return
        
        # Add/update user in database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, username, first_name, last_name, last_active, language_code)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        ''', (user_id, user.username, user.first_name, user.last_name, user.language_code))
        self.conn.commit()
        
        # Send welcome message
        welcome_text = self.config.MESSAGES['welcome']
        
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="get_signal")],
            [InlineKeyboardButton("📊 Market Analysis", callback_data="analysis")],
            [InlineKeyboardButton("💼 OTC Signal", callback_data="otc_signal")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("📈 Statistics", callback_data="stats")],
            [InlineKeyboardButton("💎 Premium Features", callback_data="premium_info")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message with all commands"""
        help_text = f"""
<b>📖 Binary Trading Bot Help</b>

<b>Available Commands:</b>
/start - Start the bot
/signal - Generate trading signal
/otc - OTC Market signals
/settings - Configure preferences  
/stats - View statistics
/history - Signal history
/analysis - Market analysis
/premium - Premium features
/help - This help message

<b>Binary Options Features:</b>
• 1-5 Minute Expiry Signals
• High/Low (CALL/PUT) Predictions
• Real-time Technical Analysis
• Risk Management
• Performance Tracking
• OTC Market Signals

<b>Support:</b> Contact {self.config.ADMIN_USERNAME} for help.

<code>⚠️ Risk Warning: Binary options trading involves high risk. Only trade what you can afford to lose.</code>
        """
        
        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)
    
    async def send_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send binary trading signal"""
        user_id = update.effective_user.id
        
        logger.info(f"User {user_id} requested signal")
        
        # Check subscription
        if self.config.CHANNEL_REQUIRED:
            is_subscribed = await self.check_channel_subscription(user_id)
            if not is_subscribed:
                await self.require_subscription(update, context)
                return
        
        # Check cooldown
        if await self.check_cooldown(user_id):
            await update.message.reply_text("⏳ Please wait a few seconds before requesting another signal.")
            return
        
        # Generate signal
        signal = await self.generate_binary_signal()
        
        # Save signal to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals (pair, direction, expiry_time, confidence, price, stop_loss, take_profit, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (signal['pair'], signal['direction'], signal['expiry_time'], 
              signal['confidence'], signal['current_price'], 
              signal['stop_loss'], signal['take_profit'], signal['strategy']))
        
        signal_id = cursor.lastrowid
        
        # Record user signal
        cursor.execute('INSERT INTO user_signals (user_id, signal_id) VALUES (?, ?)', 
                      (user_id, signal_id))
        
        # Update user stats
        cursor.execute('UPDATE users SET total_signals = total_signals + 1 WHERE user_id = ?', 
                      (user_id,))
        self.conn.commit()
        
        # Format and send signal
        signal_text = await self.format_binary_signal_message(signal)
        
        try:
            chart_path = await self.create_binary_chart(signal)
            with open(chart_path, 'rb') as chart:
                await update.message.reply_photo(
                    photo=chart,
                    caption=signal_text,
                    reply_markup=await self.get_signal_keyboard(signal_id),
                    parse_mode=ParseMode.HTML
                )
            # Clean up chart file
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending chart: {e}")
            await update.message.reply_text(
                signal_text,
                reply_markup=await self.get_signal_keyboard(signal_id),
                parse_mode=ParseMode.HTML
            )
        
        # Update cooldown
        self.user_cooldown[user_id] = time.time()
        
        logger.info(f"Binary Signal {signal_id} sent to user {user_id}")
    
    async def generate_binary_signal(self) -> Dict:
        """Generate binary options trading signal"""
        # Choose between Forex and OTC pairs
        market_type = np.random.choice(['forex', 'otc'], p=[0.7, 0.3])
        
        if market_type == 'forex':
            pair = np.random.choice(self.config.SUPPORTED_PAIRS)
            base_price = np.random.uniform(0.8, 1.2)
        else:
            pair = np.random.choice(self.config.OTC_PAIRS)
            base_price = np.random.uniform(50, 200) if 'Stock' in pair else np.random.uniform(1000, 50000)
        
        # Generate price data
        prices = self.simulate_price_data(base_price=base_price)
        current_price = prices[-1]
        
        # Technical analysis for binary options
        analysis = await self.binary_technical_analysis(prices)
        
        # Determine binary signal (HIGH/LOW)
        signal_data = await self.calculate_binary_signal(prices, analysis)
        
        # Calculate expiry (1-5 minutes for binary options) - FIXED: Convert numpy to int
        expiry_minutes = int(np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
        expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
        
        # For binary options, we don't need stop loss/take profit in traditional sense
        # but we can show potential payout levels
        stop_loss = current_price * 0.98  # 2% below for reference
        take_profit = current_price * 1.02  # 2% above for reference
        
        return {
            'pair': pair,
            'direction': signal_data['direction'],
            'expiry_time': expiry_time,
            'confidence': signal_data['confidence'],
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': signal_data['strategy'],
            'analysis': analysis,
            'prices': prices,
            'market_type': market_type,
            'expiry_minutes': expiry_minutes
        }
    
    async def binary_technical_analysis(self, prices: List[float]) -> Dict:
        """Perform technical analysis for binary options"""
        # RSI
        rsi = self.calculate_rsi(prices, 14)
        
        # Simple moving averages
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # Price momentum
        momentum = (prices[-1] / prices[-5] - 1) * 100 if len(prices) >= 5 else 0
        
        # Support and Resistance levels (simplified)
        recent_high = max(prices[-10:]) if len(prices) >= 10 else prices[-1]
        recent_low = min(prices[-10:]) if len(prices) >= 10 else prices[-1]
        
        # Volatility
        volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.01
        
        # Trend strength
        trend_strength = abs((sma_10 / sma_20 - 1) * 100) if sma_20 != 0 else 0
        
        return {
            'rsi': rsi,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'momentum': momentum,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'trend': 'bullish' if sma_10 > sma_20 else 'bearish'
        }
    
    def simulate_price_data(self, periods: int = 50, base_price: float = 1.0) -> List[float]:
        """Simulate price data for binary options (shorter timeframe)"""
        np.random.seed(int(time.time()))
        
        # Shorter periods for binary options
        periods = min(periods, 50)
        
        # More volatile for binary options
        volatility = 0.01 * (1 + 0.5 * np.sin(np.linspace(0, 4*np.pi, periods)))
        returns = np.random.normal(0.0005, volatility, periods)
        
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return prices
    
    async def calculate_binary_signal(self, prices: List[float], analysis: Dict) -> Dict:
        """Calculate binary signal (HIGH/LOW) using multiple strategies"""
        strategies = []
        current_price = prices[-1]
        
        # RSI Strategy
        if analysis['rsi'] < 35:
            strategies.append(('RSI_Oversold', 'HIGH', 0.72))
        elif analysis['rsi'] > 65:
            strategies.append(('RSI_Overbought', 'LOW', 0.72))
        
        # Trend Following Strategy
        if analysis['trend'] == 'bullish' and analysis['momentum'] > 0.1:
            strategies.append(('Trend_Bullish', 'HIGH', 0.68))
        elif analysis['trend'] == 'bearish' and analysis['momentum'] < -0.1:
            strategies.append(('Trend_Bearish', 'LOW', 0.68))
        
        # Support/Resistance Strategy
        if current_price <= analysis['recent_low'] * 1.005:  # Near support
            strategies.append(('Support_Bounce', 'HIGH', 0.70))
        elif current_price >= analysis['recent_high'] * 0.995:  # Near resistance
            strategies.append(('Resistance_Test', 'LOW', 0.70))
        
        # Momentum Strategy
        if analysis['momentum'] > 0.5:
            strategies.append(('Strong_Momentum_Up', 'HIGH', 0.75))
        elif analysis['momentum'] < -0.5:
            strategies.append(('Strong_Momentum_Down', 'LOW', 0.75))
        
        # Select best strategy
        if strategies:
            best_strategy = max(strategies, key=lambda x: x[2])
            direction = best_strategy[1]
            confidence = best_strategy[2] + np.random.uniform(-0.05, 0.05)
        else:
            # Random signal with lower confidence
            direction = np.random.choice(['HIGH', 'LOW'])
            confidence = np.random.uniform(0.55, 0.65)
            best_strategy = ('Market_Random', direction, confidence)
        
        confidence = max(0.55, min(0.95, confidence))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'strategy': best_strategy[0]
        }
    
    async def format_binary_signal_message(self, signal: Dict) -> str:
        """Format binary signal message using HTML"""
        expiry_str = signal['expiry_time'].strftime('%H:%M:%S')
        confidence_percent = signal['confidence'] * 100
        analysis = signal['analysis']
        market_type = signal.get('market_type', 'forex')
        expiry_minutes = signal.get('expiry_minutes', 3)
        
        market_icon = "💼" if market_type == 'otc' else "💱"
        market_label = "OTC" if market_type == 'otc' else "Forex"
        
        # Determine signal color and icon
        if signal['direction'] == 'HIGH':
            direction_icon = "🟢"
            direction_text = "HIGH (CALL)"
        else:
            direction_icon = "🔴" 
            direction_text = "LOW (PUT)"
        
        message = f"""
<b>🎯 BINARY TRADING SIGNAL</b>

{market_icon} <b>Asset:</b> <code>{signal['pair']}</code> ({market_label})
{direction_icon} <b>Direction:</b> <code>{direction_text}</code>
⏰ <b>Expiry:</b> <code>{expiry_minutes} minute(s) - {expiry_str}</code>
💰 <b>Current Price:</b> <code>{signal['current_price']:.4f}</code>
✅ <b>Confidence:</b> <code>{confidence_percent:.1f}%</code>

<b>📊 Technical Analysis:</b>
• RSI: <code>{analysis['rsi']:.1f}</code>
• Trend: <code>{analysis['trend'].upper()}</code>
• Momentum: <code>{analysis['momentum']:.2f}%</code>
• Volatility: <code>{analysis['volatility']*100:.2f}%</code>

<b>⚡ Strategy:</b> <code>{signal['strategy']}</code>

<b>⏳ Time remaining:</b> {self.calculate_time_remaining(signal['expiry_time'])}

<code>💡 Trade Tip: Enter at current price, expiry in {expiry_minutes} minute(s)</code>

<b>💎 Premium Features:</b> Contact {self.config.ADMIN_USERNAME}
        """
        
        return message.strip()
    
    async def create_binary_chart(self, signal: Dict) -> str:
        """Create binary options chart"""
        try:
            plt.figure(figsize=(10, 6))
            
            prices = signal['prices']
            analysis = signal['analysis']
            
            # Create x-axis labels
            x = list(range(len(prices)))
            
            plt.plot(x, prices, label='Price', color='blue', linewidth=2)
            plt.axhline(y=analysis['sma_10'], color='orange', linestyle='--', 
                       label=f'SMA 10: {analysis["sma_10"]:.4f}')
            plt.axhline(y=analysis['sma_20'], color='red', linestyle='--', 
                       label=f'SMA 20: {analysis["sma_20"]:.4f}')
            plt.axhline(y=signal['current_price'], color='green', linestyle='-', 
                       linewidth=2, label=f'Current: {signal["current_price"]:.4f}')
            
            market_type = signal.get('market_type', 'forex')
            market_label = "OTC" if market_type == 'otc' else "Forex"
            
            direction_text = "HIGH (CALL)" if signal['direction'] == 'HIGH' else "LOW (PUT)"
            
            plt.title(f'BINARY: {signal["pair"]} - {direction_text} | {signal["expiry_minutes"]}min | Conf: {signal["confidence"]*100:.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"binary_chart_{int(time.time())}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            return filename
        except Exception as e:
            logger.error(f"Chart creation error: {e}")
            return "chart_error.png"
    
    async def get_signal_keyboard(self, signal_id: int) -> InlineKeyboardMarkup:
        """Get binary signal action keyboard"""
        keyboard = [
            [
                InlineKeyboardButton("✅ Win", callback_data=f"result_win_{signal_id}"),
                InlineKeyboardButton("❌ Loss", callback_data=f"result_loss_{signal_id}")
            ],
            [
                InlineKeyboardButton("🎯 New Signal", callback_data="get_signal"),
                InlineKeyboardButton("💼 OTC Signal", callback_data="otc_signal")
            ],
            [
                InlineKeyboardButton("📈 Analysis", callback_data="analysis")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    async def check_cooldown(self, user_id: int) -> bool:
        """Check user cooldown"""
        if user_id in self.user_cooldown:
            elapsed = time.time() - self.user_cooldown[user_id]
            if elapsed < self.config.USER_COOLDOWN:
                return True
        return False
    
    async def otc_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate OTC binary signal"""
        user_id = update.effective_user.id
        
        logger.info(f"User {user_id} requested OTC signal")
        
        # Check subscription
        if self.config.CHANNEL_REQUIRED:
            is_subscribed = await self.check_channel_subscription(user_id)
            if not is_subscribed:
                await self.require_subscription(update, context)
                return
        
        # Check cooldown
        if await self.check_cooldown(user_id):
            await update.message.reply_text("⏳ Please wait a few seconds before requesting another signal.")
            return
        
        # Generate OTC signal
        signal = await self.generate_otc_binary_signal()
        
        # Save signal to database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO signals (pair, direction, expiry_time, confidence, price, stop_loss, take_profit, strategy, signal_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (signal['pair'], signal['direction'], signal['expiry_time'], 
              signal['confidence'], signal['current_price'], 
              signal['stop_loss'], signal['take_profit'], signal['strategy'], 'otc'))
        
        signal_id = cursor.lastrowid
        
        # Record user signal
        cursor.execute('INSERT INTO user_signals (user_id, signal_id) VALUES (?, ?)', 
                      (user_id, signal_id))
        
        # Update user stats
        cursor.execute('UPDATE users SET total_signals = total_signals + 1 WHERE user_id = ?', 
                      (user_id,))
        self.conn.commit()
        
        # Format and send signal
        signal_text = await self.format_otc_binary_signal_message(signal)
        
        try:
            chart_path = await self.create_binary_chart(signal)
            with open(chart_path, 'rb') as chart:
                await update.message.reply_photo(
                    photo=chart,
                    caption=signal_text,
                    reply_markup=await self.get_otc_signal_keyboard(signal_id),
                    parse_mode=ParseMode.HTML
                )
            # Clean up chart file
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending OTC chart: {e}")
            await update.message.reply_text(
                signal_text,
                reply_markup=await self.get_otc_signal_keyboard(signal_id),
                parse_mode=ParseMode.HTML
            )
        
        # Update cooldown
        self.user_cooldown[user_id] = time.time()
        
        logger.info(f"OTC Binary Signal {signal_id} sent to user {user_id}")
    
    async def generate_otc_binary_signal(self) -> Dict:
        """Generate OTC binary trading signal"""
        pair = np.random.choice(self.config.OTC_PAIRS)
        
        # Generate price data with appropriate base price
        if 'Stock' in pair:
            base_price = np.random.uniform(50, 200)
        elif 'Crypto' in pair:
            base_price = np.random.uniform(1000, 50000)
        else:
            base_price = np.random.uniform(1, 100)
        
        prices = self.simulate_price_data(base_price=base_price)
        current_price = prices[-1]
        
        # Technical analysis
        analysis = await self.binary_technical_analysis(prices)
        
        # Determine binary signal
        signal_data = await self.calculate_binary_signal(prices, analysis)
        
        # Calculate expiry (1-5 minutes for binary) - FIXED: Convert numpy to int
        expiry_minutes = int(np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
        expiry_time = datetime.now() + timedelta(minutes=expiry_minutes)
        
        # Reference levels
        stop_loss = current_price * 0.98
        take_profit = current_price * 1.02
        
        return {
            'pair': pair,
            'direction': signal_data['direction'],
            'expiry_time': expiry_time,
            'confidence': signal_data['confidence'],
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': signal_data['strategy'],
            'analysis': analysis,
            'prices': prices,
            'market_type': 'otc',
            'expiry_minutes': expiry_minutes
        }
    
    async def format_otc_binary_signal_message(self, signal: Dict) -> str:
        """Format OTC binary signal message using HTML"""
        expiry_str = signal['expiry_time'].strftime('%H:%M:%S')
        confidence_percent = signal['confidence'] * 100
        analysis = signal['analysis']
        expiry_minutes = signal.get('expiry_minutes', 3)
        
        # Determine signal color and icon
        if signal['direction'] == 'HIGH':
            direction_icon = "🟢"
            direction_text = "HIGH (CALL)"
        else:
            direction_icon = "🔴" 
            direction_text = "LOW (PUT)"
        
        message = f"""
<b>💼 OTC BINARY SIGNAL</b>

<b>📈 Asset:</b> <code>{signal['pair']}</code>
{direction_icon} <b>Direction:</b> <code>{direction_text}</code>
⏰ <b>Expiry:</b> <code>{expiry_minutes} minute(s) - {expiry_str}</code>
💰 <b>Current Price:</b> <code>{signal['current_price']:.2f}</code>
✅ <b>Confidence:</b> <code>{confidence_percent:.1f}%</code>

<b>📊 Technical Analysis:</b>
• RSI: <code>{analysis['rsi']:.1f}</code>
• Trend: <code>{analysis['trend'].upper()}</code>
• Momentum: <code>{analysis['momentum']:.2f}%</code>
• Volatility: <code>{analysis['volatility']*100:.2f}%</code>

<b>⚡ Strategy:</b> <code>{signal['strategy']}</code>

<b>⏳ Time remaining:</b> {self.calculate_time_remaining(signal['expiry_time'])}

<code>💡 Trade Tip: Enter at current price, expiry in {expiry_minutes} minute(s)</code>
<code>⚠️ OTC Warning: Higher volatility! Trade carefully!</code>
        """
        
        return message.strip()
    
    async def get_otc_signal_keyboard(self, signal_id: int) -> InlineKeyboardMarkup:
        """Get OTC binary signal action keyboard"""
        keyboard = [
            [
                InlineKeyboardButton("✅ Win", callback_data=f"result_win_{signal_id}"),
                InlineKeyboardButton("❌ Loss", callback_data=f"result_loss_{signal_id}")
            ],
            [
                InlineKeyboardButton("🎯 New Signal", callback_data="get_signal"),
                InlineKeyboardButton("💼 OTC Signal", callback_data="otc_signal")
            ],
            [
                InlineKeyboardButton("📈 Analysis", callback_data="analysis")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show settings menu"""
        user_id = update.effective_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT risk_level, preferred_pairs FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        
        risk_level = result[0] if result else 'medium'
        preferred_pairs = result[1] if result else 'EUR/USD,GBP/USD'
        
        settings_text = f"""
<b>⚙️ Binary Trading Bot Settings</b>

<b>Current Settings:</b>
• 🎯 Risk Level: <code>{risk_level.upper()}</code>
• 💱 Preferred Pairs: <code>{preferred_pairs}</code>
• 🔔 Notifications: <code>ENABLED</code>

<b>Support:</b> {self.config.ADMIN_USERNAME}
        """
        
        keyboard = [
            [InlineKeyboardButton("🎯 Risk Level", callback_data="set_risk")],
            [InlineKeyboardButton("📊 Back to Main", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def user_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show user statistics"""
        user_id = update.effective_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN s.success = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN s.success = 0 THEN 1 ELSE 0 END)
            FROM user_signals us
            JOIN signals s ON us.signal_id = s.id
            WHERE us.user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        total = result[0] if result else 0
        wins = result[1] if result and result[1] is not None else 0
        losses = result[2] if result and result[2] is not None else 0
        
        if total == 0:
            win_rate = 0
        else:
            win_rate = wins / total * 100
        
        cursor.execute('SELECT risk_level, total_signals FROM users WHERE user_id = ?', (user_id,))
        user_info = cursor.fetchone()
        risk_level = user_info[0] if user_info else 'medium'
        total_signals = user_info[1] if user_info else 0
        
        stats_text = f"""
<b>📊 Your Binary Trading Statistics</b>

• 📈 Total Signals Received: <code>{total_signals}</code>
• 🎯 Signals Acted On: <code>{total}</code>
• ✅ Winning Trades: <code>{wins}</code>
• ❌ Losing Trades: <code>{losses}</code>
• 📊 Win Rate: <code>{win_rate:.1f}%</code>
• 🎯 Risk Level: <code>{risk_level.upper()}</code>

<b>Support:</b> {self.config.ADMIN_USERNAME}
        """
        
        await update.message.reply_text(stats_text, parse_mode=ParseMode.HTML)
    
    async def signal_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show signal history"""
        user_id = update.effective_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT s.pair, s.direction, s.expiry_time, s.confidence, s.success, s.strategy, s.signal_type
            FROM user_signals us
            JOIN signals s ON us.signal_id = s.id
            WHERE us.user_id = ?
            ORDER BY us.received_at DESC
            LIMIT 5
        ''', (user_id,))
        
        signals = cursor.fetchall()
        
        if not signals:
            await update.message.reply_text("📜 No signal history found.")
            return
        
        history_text = "<b>📜 Recent Binary Signal History</b>\n\n"
        
        for i, (pair, direction, expiry, confidence, success, strategy, signal_type) in enumerate(signals, 1):
            status = "✅ Win" if success == 1 else "❌ Loss" if success == 0 else "⏳ Pending"
            confidence_pct = confidence * 100
            market_icon = "💼" if signal_type == 'otc' else "💱"
            direction_icon = "🟢" if direction == 'HIGH' else "🔴"
            
            history_text += f"""
<b>Signal #{i}</b> {market_icon}
• 💱 Pair: <code>{pair}</code>
• {direction_icon} Direction: <code>{direction}</code>
• ✅ Confidence: <code>{confidence_pct:.1f}%</code>
• 📊 Result: <code>{status}</code>
• ⚡ Strategy: <code>{strategy}</code>
            """
        
        await update.message.reply_text(history_text, parse_mode=ParseMode.HTML)
    
    async def market_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Provide market analysis"""
        analysis_text = f"""
<b>📈 Binary Trading Market Analysis</b>

<b>Overall Market Sentiment:</b>
• 🔼 <b>BULLISH</b> on major pairs
• 📊 Volatility: Medium-High
• 💹 Trend: Mixed with bullish bias

<b>Key Pairs Analysis:</b>

<b>EUR/USD:</b>
• Trend: Sideways to Bullish
• Key Level: 1.0850
• Recommendation: HIGH (CALL) on dips

<b>GBP/USD:</b>
• Trend: Bullish  
• Key Level: 1.2650
• Recommendation: Strong HIGH (CALL) opportunities

<b>OTC Markets:</b>
• High volatility - perfect for binary options
• Focus on tech stocks and major cryptos
• 1-3 minute expiries recommended

<b>Binary Trading Tips:</b>
1. Focus on 1-5 minute expiries
2. Use HIGH/LOW (CALL/PUT) signals
3. Monitor economic calendar
4. OTC markets for experienced traders

<b>Support:</b> {self.config.ADMIN_USERNAME}

<code>⚠️ Disclaimer: Binary options involve high risk. For educational purposes only.</code>
        """
        
        await update.message.reply_text(analysis_text, parse_mode=ParseMode.HTML)
    
    async def premium_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show premium features"""
        premium_text = self.config.MESSAGES['premium_offer']
        
        keyboard = [
            [InlineKeyboardButton("🚀 Contact Admin", url=f"https://t.me/{self.config.ADMIN_USERNAME.replace('@', '')}")],
            [InlineKeyboardButton("📊 Back to Main", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            premium_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def admin_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Admin panel for bot management"""
        user_id = update.effective_user.id
        
        if user_id not in self.config.ADMIN_IDS:
            await update.message.reply_text("❌ Access denied. Admin only.")
            return
        
        # Get bot statistics
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM signals')
        total_signals = cursor.fetchone()[0]
        
        admin_text = f"""
<b>👑 Admin Panel - Binary Trading Bot</b>

<b>Bot Statistics:</b>
• 👥 Total Users: <code>{total_users}</code>
• 📊 Total Signals: <code>{total_signals}</code>
• 🤖 Bot Status: <code>🟢 RUNNING</code>

<b>Quick Actions:</b>
• /stats - User statistics
• /analysis - Market analysis
• /premium - Premium info

<b>Support:</b> {self.config.ADMIN_USERNAME}
        """
        
        await update.message.reply_text(admin_text, parse_mode=ParseMode.HTML)
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "check_subscription":
            user_id = query.from_user.id
            is_subscribed = await self.check_channel_subscription(user_id)
            
            if is_subscribed:
                await query.edit_message_text("✅ Subscription verified! Use /start to begin.")
            else:
                await self.require_subscription_from_button(query)
        
        elif data.startswith("result_win_"):
            signal_id = int(data.split("_")[2])
            await self.record_signal_result(query, signal_id, True)
        
        elif data.startswith("result_loss_"):
            signal_id = int(data.split("_")[2])
            await self.record_signal_result(query, signal_id, False)
        
        elif data == "get_signal":
            await self.send_signal_from_button(query)
        
        elif data == "otc_signal":
            await self.send_otc_signal_from_button(query)
        
        elif data == "premium_info":
            await self.premium_info_from_button(query)
        
        elif data == "settings":
            await self.settings_from_button(query)
        
        elif data == "stats":
            await self.stats_from_button(query)
        
        elif data == "analysis":
            await self.analysis_from_button(query)
        
        elif data == "main_menu":
            await self.main_menu_from_button(query)
        
        elif data == "set_risk":
            await self.set_risk_level(query)
        
        elif data in ["risk_low", "risk_medium", "risk_high"]:
            await self.update_risk_level(query, data.split('_')[1])
    
    async def require_subscription_from_button(self, query):
        """Show subscription required from button"""
        keyboard = [
            [InlineKeyboardButton("📢 Join Channel", url=self.config.CHANNEL_LINK)],
            [InlineKeyboardButton("✅ Check Subscription", callback_data="check_subscription")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            self.config.MESSAGES['channel_required'],
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def send_signal_from_button(self, query):
        """Send signal from button"""
        user_id = query.from_user.id
        
        if self.config.CHANNEL_REQUIRED:
            is_subscribed = await self.check_channel_subscription(user_id)
            if not is_subscribed:
                await self.require_subscription_from_button(query)
                return
        
        if await self.check_cooldown(user_id):
            await query.edit_message_text("⏳ Please wait a few seconds...")
            return
        
        # Generate signal
        signal = await self.generate_binary_signal()
        signal_text = await self.format_binary_signal_message(signal)
        
        try:
            chart_path = await self.create_binary_chart(signal)
            with open(chart_path, 'rb') as chart:
                await query.message.reply_photo(
                    photo=chart,
                    caption=signal_text,
                    reply_markup=await self.get_signal_keyboard(0),
                    parse_mode=ParseMode.HTML
                )
            # Clean up chart file
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending chart from button: {e}")
            await query.message.reply_text(
                signal_text,
                reply_markup=await self.get_signal_keyboard(0),
                parse_mode=ParseMode.HTML
            )
        
        self.user_cooldown[user_id] = time.time()
    
    async def send_otc_signal_from_button(self, query):
        """Send OTC signal from button"""
        user_id = query.from_user.id
        
        if self.config.CHANNEL_REQUIRED:
            is_subscribed = await self.check_channel_subscription(user_id)
            if not is_subscribed:
                await self.require_subscription_from_button(query)
                return
        
        if await self.check_cooldown(user_id):
            await query.edit_message_text("⏳ Please wait a few seconds...")
            return
        
        # Generate OTC signal
        signal = await self.generate_otc_binary_signal()
        signal_text = await self.format_otc_binary_signal_message(signal)
        
        try:
            chart_path = await self.create_binary_chart(signal)
            with open(chart_path, 'rb') as chart:
                await query.message.reply_photo(
                    photo=chart,
                    caption=signal_text,
                    reply_markup=await self.get_otc_signal_keyboard(0),
                    parse_mode=ParseMode.HTML
                )
            # Clean up chart file
            if os.path.exists(chart_path):
                os.remove(chart_path)
        except Exception as e:
            logger.error(f"Error sending OTC chart from button: {e}")
            await query.message.reply_text(
                signal_text,
                reply_markup=await self.get_otc_signal_keyboard(0),
                parse_mode=ParseMode.HTML
            )
        
        self.user_cooldown[user_id] = time.time()
    
    async def premium_info_from_button(self, query):
        """Show premium info from button"""
        premium_text = self.config.MESSAGES['premium_offer']
        
        keyboard = [
            [InlineKeyboardButton("🚀 Contact Admin", url=f"https://t.me/{self.config.ADMIN_USERNAME.replace('@', '')}")],
            [InlineKeyboardButton("📊 Back to Main", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            premium_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def settings_from_button(self, query):
        """Show settings from button"""
        user_id = query.from_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT risk_level, preferred_pairs FROM users WHERE user_id = ?', (user_id,))
        result = cursor.fetchone()
        
        risk_level = result[0] if result else 'medium'
        preferred_pairs = result[1] if result else 'EUR/USD,GBP/USD'
        
        settings_text = f"""
<b>⚙️ Binary Trading Bot Settings</b>

<b>Current Settings:</b>
• 🎯 Risk Level: <code>{risk_level.upper()}</code>
• 💱 Preferred Pairs: <code>{preferred_pairs}</code>
• 🔔 Notifications: <code>ENABLED</code>

<b>Support:</b> {self.config.ADMIN_USERNAME}
        """
        
        keyboard = [
            [InlineKeyboardButton("🎯 Risk Level", callback_data="set_risk")],
            [InlineKeyboardButton("📊 Back to Main", callback_data="main_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def stats_from_button(self, query):
        """Show stats from button"""
        user_id = query.from_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*), 
                   SUM(CASE WHEN s.success = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN s.success = 0 THEN 1 ELSE 0 END)
            FROM user_signals us
            JOIN signals s ON us.signal_id = s.id
            WHERE us.user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        total = result[0] if result else 0
        wins = result[1] if result and result[1] is not None else 0
        losses = result[2] if result and result[2] is not None else 0
        
        if total == 0:
            win_rate = 0
        else:
            win_rate = wins / total * 100
        
        cursor.execute('SELECT risk_level, total_signals FROM users WHERE user_id = ?', (user_id,))
        user_info = cursor.fetchone()
        risk_level = user_info[0] if user_info else 'medium'
        total_signals = user_info[1] if user_info else 0
        
        stats_text = f"""
<b>📊 Your Binary Trading Statistics</b>

• 📈 Total Signals Received: <code>{total_signals}</code>
• 🎯 Signals Acted On: <code>{total}</code>
• ✅ Winning Trades: <code>{wins}</code>
• ❌ Losing Trades: <code>{losses}</code>
• 📊 Win Rate: <code>{win_rate:.1f}%</code>
• 🎯 Risk Level: <code>{risk_level.upper()}</code>

<b>Support:</b> {self.config.ADMIN_USERNAME}
        """
        
        await query.edit_message_text(stats_text, parse_mode=ParseMode.HTML)
    
    async def analysis_from_button(self, query):
        """Show analysis from button"""
        analysis_text = f"""
<b>📈 Binary Trading Market Analysis</b>

<b>Overall Market Sentiment:</b>
• 🔼 <b>BULLISH</b> on major pairs
• 📊 Volatility: Medium-High
• 💹 Trend: Mixed with bullish bias

<b>Key Pairs Analysis:</b>

<b>EUR/USD:</b>
• Trend: Sideways to Bullish
• Key Level: 1.0850
• Recommendation: HIGH (CALL) on dips

<b>GBP/USD:</b>
• Trend: Bullish  
• Key Level: 1.2650
• Recommendation: Strong HIGH (CALL) opportunities

<b>OTC Markets:</b>
• High volatility - perfect for binary options
• Focus on tech stocks and major cryptos
• 1-3 minute expiries recommended

<b>Binary Trading Tips:</b>
1. Focus on 1-5 minute expiries
2. Use HIGH/LOW (CALL/PUT) signals
3. Monitor economic calendar
4. OTC markets for experienced traders

<b>Support:</b> {self.config.ADMIN_USERNAME}

<code>⚠️ Disclaimer: Binary options involve high risk. For educational purposes only.</code>
        """
        
        await query.edit_message_text(analysis_text, parse_mode=ParseMode.HTML)
    
    async def main_menu_from_button(self, query):
        """Return to main menu from button"""
        welcome_text = self.config.MESSAGES['welcome']
        
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="get_signal")],
            [InlineKeyboardButton("📊 Market Analysis", callback_data="analysis")],
            [InlineKeyboardButton("💼 OTC Signal", callback_data="otc_signal")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("📈 Statistics", callback_data="stats")],
            [InlineKeyboardButton("💎 Premium Features", callback_data="premium_info")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML
        )
    
    async def set_risk_level(self, query):
        """Set risk level"""
        keyboard = [
            [InlineKeyboardButton("🟢 Low Risk", callback_data="risk_low")],
            [InlineKeyboardButton("🟡 Medium Risk", callback_data="risk_medium")],
            [InlineKeyboardButton("🔴 High Risk", callback_data="risk_high")],
            [InlineKeyboardButton("↩️ Back", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text("Select your risk level:", reply_markup=reply_markup)
    
    async def update_risk_level(self, query, risk_level: str):
        """Update risk level"""
        user_id = query.from_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('UPDATE users SET risk_level = ? WHERE user_id = ?', (risk_level, user_id))
        self.conn.commit()
        
        await query.edit_message_text(f"✅ Risk level updated to: {risk_level.upper()}")
    
    async def record_signal_result(self, query, signal_id: int, is_win: bool):
        """Record signal result"""
        user_id = query.from_user.id
        
        cursor = self.conn.cursor()
        cursor.execute('UPDATE signals SET success = ? WHERE id = ?', 
                      (1 if is_win else 0, signal_id))
        
        if is_win:
            cursor.execute('UPDATE users SET successful_signals = successful_signals + 1 WHERE user_id = ?', (user_id,))
        
        cursor.execute('UPDATE user_signals SET result_noted = 1 WHERE user_id = ? AND signal_id = ?', 
                      (user_id, signal_id))
        
        self.conn.commit()
        
        result_text = "✅ Result recorded: WIN" if is_win else "❌ Result recorded: LOSS"
        await query.edit_message_text(result_text)
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        text = update.message.text
        
        if ',' in text and any(pair in text.upper() for pair in ['EUR', 'GBP', 'USD', 'JPY']):
            user_id = update.effective_user.id
            cursor = self.conn.cursor()
            cursor.execute('UPDATE users SET preferred_pairs = ? WHERE user_id = ?', (text, user_id))
            self.conn.commit()
            
            await update.message.reply_text(f"✅ Preferred pairs updated to: {text}")
        else:
            await update.message.reply_text(
                f"🤖 I'm Binary Trading Bot! Use /help for commands.\n\nSupport: {self.config.ADMIN_USERNAME}"
            )
    
    def calculate_time_remaining(self, expiry_time: datetime) -> str:
        """Calculate time remaining"""
        now = datetime.now()
        remaining = expiry_time - now
        total_seconds = int(remaining.total_seconds())
        
        if total_seconds <= 0:
            return "EXPIRED"
        
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def run(self):
        """Start the bot"""
        print("🤖 Binary Trading Bot Starting...")
        print(f"📱 Bot: @{self.config.BOT_USERNAME}")
        print(f"📢 Channel: @{self.config.CHANNEL_USERNAME}")
        print(f"👑 Admin: {self.config.ADMIN_USERNAME}")
        print(f"🔐 Channel Required: {self.config.CHANNEL_REQUIRED}")
        print(f"💎 Premium Enabled: {self.config.PREMIUM_ENABLED}")
        print("=" * 50)
        
        self.application.run_polling()

# Main execution
if __name__ == "__main__":
    try:
        bot = BinaryTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"❌ Bot startup failed: {e}")