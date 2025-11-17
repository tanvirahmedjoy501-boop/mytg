import os
import logging
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Monster Trading Bot"""
    
    def __init__(self):
        # Bot Configuration
        self.BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
        self.BOT_USERNAME = os.getenv('BOT_USERNAME', 'your_bot_username')
        self.ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', '@your_admin_username')
        admin_ids_str = os.getenv('ADMIN_IDS', '123456789')
        self.ADMIN_IDS = [int(x.strip()) for x in admin_ids_str.split(',')] if admin_ids_str else []
        
        # Channel Configuration
        self.CHANNEL_USERNAME = os.getenv('CHANNEL_USERNAME', '@your_channel')
        self.CHANNEL_LINK = os.getenv('CHANNEL_LINK', 'https://t.me/your_channel')
        self.CHANNEL_REQUIRED = os.getenv('CHANNEL_REQUIRED', 'True').lower() == 'true'
        
        # Trading Configuration
        self.SUPPORTED_PAIRS = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
            'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/CHF'
        ]
        
        # OTC Market Configuration
        self.OTC_PAIRS = [
            'AAPL Stock', 'TSLA Stock', 'AMZN Stock', 'GOOGL Stock', 'MSFT Stock',
            'META Stock', 'NFLX Stock', 'NVDA Stock', 'AMD Stock', 'INTC Stock',
            'BTC/USD Crypto', 'ETH/USD Crypto', 'XRP/USD Crypto', 'ADA/USD Crypto',
            'DOT/USD Crypto', 'LTC/USD Crypto', 'BNB/USD Crypto', 'SOL/USD Crypto',
            'Gold OTC', 'Silver OTC', 'Oil OTC', 'Natural Gas OTC',
            'SPX500 Index', 'DJ30 Index', 'NDX100 Index', 'UK100 Index',
            'GER30 Index', 'JPN225 Index', 'AUS200 Index'
        ]
        
        # Signal Configuration
        self.MIN_EXPIRY = 1  # minutes
        self.MAX_EXPIRY = 5  # minutes
        self.MIN_CONFIDENCE = 0.60
        self.MAX_CONFIDENCE = 0.95
        self.USER_COOLDOWN = 10  # seconds
        
        # Technical Analysis Parameters
        self.RSI_PERIOD = 14
        self.SMA_SHORT = 20
        self.SMA_LONG = 50
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        self.BOLLINGER_PERIOD = 20
        self.BOLLINGER_STD = 2
        
        # Database Configuration
        self.DATABASE_NAME = 'monster_trading_bot.db'
        
        # Premium Features
        self.PREMIUM_ENABLED = os.getenv('PREMIUM_ENABLED', 'True').lower() == 'true'
        self.PREMIUM_PRICE_MONTHLY = 29.99
        self.PREMIUM_PRICE_YEARLY = 299.99
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Bot Messages - USING HTML FORMATTING (more reliable)
        self.MESSAGES = {
            'welcome': """
<b>🤖 Welcome to Monster Trading Bot</b>

<b>🚀 Advanced Trading Signals Powered by AI</b>

<b>What I Offer:</b>
• 🎯 High-Accuracy Trading Signals
• 📊 Advanced Technical Analysis  
• ⚡ Real-time Market Insights
• 💰 Risk Management Tools
• 📈 Performance Tracking
• 💼 OTC Market Signals

<b>Get Started:</b>
Use the buttons below to generate signals and analyze markets!

<code>⚠️ Risk Warning: Trading involves substantial risk. Only trade with money you can afford to lose.</code>
            """.strip(),
            
            'premium_offer': """
<b>💎 MONSTER TRADING PREMIUM</b>

<b>🚀 Unlock Advanced Features:</b>
• 🔥 Early Signal Access
• 📊 Advanced Analytics
• 🎯 Higher Confidence Signals
• ⚡ Priority Support
• 📈 Custom Strategies
• 🔔 Real-time Alerts
• 💼 Exclusive OTC Signals

<b>💵 Pricing:</b>
• Monthly: ${:.2f}
• Yearly: ${:.2f} (Save 15%)

<b>📧 Contact {} for premium access!</b>
            """.format(self.PREMIUM_PRICE_MONTHLY, self.PREMIUM_PRICE_YEARLY, self.ADMIN_USERNAME).strip(),
            
            'channel_required': """
<b>📢 Channel Subscription Required</b>

To use Monster Trading Bot, you must join our official channel for updates and announcements.

<b>✅ Requirements:</b>
1. Join our channel: {}
2. Click 'Check Subscription' below
3. Start trading!

Thank you for your cooperation! 🚀
            """.format(self.CHANNEL_LINK).strip()
        }
    
    def validate_config(self):
        """Validate configuration settings"""
        if not self.BOT_TOKEN or self.BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE':
            raise ValueError("BOT_TOKEN is not set in environment variables")
        
        if not self.ADMIN_IDS:
            raise ValueError("ADMIN_IDS must contain at least one admin ID")
        
        if self.CHANNEL_REQUIRED and (not self.CHANNEL_USERNAME or self.CHANNEL_USERNAME == '@your_channel'):
            raise ValueError("CHANNEL_USERNAME must be set when CHANNEL_REQUIRED is True")
        
        if self.MIN_CONFIDENCE < 0 or self.MAX_CONFIDENCE > 1:
            raise ValueError("Confidence values must be between 0 and 1")
        
        if self.MIN_EXPIRY >= self.MAX_EXPIRY:
            raise ValueError("MIN_EXPIRY must be less than MAX_EXPIRY")
        
        if not self.SUPPORTED_PAIRS:
            raise ValueError("SUPPORTED_PAIRS cannot be empty")
        
        if not self.OTC_PAIRS:
            raise ValueError("OTC_PAIRS cannot be empty")
        
        # Validate logging level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL not in valid_log_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_log_levels)}")
        
        logging.info("Configuration validated successfully")

# Create global instance
config = Config()