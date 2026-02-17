"""Universal CCXT-based spot exchange client.

Abstracts exchange-specific details behind a clean interface for the DCA strategy.
Supports Aster, Hyperliquid, and any generic CCXT-compatible exchange.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import ccxt

logger = logging.getLogger(__name__)


class SpotExchangeClient:
    """Universal spot trading client wrapping CCXT."""

    EXCHANGE_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "aster": {
            "env_key": "ASTER_API_KEY",
            "env_secret": "ASTER_API_SECRET",
            "options": {"defaultType": "spot"},
        },
        "hyperliquid": {
            "env_key": "HYPERLIQUID_API_KEY",
            "env_secret": "HYPERLIQUID_API_SECRET",
            "options": {"defaultType": "spot"},
        },
    }

    def __init__(self):
        self.exchange: Optional[ccxt.Exchange] = None
        self.exchange_name: str = ""
        self._markets_loaded: bool = False

    # ── connection ──────────────────────────────────────────────

    def connect(self, exchange_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CCXT exchange instance.

        Args:
            exchange_name: CCXT exchange id (e.g. 'aster', 'hyperliquid', 'binance')
            config: Optional dict with 'apiKey', 'secret', and any extra CCXT options.
                    If omitted, credentials are read from environment variables.
        """
        self.exchange_name = exchange_name.lower()
        defaults = self.EXCHANGE_DEFAULTS.get(self.exchange_name, {})
        config = config or {}

        # Resolve credentials
        api_key = config.get("apiKey") or os.environ.get(defaults.get("env_key", ""), "")
        secret = config.get("secret") or os.environ.get(defaults.get("env_secret", ""), "")

        # If env vars empty, try Windows registry (mirrors aster_trader.py logic)
        if not api_key:
            api_key, secret = self._try_winreg(defaults)

        exchange_class = getattr(ccxt, self.exchange_name, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {self.exchange_name}")

        params: Dict[str, Any] = {
            "enableRateLimit": True,
            "options": {**defaults.get("options", {}), **config.get("options", {})},
        }
        if api_key:
            params["apiKey"] = api_key
            params["secret"] = secret

        # Merge any extra config keys (e.g. 'password', 'uid', 'walletAddress')
        for k, v in config.items():
            if k not in ("apiKey", "secret", "options"):
                params[k] = v

        self.exchange = exchange_class(params)
        self._markets_loaded = False
        logger.info("Connected to %s (authenticated=%s)", self.exchange_name, bool(api_key))

    def _try_winreg(self, defaults: dict) -> Tuple[str, str]:
        """Try reading credentials from Windows registry."""
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
                api_key = winreg.QueryValueEx(k, defaults.get("env_key", ""))[0]
                secret = winreg.QueryValueEx(k, defaults.get("env_secret", ""))[0]
                return api_key, secret
        except Exception:
            return "", ""

    def _ensure_markets(self) -> None:
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    # ── market data ─────────────────────────────────────────────

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current price, bid, ask, volume for a symbol."""
        self._ensure_markets()
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error("fetch_ticker(%s) failed: %s", symbol, e)
            raise

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> List[list]:
        """Get OHLCV candle data. Returns list of [timestamp, O, H, L, C, V]."""
        self._ensure_markets()
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error("fetch_ohlcv(%s, %s) failed: %s", symbol, timeframe, e)
            raise

    def list_spot_markets(self, quote: str = "USDT") -> List[str]:
        """List all spot market symbols with given quote currency."""
        self._ensure_markets()
        return sorted(
            s for s, m in self.exchange.markets.items()
            if m.get("spot") and (not quote or m.get("quote") == quote)
        )

    # ── account data ────────────────────────────────────────────

    def fetch_balance(self, currency: str = "USDT") -> Dict[str, Any]:
        """Get balance for a specific currency. Returns {free, used, total}."""
        self._ensure_markets()
        try:
            bal = self.exchange.fetch_balance()
            c = bal.get(currency, {"free": 0, "used": 0, "total": 0})
            return {"free": c.get("free", 0), "used": c.get("used", 0), "total": c.get("total", 0)}
        except Exception as e:
            logger.error("fetch_balance() failed: %s", e)
            raise

    def fetch_my_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Fetch recent trades for a symbol."""
        self._ensure_markets()
        try:
            return self.exchange.fetch_my_trades(symbol, limit=limit)
        except Exception as e:
            logger.error("fetch_my_trades(%s) failed: %s", symbol, e)
            raise

    # ── order management ────────────────────────────────────────

    def create_limit_buy(self, symbol: str, amount: float, price: float) -> Dict:
        self._ensure_markets()
        logger.info("LIMIT BUY %s qty=%.8f price=%.8f", symbol, amount, price)
        return self.exchange.create_limit_buy_order(symbol, amount, price)

    def create_limit_sell(self, symbol: str, amount: float, price: float) -> Dict:
        self._ensure_markets()
        logger.info("LIMIT SELL %s qty=%.8f price=%.8f", symbol, amount, price)
        return self.exchange.create_limit_sell_order(symbol, amount, price)

    def create_market_buy(self, symbol: str, amount: float) -> Dict:
        self._ensure_markets()
        logger.info("MARKET BUY %s qty=%.8f", symbol, amount)
        return self.exchange.create_market_buy_order(symbol, amount)

    def create_market_sell(self, symbol: str, amount: float) -> Dict:
        self._ensure_markets()
        logger.info("MARKET SELL %s qty=%.8f", symbol, amount)
        return self.exchange.create_market_sell_order(symbol, amount)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        self._ensure_markets()
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error("fetch_open_orders(%s) failed: %s", symbol, e)
            raise

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        self._ensure_markets()
        logger.info("CANCEL order %s on %s", order_id, symbol)
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_order(self, order_id: str, symbol: str) -> Dict:
        self._ensure_markets()
        return self.exchange.fetch_order(order_id, symbol)

    # ── market info helpers ─────────────────────────────────────

    def get_min_order_size(self, symbol: str) -> Dict[str, Any]:
        """Get minimum order size info for a symbol.

        Returns: {min_amount, min_cost, min_price, amount_precision, price_precision}
        """
        self._ensure_markets()
        market = self.exchange.market(symbol)
        limits = market.get("limits", {})
        precision = market.get("precision", {})
        return {
            "min_amount": limits.get("amount", {}).get("min"),
            "min_cost": limits.get("cost", {}).get("min"),
            "min_price": limits.get("price", {}).get("min"),
            "amount_precision": precision.get("amount"),
            "price_precision": precision.get("price"),
        }

    def get_trading_fees(self, symbol: str) -> Dict[str, Any]:
        """Get maker/taker fee rates for a symbol.

        Returns: {maker, taker, percentage}
        """
        self._ensure_markets()
        try:
            fees = self.exchange.fetch_trading_fee(symbol)
            return {"maker": fees.get("maker"), "taker": fees.get("taker"), "percentage": fees.get("percentage", True)}
        except Exception:
            # Fallback to market info
            market = self.exchange.market(symbol)
            return {"maker": market.get("maker"), "taker": market.get("taker"), "percentage": True}
