"""Exchange Adapter — unified interface for spot + futures operations.

Routes through existing SpotExchangeClient for spot ops, adds futures/perp
support with exchange-specific wallet handling (unified vs split).
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Exchange Registry ──────────────────────────────────────────────────────

EXCHANGE_REGISTRY = {
    "aster": {
        "type": "cex",
        "spot": True,
        "futures": True,
        "unified_wallet": False,  # needs transfer spot↔futures
        "funding_interval_h": 4,
        "maker_fee": 0.0,
        "taker_fee": 0.0004,
        "min_notional": 5.0,
        "quote_currency": "USDT",
    },
    "hyperliquid": {
        "type": "dex",
        "spot": True,
        "futures": True,
        "unified_wallet": True,  # no transfer needed
        "funding_interval_h": 1,
        "maker_fee": 0.0002,
        "taker_fee": 0.0005,
        "min_notional": 10.0,
        "quote_currency": "USDC",
    },
}


# ── Base Adapter ───────────────────────────────────────────────────────────

class BaseExchangeAdapter:
    """Abstract base for exchange operations."""

    def __init__(self, exchange_id: str, ccxt_client, config: dict):
        self.exchange_id = exchange_id
        self.client = ccxt_client  # SpotExchangeClient instance
        self.config = config

    # ── Spot operations ────────────────────────────────────────────────

    def spot_buy(self, symbol: str, qty: float, price: Optional[float] = None) -> dict:
        """Place a spot buy order. Market order if price is None."""
        try:
            if price:
                order = self.client.create_limit_buy(symbol, qty, price)
            else:
                order = self.client.create_market_buy(symbol, qty)
            logger.info("[ADAPTER] spot_buy %s qty=%.6f price=%s → %s",
                       symbol, qty, price, order.get("id", "?"))
            return {"success": True, "order": order}
        except Exception as e:
            logger.error("[ADAPTER] spot_buy FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    def spot_sell(self, symbol: str, qty: float, price: Optional[float] = None) -> dict:
        """Place a spot sell order. Market order if price is None."""
        try:
            if price:
                order = self.client.create_limit_sell(symbol, qty, price)
            else:
                order = self.client.create_market_sell(symbol, qty)
            logger.info("[ADAPTER] spot_sell %s qty=%.6f price=%s → %s",
                       symbol, qty, price, order.get("id", "?"))
            return {"success": True, "order": order}
        except Exception as e:
            logger.error("[ADAPTER] spot_sell FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    # ── Futures operations (override in subclasses) ────────────────────

    def open_short(self, symbol: str, margin: float, leverage: int = 1,
                   price: Optional[float] = None) -> dict:
        """Open a perpetual short position."""
        raise NotImplementedError("Subclass must implement open_short")

    def close_short(self, symbol: str, qty: float,
                    price: Optional[float] = None) -> dict:
        """Close a perpetual short position."""
        raise NotImplementedError("Subclass must implement close_short")

    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for symbol."""
        try:
            exchange = self.client.exchange
            funding = exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0))
        except Exception as e:
            logger.error("[ADAPTER] get_funding_rate FAILED %s: %s", symbol, e)
            return 0.0

    def get_short_position(self, symbol: str) -> dict:
        """Get current short position details."""
        try:
            exchange = self.client.exchange
            positions = exchange.fetch_positions([symbol])
            for pos in positions:
                if pos.get("side") == "short" and float(pos.get("contracts", 0)) > 0:
                    return {
                        "symbol": symbol,
                        "qty": float(pos["contracts"]),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                        "margin": float(pos.get("initialMargin", 0)),
                    }
            return {}
        except Exception as e:
            logger.error("[ADAPTER] get_short_position FAILED %s: %s", symbol, e)
            return {}

    # ── Wallet operations ──────────────────────────────────────────────

    def transfer_to_futures(self, amount: float) -> bool:
        raise NotImplementedError

    def transfer_to_spot(self, amount: float) -> bool:
        raise NotImplementedError

    @property
    def is_paper(self) -> bool:
        return False


# ── Paper Adapter ──────────────────────────────────────────────────────────

class PaperAdapter(BaseExchangeAdapter):
    """Virtual execution — tracks positions without placing real orders."""

    def __init__(self, exchange_id: str, ccxt_client, config: dict):
        super().__init__(exchange_id, ccxt_client, config)
        self._virtual_shorts: Dict[str, dict] = {}

    @property
    def is_paper(self) -> bool:
        return True

    def spot_buy(self, symbol: str, qty: float, price: Optional[float] = None) -> dict:
        logger.debug("[PAPER] spot_buy %s qty=%.6f price=%s", symbol, qty, price)
        return {"success": True, "order": {"id": "paper", "symbol": symbol,
                "side": "buy", "amount": qty, "price": price}}

    def spot_sell(self, symbol: str, qty: float, price: Optional[float] = None) -> dict:
        logger.debug("[PAPER] spot_sell %s qty=%.6f price=%s", symbol, qty, price)
        return {"success": True, "order": {"id": "paper", "symbol": symbol,
                "side": "sell", "amount": qty, "price": price}}

    def open_short(self, symbol: str, margin: float, leverage: int = 1,
                   price: Optional[float] = None) -> dict:
        self._virtual_shorts[symbol] = {
            "margin": margin, "leverage": leverage, "entry_price": price,
            "qty": (margin * leverage) / price if price else 0,
        }
        logger.debug("[PAPER] open_short %s margin=$%.2f lev=%d", symbol, margin, leverage)
        return {"success": True, "paper": True}

    def close_short(self, symbol: str, qty: float,
                    price: Optional[float] = None) -> dict:
        self._virtual_shorts.pop(symbol, None)
        logger.debug("[PAPER] close_short %s qty=%.6f", symbol, qty)
        return {"success": True, "paper": True}

    def get_funding_rate(self, symbol: str) -> float:
        return 0.0

    def get_short_position(self, symbol: str) -> dict:
        return self._virtual_shorts.get(symbol, {})

    def transfer_to_futures(self, amount: float) -> bool:
        return True

    def transfer_to_spot(self, amount: float) -> bool:
        return True


# ── Unified Wallet Adapter (Hyperliquid, Bybit) ───────────────────────────

class UnifiedWalletAdapter(BaseExchangeAdapter):
    """For exchanges with unified spot+futures wallet — no transfers needed."""

    def open_short(self, symbol: str, margin: float, leverage: int = 1,
                   price: Optional[float] = None) -> dict:
        try:
            exchange = self.client.exchange
            # Set leverage
            try:
                exchange.set_leverage(leverage, symbol)
            except Exception:
                pass  # Some exchanges don't support dynamic leverage setting

            qty = (margin * leverage) / price if price else margin
            params = {"type": "market"}
            if price:
                order = exchange.create_order(symbol, "limit", "sell", qty, price,
                                              params={"reduceOnly": False})
            else:
                order = exchange.create_order(symbol, "market", "sell", qty,
                                              params={"reduceOnly": False})
            logger.info("[UNIFIED] open_short %s qty=%.6f margin=$%.2f → %s",
                       symbol, qty, margin, order.get("id", "?"))
            return {"success": True, "order": order}
        except Exception as e:
            logger.error("[UNIFIED] open_short FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    def close_short(self, symbol: str, qty: float,
                    price: Optional[float] = None) -> dict:
        try:
            exchange = self.client.exchange
            if price:
                order = exchange.create_order(symbol, "limit", "buy", qty, price,
                                              params={"reduceOnly": True})
            else:
                order = exchange.create_order(symbol, "market", "buy", qty,
                                              params={"reduceOnly": True})
            logger.info("[UNIFIED] close_short %s qty=%.6f → %s",
                       symbol, qty, order.get("id", "?"))
            return {"success": True, "order": order}
        except Exception as e:
            logger.error("[UNIFIED] close_short FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    def transfer_to_futures(self, amount: float) -> bool:
        return True  # Unified wallet — no-op

    def transfer_to_spot(self, amount: float) -> bool:
        return True  # Unified wallet — no-op


# ── Split Wallet Adapter (Aster/Binance) ──────────────────────────────────

class SplitWalletAdapter(BaseExchangeAdapter):
    """For exchanges with separate spot/futures wallets — requires transfers."""

    def _transfer(self, amount: float, from_account: str, to_account: str) -> bool:
        """Internal transfer between spot and futures wallets."""
        try:
            exchange = self.client.exchange
            quote = self.config.get("quote_currency", "USDT")
            exchange.transfer(quote, amount, from_account, to_account)
            logger.info("[SPLIT] transfer $%.2f %s → %s", amount, from_account, to_account)
            return True
        except Exception as e:
            logger.error("[SPLIT] transfer FAILED $%.2f %s→%s: %s",
                        amount, from_account, to_account, e)
            return False

    def transfer_to_futures(self, amount: float) -> bool:
        return self._transfer(amount, "spot", "future")

    def transfer_to_spot(self, amount: float) -> bool:
        return self._transfer(amount, "future", "spot")

    def open_short(self, symbol: str, margin: float, leverage: int = 1,
                   price: Optional[float] = None) -> dict:
        try:
            # Step 1: Transfer margin to futures wallet
            if not self.transfer_to_futures(margin):
                return {"success": False, "error": "Transfer to futures failed"}

            exchange = self.client.exchange
            # Step 2: Set leverage on futures
            try:
                exchange.set_leverage(leverage, symbol, params={"marginType": "ISOLATED"})
            except Exception as e:
                logger.warning("[SPLIT] set_leverage warning: %s", e)

            # Step 3: Place short order via /fapi/v1/ (Binance-compatible)
            qty = (margin * leverage) / price if price else margin
            if price:
                order = exchange.create_order(symbol, "limit", "sell", qty, price,
                                              params={"reduceOnly": False, "positionSide": "SHORT"})
            else:
                order = exchange.create_order(symbol, "market", "sell", qty,
                                              params={"reduceOnly": False, "positionSide": "SHORT"})
            logger.info("[SPLIT] open_short %s qty=%.6f margin=$%.2f → %s",
                       symbol, qty, margin, order.get("id", "?"))
            return {"success": True, "order": order}
        except Exception as e:
            # Attempt to transfer margin back on failure
            try:
                self.transfer_to_spot(margin)
            except Exception:
                pass
            logger.error("[SPLIT] open_short FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}

    def close_short(self, symbol: str, qty: float,
                    price: Optional[float] = None) -> dict:
        try:
            exchange = self.client.exchange
            if price:
                order = exchange.create_order(symbol, "limit", "buy", qty, price,
                                              params={"reduceOnly": True, "positionSide": "SHORT"})
            else:
                order = exchange.create_order(symbol, "market", "buy", qty,
                                              params={"reduceOnly": True, "positionSide": "SHORT"})
            logger.info("[SPLIT] close_short %s qty=%.6f → %s",
                       symbol, qty, order.get("id", "?"))

            # Transfer released margin back to spot
            # (best-effort — position PnL settles async on some exchanges)
            try:
                pos = self.get_short_position(symbol)
                if not pos:  # position fully closed
                    bal = exchange.fetch_balance(params={"type": "future"})
                    quote = self.config.get("quote_currency", "USDT")
                    free = float(bal.get(quote, {}).get("free", 0))
                    if free > 1.0:
                        self.transfer_to_spot(free)
            except Exception as e:
                logger.warning("[SPLIT] post-close transfer failed: %s", e)

            return {"success": True, "order": order}
        except Exception as e:
            logger.error("[SPLIT] close_short FAILED %s: %s", symbol, e)
            return {"success": False, "error": str(e)}


# ── Factory ────────────────────────────────────────────────────────────────

def create_adapter(exchange_id: str, ccxt_client=None, paper: bool = False) -> BaseExchangeAdapter:
    """Create the appropriate exchange adapter.

    Args:
        exchange_id: Exchange name (e.g. 'aster', 'hyperliquid')
        ccxt_client: SpotExchangeClient instance (can be None for paper)
        paper: If True, returns PaperAdapter regardless of exchange
    """
    config = EXCHANGE_REGISTRY.get(exchange_id, EXCHANGE_REGISTRY.get("aster", {}))

    if paper:
        return PaperAdapter(exchange_id, ccxt_client, config)

    if config.get("unified_wallet"):
        return UnifiedWalletAdapter(exchange_id, ccxt_client, config)
    else:
        return SplitWalletAdapter(exchange_id, ccxt_client, config)
