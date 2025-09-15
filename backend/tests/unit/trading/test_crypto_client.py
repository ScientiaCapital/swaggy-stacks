"""
Unit tests for CryptoClient

Tests cryptocurrency trading operations with Alpaca API integration,
including 24/7 trading support, fractional quantities, and order execution.

Test Coverage:
- Account information retrieval
- Position management and tracking
- Order execution with fractional support
- Order status monitoring and cancellation
- Symbol validation and error handling
- 24/7 trading scenarios
- API error handling and recovery
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
from decimal import Decimal

import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError

from app.trading.crypto_client import CryptoClient
from app.core.exceptions import TradingError, MarketDataError


class TestCryptoClient:
    """Test suite for CryptoClient cryptocurrency trading operations"""

    @pytest.fixture
    def mock_alpaca_api(self):
        """Mock Alpaca API for testing"""
        api = Mock(spec=tradeapi.REST)
        api.get_account = Mock()
        api.list_positions = Mock()
        api.submit_order = Mock()
        api.cancel_order = Mock()
        api.get_order = Mock()
        api.list_orders = Mock()
        return api

    @pytest.fixture
    def mock_alpaca_data_api(self):
        """Mock Alpaca Data API for testing"""
        data_api = Mock(spec=tradeapi.REST)
        return data_api

    @pytest.fixture
    def crypto_client(self, mock_alpaca_api, mock_alpaca_data_api):
        """CryptoClient instance with mocked APIs"""
        with patch('app.trading.crypto_client.tradeapi.REST') as mock_rest:
            mock_rest.side_effect = [mock_alpaca_api, mock_alpaca_data_api]

            client = CryptoClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )
            client.api = mock_alpaca_api
            client.data_api = mock_alpaca_data_api
            return client

    @pytest.fixture
    def sample_crypto_account(self):
        """Sample cryptocurrency account data"""
        account = Mock()
        account.id = "crypto_account_123"
        account.account_number = "12345678"
        account.status = "ACTIVE"
        account.currency = "USD"
        account.buying_power = "50000.00"
        account.non_marginable_buying_power = "50000.00"
        account.crypto_buying_power = "50000.00"
        account.portfolio_value = "75000.00"
        account.created_at = datetime.now(timezone.utc)
        account.trading_blocked = False
        account.crypto_status = "ACTIVE"
        return account

    @pytest.fixture
    def sample_crypto_positions(self):
        """Sample cryptocurrency positions"""
        position1 = Mock()
        position1.symbol = "BTCUSD"
        position1.qty = "1.5"
        position1.market_value = "67500.00"
        position1.avg_entry_price = "45000.00"
        position1.current_price = "45000.00"
        position1.unrealized_pl = "0.00"
        position1.unrealized_plpc = "0.00"
        position1.side = "long"
        position1.asset_class = "crypto"

        position2 = Mock()
        position2.symbol = "ETHUSD"
        position2.qty = "10.0"
        position2.market_value = "30000.00"
        position2.avg_entry_price = "3000.00"
        position2.current_price = "3000.00"
        position2.unrealized_pl = "0.00"
        position2.unrealized_plpc = "0.00"
        position2.side = "long"
        position2.asset_class = "crypto"

        return [position1, position2]

    @pytest.fixture
    def sample_crypto_order(self):
        """Sample cryptocurrency order"""
        order = Mock()
        order.id = "crypto_order_123"
        order.client_order_id = "client_123"
        order.symbol = "BTCUSD"
        order.qty = "0.5"
        order.side = "buy"
        order.order_type = "market"
        order.time_in_force = "gtc"
        order.status = "filled"
        order.limit_price = None
        order.stop_price = None
        order.filled_qty = "0.5"
        order.filled_avg_price = "45000.00"
        order.submitted_at = datetime.now(timezone.utc)
        order.updated_at = datetime.now(timezone.utc)
        return order

    def test_crypto_client_initialization_success(self):
        """Test successful CryptoClient initialization"""
        with patch('app.trading.crypto_client.tradeapi.REST') as mock_rest:
            client = CryptoClient(
                api_key="test_key",
                secret_key="test_secret",
                paper=True
            )

            assert client.api_key == "test_key"
            assert client.secret_key == "test_secret"
            assert client.paper is True
            assert mock_rest.call_count == 2  # api and data_api

    def test_crypto_client_initialization_missing_credentials(self):
        """Test CryptoClient initialization with missing credentials"""
        with patch('app.core.config.settings') as mock_settings:
            mock_settings.ALPACA_API_KEY = None
            mock_settings.ALPACA_SECRET_KEY = None

            with pytest.raises(TradingError) as exc_info:
                CryptoClient()

            assert "credentials not provided" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_crypto_account_success(self, crypto_client, sample_crypto_account):
        """Test successful crypto account retrieval"""
        crypto_client.api.get_account.return_value = sample_crypto_account

        account = await crypto_client.get_crypto_account()

        assert account["id"] == "crypto_account_123"
        assert account["account_number"] == "12345678"
        assert account["status"] == "ACTIVE"
        assert account["buying_power"] == 50000.0
        assert account["crypto_buying_power"] == 50000.0
        assert account["portfolio_value"] == 75000.0
        assert account["trading_blocked"] is False
        assert account["crypto_status"] == "ACTIVE"

    @pytest.mark.asyncio
    async def test_get_crypto_account_api_error(self, crypto_client):
        """Test crypto account retrieval with API error"""
        crypto_client.api.get_account.side_effect = APIError("Account not found")

        with pytest.raises(TradingError) as exc_info:
            await crypto_client.get_crypto_account()

        assert "Failed to get crypto account" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_crypto_positions_success(self, crypto_client, sample_crypto_positions):
        """Test successful crypto positions retrieval"""
        crypto_client.api.list_positions.return_value = sample_crypto_positions

        positions = await crypto_client.get_crypto_positions()

        assert len(positions) == 2

        btc_position = positions[0]
        assert btc_position["symbol"] == "BTCUSD"
        assert btc_position["quantity"] == 1.5
        assert btc_position["market_value"] == 67500.0
        assert btc_position["avg_cost"] == 45000.0
        assert btc_position["side"] == "long"
        assert btc_position["asset_class"] == "crypto"

        eth_position = positions[1]
        assert eth_position["symbol"] == "ETHUSD"
        assert eth_position["quantity"] == 10.0

    @pytest.mark.asyncio
    async def test_get_crypto_positions_filters_non_crypto(self, crypto_client):
        """Test crypto positions filtering excludes non-crypto assets"""
        mixed_positions = [
            Mock(symbol="BTCUSD", qty="1.0", market_value="45000.00",
                 avg_entry_price="45000.00", current_price="45000.00",
                 unrealized_pl="0.00", unrealized_plpc="0.00", side="long"),
            Mock(symbol="AAPL", qty="100", market_value="15000.00",
                 avg_entry_price="150.00", current_price="150.00",
                 unrealized_pl="0.00", unrealized_plpc="0.00", side="long"),
            Mock(symbol="ETHUSD", qty="5.0", market_value="15000.00",
                 avg_entry_price="3000.00", current_price="3000.00",
                 unrealized_pl="0.00", unrealized_plpc="0.00", side="long")
        ]

        crypto_client.api.list_positions.return_value = mixed_positions

        positions = await crypto_client.get_crypto_positions()

        # Should only return crypto positions (BTCUSD, ETHUSD)
        assert len(positions) == 2
        symbols = [pos["symbol"] for pos in positions]
        assert "BTCUSD" in symbols
        assert "ETHUSD" in symbols
        assert "AAPL" not in symbols

    @pytest.mark.asyncio
    async def test_execute_crypto_order_market_buy_success(self, crypto_client, sample_crypto_order):
        """Test successful crypto market buy order execution"""
        crypto_client.api.submit_order.return_value = sample_crypto_order

        result = await crypto_client.execute_crypto_order(
            symbol="BTCUSD",
            quantity=0.5,
            side="buy",
            order_type="market"
        )

        # Verify API call
        crypto_client.api.submit_order.assert_called_once()
        call_args = crypto_client.api.submit_order.call_args[1]
        assert call_args["symbol"] == "BTCUSD"
        assert call_args["qty"] == "0.5"  # Should be string for fractional support
        assert call_args["side"] == "buy"
        assert call_args["type"] == "market"
        assert call_args["time_in_force"] == "gtc"

        # Verify result
        assert result["id"] == "crypto_order_123"
        assert result["symbol"] == "BTCUSD"
        assert result["quantity"] == 0.5
        assert result["side"] == "buy"
        assert result["order_type"] == "market"
        assert result["status"] == "filled"
        assert result["asset_class"] == "crypto"

    @pytest.mark.asyncio
    async def test_execute_crypto_order_limit_sell_success(self, crypto_client):
        """Test successful crypto limit sell order execution"""
        limit_order = Mock()
        limit_order.id = "limit_order_123"
        limit_order.client_order_id = "client_limit_123"
        limit_order.symbol = "ETHUSD"
        limit_order.qty = "2.5"
        limit_order.side = "sell"
        limit_order.order_type = "limit"
        limit_order.time_in_force = "day"
        limit_order.status = "new"
        limit_order.limit_price = "3100.00"
        limit_order.stop_price = None
        limit_order.filled_qty = "0"
        limit_order.filled_avg_price = "0"
        limit_order.submitted_at = datetime.now(timezone.utc)

        crypto_client.api.submit_order.return_value = limit_order

        result = await crypto_client.execute_crypto_order(
            symbol="ETHUSD",
            quantity=2.5,
            side="sell",
            order_type="limit",
            time_in_force="day",
            limit_price=3100.0,
            client_order_id="client_limit_123"
        )

        # Verify API call includes limit price
        call_args = crypto_client.api.submit_order.call_args[1]
        assert call_args["limit_price"] == "3100.0"
        assert call_args["client_order_id"] == "client_limit_123"

        # Verify result
        assert result["limit_price"] == 3100.0
        assert result["filled_qty"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_crypto_order_invalid_symbol(self, crypto_client):
        """Test crypto order execution with invalid symbol"""
        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="AAPL",  # Not a crypto symbol
                quantity=1.0,
                side="buy"
            )

        assert "Invalid crypto symbol format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_crypto_order_invalid_quantity(self, crypto_client):
        """Test crypto order execution with invalid quantity"""
        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=-1.0,  # Negative quantity
                side="buy"
            )

        assert "Quantity must be positive" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_crypto_order_invalid_side(self, crypto_client):
        """Test crypto order execution with invalid side"""
        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=1.0,
                side="invalid_side"
            )

        assert "Invalid side" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_crypto_order_invalid_order_type(self, crypto_client):
        """Test crypto order execution with invalid order type"""
        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=1.0,
                side="buy",
                order_type="invalid_type"
            )

        assert "Invalid order type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_crypto_order_invalid_time_in_force(self, crypto_client):
        """Test crypto order execution with invalid time in force"""
        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=1.0,
                side="buy",
                time_in_force="invalid_tif"
            )

        assert "Invalid time in force" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_crypto_order_fractional_quantity(self, crypto_client, sample_crypto_order):
        """Test crypto order execution with fractional quantity"""
        sample_crypto_order.qty = "0.001"  # Very small fractional amount
        crypto_client.api.submit_order.return_value = sample_crypto_order

        result = await crypto_client.execute_crypto_order(
            symbol="BTCUSD",
            quantity=0.001,  # Fractional Bitcoin
            side="buy"
        )

        # Verify fractional quantity handling
        call_args = crypto_client.api.submit_order.call_args[1]
        assert call_args["qty"] == "0.001"
        assert result["quantity"] == 0.001

    @pytest.mark.asyncio
    async def test_execute_crypto_order_api_error(self, crypto_client):
        """Test crypto order execution with API error"""
        crypto_client.api.submit_order.side_effect = APIError("Insufficient funds")

        with pytest.raises(TradingError) as exc_info:
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=1.0,
                side="buy"
            )

        assert "Failed to execute crypto order" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cancel_crypto_order_success(self, crypto_client):
        """Test successful crypto order cancellation"""
        crypto_client.api.cancel_order.return_value = None

        result = await crypto_client.cancel_crypto_order("order_123")

        assert result is True
        crypto_client.api.cancel_order.assert_called_once_with("order_123")

    @pytest.mark.asyncio
    async def test_cancel_crypto_order_api_error(self, crypto_client):
        """Test crypto order cancellation with API error"""
        crypto_client.api.cancel_order.side_effect = APIError("Order not found")

        with pytest.raises(TradingError) as exc_info:
            await crypto_client.cancel_crypto_order("invalid_order_id")

        assert "Failed to cancel crypto order" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_crypto_order_success(self, crypto_client, sample_crypto_order):
        """Test successful crypto order retrieval"""
        crypto_client.api.get_order.return_value = sample_crypto_order

        order = await crypto_client.get_crypto_order("crypto_order_123")

        assert order["id"] == "crypto_order_123"
        assert order["symbol"] == "BTCUSD"
        assert order["quantity"] == 0.5
        assert order["status"] == "filled"
        assert order["filled_qty"] == 0.5
        assert order["filled_avg_price"] == 45000.0
        assert order["asset_class"] == "crypto"

    @pytest.mark.asyncio
    async def test_get_crypto_order_non_crypto_symbol(self, crypto_client):
        """Test get crypto order with non-crypto symbol"""
        non_crypto_order = Mock()
        non_crypto_order.symbol = "AAPL"  # Not a crypto symbol

        crypto_client.api.get_order.return_value = non_crypto_order

        with pytest.raises(TradingError) as exc_info:
            await crypto_client.get_crypto_order("order_123")

        assert "not a cryptocurrency order" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_crypto_orders_success(self, crypto_client, sample_crypto_order):
        """Test successful crypto orders list retrieval"""
        orders_list = [sample_crypto_order]
        crypto_client.api.list_orders.return_value = orders_list

        orders = await crypto_client.get_crypto_orders(
            status="filled",
            limit=50
        )

        assert len(orders) == 1
        assert orders[0]["symbol"] == "BTCUSD"

        # Verify API call parameters
        crypto_client.api.list_orders.assert_called_once_with(
            status="filled",
            limit=50,
            after=None,
            until=None
        )

    @pytest.mark.asyncio
    async def test_get_crypto_orders_with_date_filters(self, crypto_client, sample_crypto_order):
        """Test crypto orders retrieval with date filtering"""
        from datetime import datetime, timezone

        after_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        until_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

        crypto_client.api.list_orders.return_value = [sample_crypto_order]

        orders = await crypto_client.get_crypto_orders(
            status="all",
            limit=100,
            after=after_date,
            until=until_date
        )

        # Verify date filtering parameters
        call_args = crypto_client.api.list_orders.call_args[1]
        assert call_args["after"] == after_date.isoformat()
        assert call_args["until"] == until_date.isoformat()

    def test_is_crypto_symbol_valid_symbols(self, crypto_client):
        """Test crypto symbol validation with valid symbols"""
        valid_symbols = [
            "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD",
            "LTCUSD", "XRPUSD", "SOLUSD", "MATICUSD", "AVAXUSD"
        ]

        for symbol in valid_symbols:
            assert crypto_client._is_crypto_symbol(symbol) is True

    def test_is_crypto_symbol_invalid_symbols(self, crypto_client):
        """Test crypto symbol validation with invalid symbols"""
        invalid_symbols = [
            "AAPL", "GOOGL", "TSLA", "SPY", "QQQ",
            "BTCEUR", "ETHBTC", "BTC", "ETH", ""
        ]

        for symbol in invalid_symbols:
            assert crypto_client._is_crypto_symbol(symbol) is False

    @pytest.mark.asyncio
    async def test_24_7_trading_support(self, crypto_client, sample_crypto_order):
        """Test 24/7 trading support for cryptocurrency"""
        # Test order execution outside normal market hours
        weekend_order = sample_crypto_order
        weekend_order.submitted_at = datetime(2024, 1, 6, 15, 30, tzinfo=timezone.utc)  # Saturday

        crypto_client.api.submit_order.return_value = weekend_order

        result = await crypto_client.execute_crypto_order(
            symbol="BTCUSD",
            quantity=0.1,
            side="buy"
        )

        # Should execute successfully regardless of time
        assert result["id"] == "crypto_order_123"
        assert result["symbol"] == "BTCUSD"

    @pytest.mark.asyncio
    async def test_paper_trading_enforcement(self, crypto_client, sample_crypto_order):
        """Test that paper trading is enforced for safety"""
        # Even if initialized with paper=False, should force paper trading
        crypto_client.paper = False
        crypto_client.api.submit_order.return_value = sample_crypto_order

        with patch('app.trading.crypto_client.logger') as mock_logger:
            result = await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=0.1,
                side="buy"
            )

            # Should log warning about forcing paper trading
            mock_logger.warning.assert_called_with("Forcing paper trading mode for crypto orders")

    @pytest.mark.asyncio
    async def test_stop_limit_order_execution(self, crypto_client):
        """Test stop-limit order execution with both prices"""
        stop_limit_order = Mock()
        stop_limit_order.id = "stop_limit_123"
        stop_limit_order.symbol = "BTCUSD"
        stop_limit_order.qty = "1.0"
        stop_limit_order.side = "sell"
        stop_limit_order.order_type = "stop_limit"
        stop_limit_order.limit_price = "44000.00"
        stop_limit_order.stop_price = "44500.00"
        stop_limit_order.status = "new"
        stop_limit_order.filled_qty = "0"
        stop_limit_order.filled_avg_price = "0"
        stop_limit_order.time_in_force = "gtc"
        stop_limit_order.client_order_id = None
        stop_limit_order.submitted_at = datetime.now(timezone.utc)

        crypto_client.api.submit_order.return_value = stop_limit_order

        result = await crypto_client.execute_crypto_order(
            symbol="BTCUSD",
            quantity=1.0,
            side="sell",
            order_type="stop_limit",
            limit_price=44000.0,
            stop_price=44500.0
        )

        # Verify both prices are included
        call_args = crypto_client.api.submit_order.call_args[1]
        assert call_args["limit_price"] == "44000.0"
        assert call_args["stop_price"] == "44500.0"

        assert result["limit_price"] == 44000.0
        assert result["stop_price"] == 44500.0

    @pytest.mark.asyncio
    async def test_high_precision_fractional_trading(self, crypto_client, sample_crypto_order):
        """Test high precision fractional cryptocurrency trading"""
        # Test very small fractional amounts
        tiny_amount = 0.00000001  # 1 satoshi equivalent

        sample_crypto_order.qty = str(tiny_amount)
        crypto_client.api.submit_order.return_value = sample_crypto_order

        result = await crypto_client.execute_crypto_order(
            symbol="BTCUSD",
            quantity=tiny_amount,
            side="buy"
        )

        # Verify high precision handling
        call_args = crypto_client.api.submit_order.call_args[1]
        assert call_args["qty"] == str(tiny_amount)
        assert result["quantity"] == tiny_amount

    @pytest.mark.asyncio
    async def test_error_handling_network_issues(self, crypto_client):
        """Test error handling for network connectivity issues"""
        import requests

        # Simulate network error
        crypto_client.api.submit_order.side_effect = requests.ConnectionError("Network unreachable")

        with pytest.raises(TradingError):
            await crypto_client.execute_crypto_order(
                symbol="BTCUSD",
                quantity=1.0,
                side="buy"
            )