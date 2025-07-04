# backtest/data/debug/test_circuit_breaker_cli.py
"""
Module: Circuit Breaker CLI Tester
Purpose: Interactive testing of circuit breaker functionality
Features: Simulate various failure scenarios, test data availability
"""

import asyncio
import click
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
import time

from backtest.data.circuit_breaker import (
    CircuitBreaker, CircuitBreakerError, RateLimitError, NoDataAvailableError
)
from backtest.data.protected_data_manager import ProtectedDataManager
from backtest.data.polygon_data_manager import PolygonDataManager

console = Console()
logger = logging.getLogger(__name__)


class CircuitBreakerTester:
    """Interactive circuit breaker tester"""
    
    def __init__(self, use_real_api: bool = False):
        self.use_real_api = use_real_api
        self.circuit_breaker = None
        self.protected_manager = None
        self.request_count = 0
        self.failure_count = 0
        
        if use_real_api:
            self._setup_real_api()
        else:
            self._setup_mock_api()
    
    def _setup_real_api(self):
        """Setup with real Polygon API"""
        try:
            from config.polygon import POLYGON_API_KEY
            polygon_manager = PolygonDataManager(
                api_key=POLYGON_API_KEY,
                use_cache=True
            )
            self.protected_manager = ProtectedDataManager(polygon_manager)
            self.circuit_breaker = self.protected_manager.circuit_breaker
            console.print("[green]Connected to real Polygon API[/green]")
        except ImportError:
            console.print("[red]Polygon API key not found, using mock[/red]")
            self._setup_mock_api()
    
    def _setup_mock_api(self):
        """Setup with mock API"""
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=0.5,
            consecutive_failures=3,
            recovery_timeout=10,
            rate_limits={
                'bars': {'per_minute': 60, 'burst': 5},
                'trades': {'per_minute': 30, 'burst': 3},
                'quotes': {'per_minute': 30, 'burst': 3}
            }
        )
        console.print("[yellow]Using mock API for testing[/yellow]")
    
    async def simulate_success(self):
        """Simulate successful API call"""
        async def success_func():
            await asyncio.sleep(0.1)  # Simulate API latency
            return pd.DataFrame({'value': [1, 2, 3]})
        
        try:
            self.request_count += 1
            result = await self.circuit_breaker.call(success_func)
            console.print(f"[green]✓ Success #{self.request_count}[/green]")
            return result
        except Exception as e:
            self.failure_count += 1
            console.print(f"[red]✗ Failed: {e}[/red]")
            raise
    
    async def simulate_failure(self):
        """Simulate API failure"""
        async def failing_func():
            await asyncio.sleep(0.05)
            raise Exception("Simulated API error")
        
        try:
            self.request_count += 1
            await self.circuit_breaker.call(failing_func)
        except CircuitBreakerError as e:
            console.print(f"[red]⚡ Circuit Breaker: {e}[/red]")
            raise
        except Exception as e:
            self.failure_count += 1
            console.print(f"[red]✗ API Error #{self.failure_count}[/red]")
            raise
    
    async def simulate_no_data(self, symbol: str = 'TEST'):
        """Simulate no data scenario"""
        async def no_data_func(symbol, start_time, end_time, **kwargs):
            return pd.DataFrame()  # Empty DataFrame
        
        try:
            self.request_count += 1
            result = await self.circuit_breaker.call(
                no_data_func,
                symbol=symbol,
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                end_time=datetime.now(timezone.utc),
                operation_type='quotes'
            )
        except NoDataAvailableError as e:
            console.print(f"[yellow]📭 No Data: {e}[/yellow]")
            raise
    
    async def simulate_rate_limit(self, operation_type: str = 'bars'):
        """Simulate hitting rate limit"""
        async def data_func():
            return pd.DataFrame({'value': [1]})
        
        burst_limit = self.circuit_breaker.rate_limiters[operation_type].burst_size
        
        # Use up burst capacity
        for i in range(burst_limit):
            try:
                await self.circuit_breaker.call(data_func, operation_type=operation_type)
                console.print(f"[green]Request {i+1}/{burst_limit}[/green]")
            except Exception as e:
                console.print(f"[red]Failed at {i+1}: {e}[/red]")
                return
        
        # Next request should hit rate limit
        try:
            await self.circuit_breaker.call(data_func, operation_type=operation_type)
        except RateLimitError as e:
            console.print(f"[yellow]🚦 Rate Limited: {e}[/yellow]")
            raise
    
    def get_status_table(self) -> Table:
        """Create status table"""
        status = self.circuit_breaker.get_status()
        
        table = Table(title="Circuit Breaker Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        # State with color coding
        state_color = {
            'CLOSED': 'green',
            'OPEN': 'red',
            'HALF_OPEN': 'yellow'
        }[status['state']]
        table.add_row("State", f"[{state_color}]{status['state']}[/{state_color}]")
        
        table.add_row("Failure Rate", f"{status['failure_rate']:.1%}")
        table.add_row("Consecutive Failures", str(status['consecutive_failures']))
        table.add_row("Recovery Attempts", str(status['recovery_attempts']))
        
        if status['state'] == 'OPEN':
            table.add_row("Time Until Recovery", f"{status['time_until_recovery']:.1f}s")
        
        # Metrics
        metrics = status['metrics']
        table.add_row("Total Requests", str(metrics['total_requests']))
        table.add_row("Successful", str(metrics['successful_requests']))
        table.add_row("Failed", str(metrics['failed_requests']))
        table.add_row("Circuit Opens", str(metrics['circuit_opens']))
        table.add_row("Rate Limit Hits", str(metrics['rate_limit_hits']))
        table.add_row("No Data Responses", str(metrics['no_data_responses']))
        
        return table
    
    def get_rate_limit_table(self) -> Table:
        """Create rate limit status table"""
        status = self.circuit_breaker.get_status()
        
        table = Table(title="Rate Limits")
        table.add_column("Operation", style="cyan")
        table.add_column("Available", style="green")
        table.add_column("Rate/min", style="yellow")
        
        for op_type, limits in status['rate_limits'].items():
            table.add_row(
                op_type,
                f"{limits['tokens_available']:.1f}",
                str(limits['rate_per_minute'])
            )
        
        return table
    
    async def run_scenario(self, scenario: str):
        """Run a specific test scenario"""
        console.print(f"\n[bold blue]Running scenario: {scenario}[/bold blue]")
        
        if scenario == "open_circuit":
            # Cause circuit to open
            for i in range(5):
                try:
                    await self.simulate_failure()
                except Exception:
                    pass
                await asyncio.sleep(0.1)
            
            console.print(self.get_status_table())
            
        elif scenario == "recovery":
            # Open circuit first
            for _ in range(3):
                try:
                    await self.simulate_failure()
                except Exception:
                    pass
            
            console.print("[yellow]Circuit opened, waiting for recovery...[/yellow]")
            
            # Wait and show countdown
            recovery_time = self.circuit_breaker._time_until_recovery()
            with console.status(f"Waiting {recovery_time:.1f}s...") as status:
                await asyncio.sleep(recovery_time + 0.5)
            
            # Try successful call
            try:
                await self.simulate_success()
                console.print("[green]✓ Circuit recovered![/green]")
            except Exception as e:
                console.print(f"[red]Recovery failed: {e}[/red]")
            
        elif scenario == "rate_limit":
            await self.simulate_rate_limit('bars')
            console.print(self.get_rate_limit_table())
            
        elif scenario == "no_data":
            # Test missing data scenario
            try:
                await self.simulate_no_data('AAPL')
            except NoDataAvailableError:
                pass
            
            # Try again - should use cache
            try:
                await self.simulate_no_data('AAPL')
                console.print("[yellow]Second attempt used no-data cache[/yellow]")
            except NoDataAvailableError:
                pass
            
        elif scenario == "real_quotes":
            if not self.use_real_api:
                console.print("[red]This scenario requires real API[/red]")
                return
            
            # Test with very recent time that might not have quotes
            end_time = datetime.now(timezone.utc) - timedelta(minutes=2)
            start_time = end_time - timedelta(minutes=5)
            
            try:
                result = await self.protected_manager.load_quotes(
                    symbol='AAPL',
                    start_time=start_time,
                    end_time=end_time
                )
                console.print(f"[green]Found {len(result)} quotes[/green]")
            except NoDataAvailableError as e:
                console.print(f"[yellow]No quotes available: {e}[/yellow]")
    
    async def monitor_loop(self, duration: int = 30):
        """Live monitoring of circuit breaker"""
        console.print(f"[bold]Monitoring for {duration} seconds...[/bold]")
        
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=15),
            Layout(name="rates", size=8),
            Layout(name="log", size=10)
        )
        
        logs = []
        
        async def make_random_requests():
            """Make random requests during monitoring"""
            while True:
                await asyncio.sleep(0.5)
                
                # Random action
                import random
                action = random.choice(['success', 'success', 'failure', 'rate'])
                
                try:
                    if action == 'success':
                        await self.simulate_success()
                        logs.append(f"[green]✓ Success at {datetime.now().strftime('%H:%M:%S')}[/green]")
                    elif action == 'failure':
                        await self.simulate_failure()
                    elif action == 'rate':
                        await self.simulate_rate_limit('bars')
                except Exception as e:
                    logs.append(f"[red]✗ {type(e).__name__} at {datetime.now().strftime('%H:%M:%S')}[/red]")
                
                # Keep last 5 logs
                if len(logs) > 5:
                    logs.pop(0)
        
        # Start background requests
        task = asyncio.create_task(make_random_requests())
        
        try:
            with Live(layout, refresh_per_second=2) as live:
                start_time = time.time()
                
                while time.time() - start_time < duration:
                    layout["status"].update(Panel(self.get_status_table()))
                    layout["rates"].update(Panel(self.get_rate_limit_table()))
                    layout["log"].update(Panel("\n".join(logs), title="Recent Activity"))
                    
                    await asyncio.sleep(0.5)
        finally:
            task.cancel()


@click.command()
@click.option('--scenario', '-s', 
              type=click.Choice(['open_circuit', 'recovery', 'rate_limit', 
                               'no_data', 'real_quotes', 'monitor']),
              help='Test scenario to run')
@click.option('--real-api', is_flag=True, help='Use real Polygon API')
@click.option('--duration', '-d', default=30, help='Duration for monitoring (seconds)')
def main(scenario, real_api, duration):
    """Circuit Breaker interactive tester"""
    tester = CircuitBreakerTester(use_real_api=real_api)
    
    if scenario == 'monitor':
        asyncio.run(tester.monitor_loop(duration))
    elif scenario:
        asyncio.run(tester.run_scenario(scenario))
    else:
        # Interactive mode
        console.print("[bold]Circuit Breaker Test Scenarios:[/bold]")
        console.print("1. open_circuit - Force circuit to open")
        console.print("2. recovery - Test recovery mechanism")
        console.print("3. rate_limit - Hit rate limits")
        console.print("4. no_data - Test missing data handling")
        console.print("5. real_quotes - Test real missing quotes (requires --real-api)")
        console.print("6. monitor - Live monitoring dashboard")
        
        console.print("\n[bold]Current Status:[/bold]")
        console.print(tester.get_status_table())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()