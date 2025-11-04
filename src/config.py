import yaml
import datetime as dt
from pathlib import Path
from typing import Dict, List, Any
from signals import get_signal
from filters import get_filter
from models import Signal, Filter


class Config:
    """Configuration parser for backtest YAML files."""
    
    def __init__(self, config_path: str = "config.yml"):
        """
        Initialize the configuration parser.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @property
    def name(self) -> str:
        """Get the backtest name."""
        return self._config.get('name')
    
    @property
    def start(self) -> dt.date:
        """Get the backtest start date."""
        return self._config.get('start')
    
    @property
    def end(self) -> dt.date:
        """Get the backtest end date."""
        return self._config.get('end')
    
    @property
    def rebalance_frequency(self) -> str:
        """Get the rebalance frequency."""
        return self._config.get('rebalance-frequency')
    
    @property
    def dataset(self) -> str:
        """Get the core dataset."""
        return self._config.get('dataset')
    
    @property
    def signal(self) -> Signal:
        """Get the signal type."""
        signal_name = self._config.get('signal')
        return get_signal(signal_name)
    
    @property
    def portfolio_constructor_type(self) -> str:
        """Get the portfolio constructor type."""
        pc: dict = self._config.get('portfolio-constructor', {})
        return pc.get('type')
    
    @property
    def n_bins(self) -> int:
        """Get the number of quantile bins."""
        pc: dict = self._config.get('portfolio-constructor', {})
        return pc.get('n-bins')
    
    @property
    def weighting_scheme(self) -> str:
        """Get the portfolio weighting scheme."""
        pc: dict = self._config.get('portfolio-constructor', {})
        return pc.get('weighting-scheme')
    
    @property
    def filters(self) -> List[Filter]:
        """Get the list of filters to apply."""
        filter_names = self._config.get('filters', [])
        return [get_filter(filter_name) for filter_name in filter_names]
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"BacktestConfig(name='{self.name}', start={self.start}, end={self.end})"