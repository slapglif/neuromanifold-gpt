"""Search space configuration for hyperparameter optimization.

This module defines the search space for HPO, supporting various parameter types:
- Float parameters (linear or log scale)
- Integer parameters
- Categorical parameters (including boolean)

The search space is configured via YAML files and used with Optuna trials.
"""

from typing import Any, Dict, List, Optional, Union
import optuna


class SearchSpace:
    """Defines the hyperparameter search space for optimization.

    The search space configuration supports:
    - Float parameters: continuous values with optional log scale
    - Integer parameters: discrete integer values
    - Categorical parameters: discrete choices from a set

    Example YAML configuration:
        search_space:
          learning_rate:
            type: float
            low: 1e-5
            high: 1e-2
            log: true
          n_layer:
            type: int
            low: 2
            high: 8
          use_sdr:
            type: categorical
            choices: [true, false]

        fixed_params:
          dataset: shakespeare_char
          max_iters: 1000

    Attributes:
        search_params: Dictionary of parameters to optimize
        fixed_params: Dictionary of fixed parameters (not optimized)
        config: Full configuration dictionary
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize search space from configuration dictionary.

        Args:
            config: Configuration dictionary with 'search_space' and optional
                   'fixed_params' keys

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.search_params = config.get("search_space", {})
        self.fixed_params = config.get("fixed_params", {})

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the search space configuration.

        Raises:
            ValueError: If any parameter configuration is invalid
        """
        if not self.search_params:
            raise ValueError("search_space must contain at least one parameter")

        for param_name, param_config in self.search_params.items():
            if not isinstance(param_config, dict):
                raise ValueError(
                    f"Parameter '{param_name}' must be a dictionary"
                )

            param_type = param_config.get("type")
            if param_type not in ["float", "int", "categorical"]:
                raise ValueError(
                    f"Parameter '{param_name}' has invalid type '{param_type}'. "
                    f"Must be one of: float, int, categorical"
                )

            # Validate type-specific fields
            if param_type in ["float", "int"]:
                if "low" not in param_config or "high" not in param_config:
                    raise ValueError(
                        f"Parameter '{param_name}' of type '{param_type}' must "
                        f"have 'low' and 'high' fields"
                    )
                if param_config["low"] >= param_config["high"]:
                    raise ValueError(
                        f"Parameter '{param_name}': 'low' must be less than 'high'"
                    )
            elif param_type == "categorical":
                if "choices" not in param_config:
                    raise ValueError(
                        f"Parameter '{param_name}' of type 'categorical' must "
                        f"have 'choices' field"
                    )
                if not isinstance(param_config["choices"], list):
                    raise ValueError(
                        f"Parameter '{param_name}': 'choices' must be a list"
                    )
                if len(param_config["choices"]) < 2:
                    raise ValueError(
                        f"Parameter '{param_name}': 'choices' must have at least "
                        f"2 options"
                    )

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Generate hyperparameter suggestions for an Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameter values
        """
        params = {}

        for param_name, param_config in self.search_params.items():
            param_type = param_config["type"]

            if param_type == "float":
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=log_scale
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )

        # Add fixed parameters
        params.update(self.fixed_params)

        return params

    def get_param_names(self) -> List[str]:
        """Get list of all parameter names in the search space.

        Returns:
            List of parameter names
        """
        return list(self.search_params.keys())

    def get_fixed_param_names(self) -> List[str]:
        """Get list of fixed parameter names.

        Returns:
            List of fixed parameter names
        """
        return list(self.fixed_params.keys())

    def is_log_scale(self, param_name: str) -> bool:
        """Check if a float parameter uses log scale.

        Args:
            param_name: Name of the parameter

        Returns:
            True if parameter uses log scale, False otherwise
        """
        if param_name not in self.search_params:
            return False

        param_config = self.search_params[param_name]
        if param_config.get("type") != "float":
            return False

        return param_config.get("log", False)

    def __repr__(self) -> str:
        """String representation of the search space."""
        lines = ["SearchSpace:"]
        lines.append(f"  Search parameters: {len(self.search_params)}")
        for param_name, param_config in self.search_params.items():
            param_type = param_config["type"]
            if param_type in ["float", "int"]:
                log_info = " (log scale)" if param_config.get("log", False) else ""
                lines.append(
                    f"    {param_name}: [{param_config['low']}, "
                    f"{param_config['high']}]{log_info}"
                )
            else:
                lines.append(
                    f"    {param_name}: {param_config['choices']}"
                )

        if self.fixed_params:
            lines.append(f"  Fixed parameters: {len(self.fixed_params)}")
            for param_name, value in self.fixed_params.items():
                lines.append(f"    {param_name}: {value}")

        return "\n".join(lines)
