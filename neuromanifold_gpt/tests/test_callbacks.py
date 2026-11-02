"""Tests for training callback modules.

Tests for the focused single-purpose callbacks that replaced the monolithic
TrainingHealthCallback. Each callback is tested independently with mocked
PyTorch Lightning components.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from collections import deque


class TestGradientMonitorCallback:
    """Test suite for GradientMonitorCallback."""

    def test_callback_initialization(self):
        """Test GradientMonitorCallback initializes with correct defaults."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback()
        assert callback.log_interval == 100
        assert callback.gradient_norm_history_size == 100
        assert callback.grad_explosion_threshold == 3.0
        assert callback.min_grad_history_for_detection == 20
        assert len(callback.grad_norm_history) == 0
        assert callback.total_clip_events == 0
        assert callback.total_steps == 0

    def test_callback_custom_initialization(self):
        """Test GradientMonitorCallback initializes with custom parameters."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(
            log_interval=50,
            gradient_norm_history_size=200,
            grad_explosion_threshold=2.5,
            min_grad_history_for_detection=10
        )
        assert callback.log_interval == 50
        assert callback.gradient_norm_history_size == 200
        assert callback.grad_explosion_threshold == 2.5
        assert callback.min_grad_history_for_detection == 10

    def test_gradient_norm_tracking(self):
        """Test that gradient norms are tracked correctly."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback()

        # Mock trainer and model
        trainer = Mock()
        pl_module = Mock()

        # Create mock parameters with gradients
        param1 = Mock()
        param1.grad = Mock()
        param1.grad.data.norm.return_value = torch.tensor(2.0)

        param2 = Mock()
        param2.grad = Mock()
        param2.grad.data.norm.return_value = torch.tensor(3.0)

        pl_module.parameters.return_value = [param1, param2]

        # Call on_after_backward
        callback.on_after_backward(trainer, pl_module)

        # Check that gradient norm was calculated and stored
        # Expected: sqrt(2^2 + 3^2) = sqrt(13) ≈ 3.606
        assert len(callback.grad_norm_history) == 1
        assert callback.grad_norm_before_clip is not None
        assert callback.grad_norm_before_clip > 0

    def test_gradient_clipping_detection(self):
        """Test that gradient clipping events are detected."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback()
        callback.grad_norm_before_clip = 10.0  # Set high pre-clip norm

        # Mock trainer, model, and optimizer
        trainer = Mock()
        optimizer = Mock()
        pl_module = Mock()

        # Mock config with gradient clipping enabled
        pl_module.config = Mock()
        pl_module.config.grad_clip = 5.0

        # Create mock parameters with lower post-clip gradients
        param1 = Mock()
        param1.grad = Mock()
        param1.grad.data.norm.return_value = torch.tensor(1.5)

        param2 = Mock()
        param2.grad = Mock()
        param2.grad.data.norm.return_value = torch.tensor(2.0)

        pl_module.parameters.return_value = [param1, param2]

        # Call on_before_optimizer_step
        callback.on_before_optimizer_step(trainer, pl_module, optimizer)

        # Check that clipping was detected
        assert callback.total_clip_events == 1
        assert callback.total_steps == 1
        assert len(callback.clip_ratios) > 0

    def test_gradient_explosion_detection(self):
        """Test that gradient explosions are detected."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(min_grad_history_for_detection=5)

        # Add normal gradient norms
        for _ in range(20):
            callback.grad_norm_history.append(1.0)

        # Test explosion detection with high gradient norm
        with patch('builtins.print') as mock_print:
            callback._detect_gradient_explosion(10.0, current_step=100)
            # Should print warning for explosion
            assert mock_print.called
            assert 'GRADIENT EXPLOSION' in str(mock_print.call_args)

    def test_no_explosion_with_insufficient_history(self):
        """Test that explosion detection requires minimum history."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback(min_grad_history_for_detection=20)

        # Add insufficient history
        for _ in range(10):
            callback.grad_norm_history.append(1.0)

        # Should not detect explosion with insufficient history
        with patch('builtins.print') as mock_print:
            callback._detect_gradient_explosion(100.0, current_step=10)
            # Should not print warning
            assert not mock_print.called

    def test_gradient_metrics_logging(self):
        """Test that gradient metrics are logged correctly."""
        from neuromanifold_gpt.callbacks.gradient_monitor import GradientMonitorCallback

        callback = GradientMonitorCallback()
        callback.grad_norm_history.append(2.5)
        callback.grad_norm_history.append(3.0)
        callback.total_steps = 2
        callback.total_clip_events = 1
        callback.clip_ratios.append(0.8)

        # Mock trainer and model
        trainer = Mock()
        pl_module = Mock()

        # Call on_train_batch_end
        callback.on_train_batch_end(trainer, pl_module, None, None, 1)

        # Check that metrics were logged
        assert pl_module.log.called
        # Verify specific metrics were logged
        log_calls = {call[0][0]: call[0][1] for call in pl_module.log.call_args_list}
        assert 'train/grad_norm' in log_calls
        assert 'train/grad_norm_avg' in log_calls
        assert 'train/clip_rate' in log_calls


class TestMemoryMonitorCallback:
    """Test suite for MemoryMonitorCallback."""

    def test_callback_initialization(self):
        """Test MemoryMonitorCallback initializes with correct defaults."""
        from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

        callback = MemoryMonitorCallback()
        assert callback.log_interval == 100
        assert callback.peak_memory_mb == 0.0
        assert callback.current_step == 0

    def test_callback_custom_initialization(self):
        """Test MemoryMonitorCallback initializes with custom parameters."""
        from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

        callback = MemoryMonitorCallback(log_interval=50)
        assert callback.log_interval == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_tracking_with_cuda(self):
        """Test that memory is tracked when CUDA is available."""
        from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

        callback = MemoryMonitorCallback()

        # Mock trainer and model
        trainer = Mock()
        pl_module = Mock()

        # Call on_train_batch_end
        callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Check that memory metrics were logged
        assert pl_module.log.called
        log_calls = {call[0][0]: call[0][1] for call in pl_module.log.call_args_list}
        assert 'train/memory_current_mb' in log_calls
        assert 'train/memory_peak_mb' in log_calls

    def test_memory_tracking_without_cuda(self):
        """Test that callback handles absence of CUDA gracefully."""
        from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

        callback = MemoryMonitorCallback()

        # Mock trainer and model
        trainer = Mock()
        pl_module = Mock()

        # Mock cuda.is_available to return False
        with patch('torch.cuda.is_available', return_value=False):
            callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Current step should still be updated
        assert callback.current_step == 0

    def test_peak_memory_update(self):
        """Test that peak memory is updated correctly."""
        from neuromanifold_gpt.callbacks.memory_monitor import MemoryMonitorCallback

        callback = MemoryMonitorCallback()
        callback.peak_memory_mb = 100.0

        # Mock higher memory usage
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.memory_allocated', return_value=150_000_000):  # 150 MB
                with patch('torch.cuda.max_memory_allocated', return_value=200_000_000):  # 200 MB
                    trainer = Mock()
                    pl_module = Mock()
                    callback.on_train_batch_end(trainer, pl_module, None, None, 0)

        # Peak should be updated
        assert callback.peak_memory_mb == 200.0


class TestThroughputMonitorCallback:
    """Test suite for ThroughputMonitorCallback."""

    def test_callback_initialization(self):
        """Test ThroughputMonitorCallback initializes with correct defaults."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

        callback = ThroughputMonitorCallback()
        assert callback.log_interval == 100
        assert callback.step_time_history_size == 20
        assert callback.warmup_steps == 10
        assert len(callback.step_times) == 0
        assert callback.step_start_time is None
        assert callback.training_start_time is None

    def test_callback_custom_initialization(self):
        """Test ThroughputMonitorCallback initializes with custom parameters."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

        callback = ThroughputMonitorCallback(
            log_interval=50,
            step_time_history_size=30,
            warmup_steps=5
        )
        assert callback.log_interval == 50
        assert callback.step_time_history_size == 30
        assert callback.warmup_steps == 5

    def test_eta_formatting(self):
        """Test ETA formatting for different time durations."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

        callback = ThroughputMonitorCallback()

        # Test seconds
        assert callback._format_eta(45) == "45s"

        # Test minutes
        assert callback._format_eta(125) == "2m 5s"

        # Test hours
        assert callback._format_eta(7260) == "2h 1m"

    def test_training_start_initialization(self):
        """Test that training start time and max steps are initialized."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

        callback = ThroughputMonitorCallback()

        # Mock trainer with max_steps
        trainer = Mock()
        trainer.max_steps = 1000
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.training_start_time is not None
        assert callback.max_steps == 1000

    def test_step_time_tracking(self):
        """Test that step times are tracked correctly."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback
        import time

        callback = ThroughputMonitorCallback()

        trainer = Mock()
        pl_module = Mock()
        batch = [torch.randn(4, 256)]  # batch_size=4, seq_len=256

        # Start batch
        callback.on_train_batch_start(trainer, pl_module, batch, 0)
        assert callback.step_start_time is not None

        # Simulate some processing time
        time.sleep(0.01)

        # End batch
        callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # Check that step time was recorded
        assert len(callback.step_times) == 1
        assert callback.step_times[0] > 0

    def test_throughput_calculation(self):
        """Test that throughput is calculated and logged."""
        from neuromanifold_gpt.callbacks.throughput_monitor import ThroughputMonitorCallback

        callback = ThroughputMonitorCallback()
        callback.step_start_time = 0.0

        trainer = Mock()
        pl_module = Mock()
        batch = [torch.randn(4, 256)]  # batch_size=4, seq_len=256

        with patch('time.perf_counter', return_value=0.1):  # 100ms step time
            callback.on_train_batch_end(trainer, pl_module, None, batch, 0)

        # Check that throughput was logged
        assert pl_module.log.called
        log_calls = {call[0][0]: call[0][1] for call in pl_module.log.call_args_list}
        assert 'train/throughput_tokens_per_sec' in log_calls
        assert 'train/step_time_ms' in log_calls


class TestLossMonitorCallback:
    """Test suite for LossMonitorCallback."""

    def test_callback_initialization(self):
        """Test LossMonitorCallback initializes with correct defaults."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback()
        assert callback.log_interval == 100
        assert callback.loss_history_size == 100
        assert callback.loss_spike_threshold == 3.0
        assert callback.min_loss_history_for_detection == 20
        assert len(callback.loss_history) == 0

    def test_callback_custom_initialization(self):
        """Test LossMonitorCallback initializes with custom parameters."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback(
            log_interval=50,
            loss_history_size=200,
            loss_spike_threshold=2.5,
            min_loss_history_for_detection=10
        )
        assert callback.log_interval == 50
        assert callback.loss_history_size == 200
        assert callback.loss_spike_threshold == 2.5
        assert callback.min_loss_history_for_detection == 10

    def test_loss_tracking_from_dict(self):
        """Test that loss is tracked from dict outputs."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback()

        trainer = Mock()
        pl_module = Mock()
        outputs = {'loss': torch.tensor(2.5)}

        callback.on_train_batch_end(trainer, pl_module, outputs, None, 0)

        assert len(callback.loss_history) == 1
        assert callback.loss_history[0] == 2.5

    def test_loss_tracking_from_tensor(self):
        """Test that loss is tracked from tensor outputs."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback()

        trainer = Mock()
        pl_module = Mock()
        outputs = torch.tensor(3.2)

        callback.on_train_batch_end(trainer, pl_module, outputs, None, 0)

        assert len(callback.loss_history) == 1
        assert abs(callback.loss_history[0] - 3.2) < 0.01  # Use approximate comparison for float

    def test_loss_spike_detection(self):
        """Test that loss spikes are detected."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback(min_loss_history_for_detection=5)

        # Add normal loss values
        for _ in range(20):
            callback.loss_history.append(1.0)

        # Test spike detection with high loss
        with patch('builtins.print') as mock_print:
            callback._detect_loss_spike(10.0, current_step=100)
            # Should print warning for spike
            assert mock_print.called
            assert 'LOSS SPIKE' in str(mock_print.call_args)

    def test_no_spike_with_insufficient_history(self):
        """Test that spike detection requires minimum history."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback(min_loss_history_for_detection=20)

        # Add insufficient history
        for _ in range(10):
            callback.loss_history.append(1.0)

        # Should not detect spike with insufficient history
        with patch('builtins.print') as mock_print:
            callback._detect_loss_spike(100.0, current_step=10)
            # Should not print warning
            assert not mock_print.called

    def test_loss_average_logging(self):
        """Test that average loss is logged."""
        from neuromanifold_gpt.callbacks.loss_monitor import LossMonitorCallback

        callback = LossMonitorCallback()
        callback.loss_history.append(2.0)
        callback.loss_history.append(3.0)

        trainer = Mock()
        pl_module = Mock()
        outputs = {'loss': torch.tensor(2.5)}

        callback.on_train_batch_end(trainer, pl_module, outputs, None, 0)

        # Check that average loss was logged
        assert pl_module.log.called
        log_calls = {call[0][0]: call[0][1] for call in pl_module.log.call_args_list}
        assert 'train/loss_avg' in log_calls


class TestAnomalyDetectorCallback:
    """Test suite for AnomalyDetectorCallback."""

    def test_callback_initialization(self):
        """Test AnomalyDetectorCallback initializes correctly."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()
        assert len(callback.warnings) == 0
        assert callback.current_step == 0

    def test_nan_loss_detection(self):
        """Test that NaN in loss is detected."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()

        with patch('builtins.print') as mock_print:
            callback._detect_nan_inf_loss(float('nan'), current_step=10)

            # Should print warning and record it
            assert mock_print.called
            # Check all print calls for the warning message
            all_print_calls = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'NaN DETECTED IN LOSS' in all_print_calls
            assert len(callback.warnings) == 1
            assert callback.warnings[0]['type'] == 'nan_loss'
            assert callback.warnings[0]['severity'] == 'critical'

    def test_inf_loss_detection(self):
        """Test that Inf in loss is detected."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()

        with patch('builtins.print') as mock_print:
            callback._detect_nan_inf_loss(float('inf'), current_step=10)

            # Should print warning and record it
            assert mock_print.called
            # Check all print calls for the warning message
            all_print_calls = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'Inf DETECTED IN LOSS' in all_print_calls
            assert len(callback.warnings) == 1
            assert callback.warnings[0]['type'] == 'inf_loss'

    def test_nan_gradient_detection(self):
        """Test that NaN in gradients is detected."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()

        # Mock module with NaN gradients
        pl_module = Mock()
        param = Mock()
        param.grad = torch.tensor([1.0, float('nan'), 2.0])
        pl_module.named_parameters.return_value = [('layer.weight', param)]

        with patch('builtins.print') as mock_print:
            callback._detect_nan_inf_gradients(pl_module, current_step=5)

            # Should print warning
            assert mock_print.called
            # Check all print calls for the warning message
            all_print_calls = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'NaN DETECTED IN GRADIENTS' in all_print_calls
            assert len(callback.warnings) == 1
            assert callback.warnings[0]['type'] == 'nan_gradients'

    def test_inf_gradient_detection(self):
        """Test that Inf in gradients is detected."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()

        # Mock module with Inf gradients
        pl_module = Mock()
        param = Mock()
        param.grad = torch.tensor([1.0, float('inf'), 2.0])
        pl_module.named_parameters.return_value = [('layer.bias', param)]

        with patch('builtins.print') as mock_print:
            callback._detect_nan_inf_gradients(pl_module, current_step=5)

            # Should print warning
            assert mock_print.called
            # Check all print calls for the warning message
            all_print_calls = ' '.join(str(call) for call in mock_print.call_args_list)
            assert 'Inf DETECTED IN GRADIENTS' in all_print_calls
            assert len(callback.warnings) == 1
            assert callback.warnings[0]['type'] == 'inf_gradients'

    def test_no_anomaly_detection(self):
        """Test that normal values don't trigger anomalies."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()

        # Test normal loss
        with patch('builtins.print') as mock_print:
            callback._detect_nan_inf_loss(2.5, current_step=10)
            assert not mock_print.called
            assert len(callback.warnings) == 0

    def test_loss_anomaly_detection_in_batch_end(self):
        """Test loss anomaly detection in on_train_batch_end."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()
        trainer = Mock()
        pl_module = Mock()
        outputs = {'loss': torch.tensor(float('nan'))}

        with patch('builtins.print'):
            callback.on_train_batch_end(trainer, pl_module, outputs, None, 0)

        # Should detect NaN
        assert len(callback.warnings) == 1
        assert callback.current_step == 0

    def test_gradient_anomaly_detection_in_backward(self):
        """Test gradient anomaly detection in on_after_backward."""
        from neuromanifold_gpt.callbacks.anomaly_detector import AnomalyDetectorCallback

        callback = AnomalyDetectorCallback()
        callback.current_step = 5

        trainer = Mock()
        pl_module = Mock()
        param = Mock()
        param.grad = torch.tensor([float('nan')])
        pl_module.named_parameters.return_value = [('weight', param)]

        with patch('builtins.print'):
            callback.on_after_backward(trainer, pl_module)

        # Should detect NaN in gradients
        assert len(callback.warnings) == 1


class TestTrainingDashboardCallback:
    """Test suite for TrainingDashboardCallback."""

    def test_callback_initialization(self):
        """Test TrainingDashboardCallback initializes with correct defaults."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback()
        assert callback.refresh_rate == 100
        assert callback.training_start_time is None
        assert callback.max_steps is None
        assert callback.warmup_steps == 10

    def test_callback_custom_initialization(self):
        """Test TrainingDashboardCallback initializes with custom parameters."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback(
            refresh_rate=50,
            enable_dashboard=False
        )
        assert callback.refresh_rate == 50
        assert callback.enable_dashboard is False

    def test_eta_formatting(self):
        """Test ETA formatting for different time durations."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback()

        # Test seconds
        assert callback._format_eta(30) == "30s"

        # Test minutes
        assert callback._format_eta(150) == "2m 30s"

        # Test hours
        assert callback._format_eta(9000) == "2h 30m"

    def test_dashboard_generation_with_metrics(self):
        """Test that dashboard table is generated with metrics."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback()

        # Mock trainer with metrics
        trainer = Mock()
        trainer.logged_metrics = {
            'loss': torch.tensor(2.5),
            'train/grad_norm': torch.tensor(1.2),
            'train/memory_current_mb': torch.tensor(150.0),
        }
        trainer.global_step = 100

        pl_module = Mock()

        # Generate dashboard
        dashboard = callback._generate_dashboard(trainer, pl_module)

        # Should return a Table object
        from rich.table import Table
        assert isinstance(dashboard, Table)

    def test_dashboard_disabled_in_non_tty(self):
        """Test that dashboard is disabled in non-TTY environments."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        with patch('sys.stdout.isatty', return_value=False):
            callback = TrainingDashboardCallback()
            assert callback.enable_dashboard is False

    def test_training_start_sets_time_and_steps(self):
        """Test that on_train_start initializes timing variables."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback(enable_dashboard=False)

        trainer = Mock()
        trainer.max_steps = 1000
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.training_start_time is not None
        assert callback.max_steps == 1000

    def test_training_end_stops_dashboard(self):
        """Test that on_train_end stops the live dashboard."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback(enable_dashboard=False)
        callback.live = Mock()

        trainer = Mock()
        pl_module = Mock()

        callback.on_train_end(trainer, pl_module)

        # Should call stop on live dashboard
        callback.live.stop.assert_called_once()

    def test_dashboard_update_at_refresh_interval(self):
        """Test that dashboard updates at specified refresh rate."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback(refresh_rate=10, enable_dashboard=False)
        callback.live = Mock()

        trainer = Mock()
        trainer.logged_metrics = {}
        trainer.global_step = 10
        pl_module = Mock()

        # Batch index divisible by refresh_rate
        callback.on_train_batch_end(trainer, pl_module, None, None, 10)

        # Should update dashboard
        callback.live.update.assert_called_once()

    def test_no_dashboard_update_between_intervals(self):
        """Test that dashboard doesn't update between refresh intervals."""
        from neuromanifold_gpt.callbacks.training_dashboard import TrainingDashboardCallback

        callback = TrainingDashboardCallback(refresh_rate=10, enable_dashboard=False)
        callback.live = Mock()

        trainer = Mock()
        pl_module = Mock()

        # Batch index not divisible by refresh_rate
        callback.on_train_batch_end(trainer, pl_module, None, None, 5)

        # Should not update dashboard
        callback.live.update.assert_not_called()


class TestCallbackIntegration:
    """Integration tests for callback composition."""

    def test_all_callbacks_can_be_imported(self):
        """Test that all callbacks can be imported from the package."""
        from neuromanifold_gpt.callbacks import (
            GradientMonitorCallback,
            MemoryMonitorCallback,
            ThroughputMonitorCallback,
            LossMonitorCallback,
            AnomalyDetectorCallback,
            TrainingDashboardCallback
        )

        # All imports should succeed
        assert GradientMonitorCallback is not None
        assert MemoryMonitorCallback is not None
        assert ThroughputMonitorCallback is not None
        assert LossMonitorCallback is not None
        assert AnomalyDetectorCallback is not None
        assert TrainingDashboardCallback is not None

    def test_callbacks_can_be_instantiated_together(self):
        """Test that all callbacks can be instantiated together."""
        from neuromanifold_gpt.callbacks import (
            GradientMonitorCallback,
            MemoryMonitorCallback,
            ThroughputMonitorCallback,
            LossMonitorCallback,
            AnomalyDetectorCallback,
            TrainingDashboardCallback
        )

        # Create all callbacks
        callbacks = [
            GradientMonitorCallback(),
            MemoryMonitorCallback(),
            ThroughputMonitorCallback(),
            LossMonitorCallback(),
            AnomalyDetectorCallback(),
            TrainingDashboardCallback(enable_dashboard=False)
        ]

        # All callbacks should be created successfully
        assert len(callbacks) == 6
        for callback in callbacks:
            assert callback is not None

    def test_callbacks_inherit_from_lightning_callback(self):
        """Test that all callbacks inherit from PyTorch Lightning Callback."""
        from lightning.pytorch.callbacks import Callback
        from neuromanifold_gpt.callbacks import (
            GradientMonitorCallback,
            MemoryMonitorCallback,
            ThroughputMonitorCallback,
            LossMonitorCallback,
            AnomalyDetectorCallback,
            TrainingDashboardCallback
        )

        # Check inheritance
        assert issubclass(GradientMonitorCallback, Callback)
        assert issubclass(MemoryMonitorCallback, Callback)
        assert issubclass(ThroughputMonitorCallback, Callback)
        assert issubclass(LossMonitorCallback, Callback)
        assert issubclass(AnomalyDetectorCallback, Callback)
        assert issubclass(TrainingDashboardCallback, Callback)
