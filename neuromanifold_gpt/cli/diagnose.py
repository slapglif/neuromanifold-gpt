#!/usr/bin/env python3
"""CLI diagnostic tool for training stability issues.

This command-line tool analyzes training logs to identify common stability issues:
- NaN/Inf detection in loss or gradients
- Loss spikes (sudden large increases)
- Gradient explosions
- SDR collapse patterns
- Memory issues
- Learning rate anomalies

It produces a rich-formatted report with:
- Summary table of detected issues
- Timeline of events
- Severity levels (CRITICAL, WARNING, INFO)
- Suggested remediation steps

Usage:
    # Analyze a single log file
    python -m neuromanifold_gpt.cli.diagnose logs/training.log

    # Analyze all logs in a directory
    python -m neuromanifold_gpt.cli.diagnose logs/

    # Show verbose output
    python -m neuromanifold_gpt.cli.diagnose logs/training.log --verbose

    # Filter by issue type
    python -m neuromanifold_gpt.cli.diagnose logs/ --issue-type loss_spike

    # Set custom thresholds
    python -m neuromanifold_gpt.cli.diagnose logs/ --loss-spike-threshold 2.5
"""

import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neuromanifold_gpt.cli.help_formatter import RichArgumentParser


@dataclass
class TrainingIssue:
    """Represents a detected training issue.

    Attributes:
        step: Training step where issue occurred
        issue_type: Type of issue (e.g., 'loss_spike', 'nan_detected', 'sdr_collapse')
        severity: Severity level ('CRITICAL', 'WARNING', 'INFO')
        message: Detailed message describing the issue
        metrics: Optional dict of relevant metrics at time of issue
        suggestions: List of suggested remediation steps
    """

    step: int
    issue_type: str
    severity: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class LogParser:
    """Parse training logs to extract metrics and detect issues.

    Supports both loguru-formatted logs and plain text logs with metric patterns.
    """

    # Regex patterns for extracting metrics from logs
    STEP_PATTERN = re.compile(r'step[:\s]+(\d+)', re.IGNORECASE)
    LOSS_PATTERN = re.compile(r'loss[:\s]+([\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE)
    GRAD_NORM_PATTERN = re.compile(r'grad[_\s]?norm[:\s]+([\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE)
    LR_PATTERN = re.compile(r'(?:learning[_\s]?rate|lr)[:\s]+([\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE)
    NAN_PATTERN = re.compile(r'\bnan\b', re.IGNORECASE)
    INF_PATTERN = re.compile(r'\binf\b', re.IGNORECASE)

    # SDR-specific patterns
    SDR_COLLAPSE_PATTERN = re.compile(r'SDR COLLAPSE DETECTED', re.IGNORECASE)
    SDR_LOW_DIVERSITY_PATTERN = re.compile(
        r'Low SDR diversity.*?(\d+\.?\d*)%.*?unique patterns', re.IGNORECASE
    )
    SDR_BIT_USAGE_PATTERN = re.compile(r'bit.*?usage[:\s]+([\d.]+)', re.IGNORECASE)

    # Divergence and rollback patterns
    DIVERGENCE_PATTERN = re.compile(r'DIVERGENCE DETECTED|Loss spike detected', re.IGNORECASE)
    ROLLBACK_PATTERN = re.compile(r'Rolling back|Rollback to', re.IGNORECASE)

    def __init__(
        self,
        loss_spike_threshold: float = 3.0,
        grad_explosion_threshold: float = 3.0,
        min_samples_for_detection: int = 20,
    ):
        """Initialize log parser with detection thresholds.

        Args:
            loss_spike_threshold: Number of std devs for loss spike detection
            grad_explosion_threshold: Number of std devs for gradient explosion
            min_samples_for_detection: Minimum samples before detecting anomalies
        """
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_explosion_threshold = grad_explosion_threshold
        self.min_samples_for_detection = min_samples_for_detection

        # Tracking for anomaly detection
        self.loss_history: deque = deque(maxlen=100)
        self.grad_norm_history: deque = deque(maxlen=100)

    def parse_log_file(self, log_path: Path) -> List[TrainingIssue]:
        """Parse a single log file and return detected issues.

        Args:
            log_path: Path to log file

        Returns:
            List of TrainingIssue objects
        """
        issues = []

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    # Extract step number
                    step_match = self.STEP_PATTERN.search(line)
                    step = int(step_match.group(1)) if step_match else line_num

                    # Check for explicit issue markers first
                    issue = self._check_explicit_issues(line, step)
                    if issue:
                        issues.append(issue)
                        continue

                    # Extract metrics and check for anomalies
                    metrics = self._extract_metrics(line)
                    if metrics:
                        anomaly_issues = self._detect_anomalies(metrics, step)
                        issues.extend(anomaly_issues)

        except Exception as e:
            console = Console()
            console.print(f"[yellow]Warning: Error parsing {log_path}: {e}[/yellow]")

        return issues

    def _extract_metrics(self, line: str) -> Dict[str, float]:
        """Extract numerical metrics from a log line.

        Args:
            line: Log line text

        Returns:
            Dict of metric names to values
        """
        metrics = {}

        # Extract loss
        loss_match = self.LOSS_PATTERN.search(line)
        if loss_match:
            try:
                metrics['loss'] = float(loss_match.group(1))
                self.loss_history.append(metrics['loss'])
            except ValueError:
                pass

        # Extract gradient norm
        grad_match = self.GRAD_NORM_PATTERN.search(line)
        if grad_match:
            try:
                metrics['grad_norm'] = float(grad_match.group(1))
                self.grad_norm_history.append(metrics['grad_norm'])
            except ValueError:
                pass

        # Extract learning rate
        lr_match = self.LR_PATTERN.search(line)
        if lr_match:
            try:
                metrics['lr'] = float(lr_match.group(1))
            except ValueError:
                pass

        return metrics

    def _check_explicit_issues(self, line: str, step: int) -> Optional[TrainingIssue]:
        """Check for explicit issue markers in log line.

        Args:
            line: Log line text
            step: Training step number

        Returns:
            TrainingIssue if issue detected, None otherwise
        """
        # Check for NaN
        if self.NAN_PATTERN.search(line):
            return TrainingIssue(
                step=step,
                issue_type='nan_detected',
                severity='CRITICAL',
                message=f'NaN detected in logs: {line.strip()[:100]}',
                suggestions=[
                    'Check for numerical instability in loss computation',
                    'Reduce learning rate',
                    'Enable gradient clipping',
                    'Check input data for NaN values',
                ],
            )

        # Check for Inf
        if self.INF_PATTERN.search(line):
            return TrainingIssue(
                step=step,
                issue_type='inf_detected',
                severity='CRITICAL',
                message=f'Inf detected in logs: {line.strip()[:100]}',
                suggestions=[
                    'Reduce learning rate',
                    'Enable gradient clipping',
                    'Check for division by zero',
                ],
            )

        # Check for SDR collapse
        if self.SDR_COLLAPSE_PATTERN.search(line):
            # Try to extract diversity percentage
            diversity_match = self.SDR_LOW_DIVERSITY_PATTERN.search(line)
            diversity = diversity_match.group(1) if diversity_match else 'unknown'

            return TrainingIssue(
                step=step,
                issue_type='sdr_collapse',
                severity='WARNING',
                message=f'SDR collapse detected: {diversity}% unique patterns',
                suggestions=[
                    'Increase temperature parameter',
                    'Check token discrimination loss',
                    'Verify diversity targets are being met',
                    'Review SDR sparsity settings',
                ],
            )

        # Check for divergence
        if self.DIVERGENCE_PATTERN.search(line):
            return TrainingIssue(
                step=step,
                issue_type='divergence',
                severity='CRITICAL',
                message=f'Training divergence detected: {line.strip()[:100]}',
                suggestions=[
                    'Review recent hyperparameter changes',
                    'Enable automatic checkpoint rollback',
                    'Reduce learning rate',
                    'Increase gradient clipping threshold',
                ],
            )

        # Check for rollback events
        if self.ROLLBACK_PATTERN.search(line):
            return TrainingIssue(
                step=step,
                issue_type='rollback',
                severity='WARNING',
                message=f'Checkpoint rollback occurred: {line.strip()[:100]}',
                suggestions=[
                    'Review what caused divergence before rollback',
                    'Consider more conservative hyperparameters',
                ],
            )

        return None

    def _detect_anomalies(self, metrics: Dict[str, float], step: int) -> List[TrainingIssue]:
        """Detect anomalies in extracted metrics using statistical methods.

        Args:
            metrics: Dict of metric names to values
            step: Training step number

        Returns:
            List of detected TrainingIssue objects
        """
        issues = []

        # Detect loss spikes
        if 'loss' in metrics and len(self.loss_history) >= self.min_samples_for_detection:
            loss = metrics['loss']
            loss_mean = sum(self.loss_history) / len(self.loss_history)
            loss_std = (
                sum((x - loss_mean) ** 2 for x in self.loss_history) / len(self.loss_history)
            ) ** 0.5

            if loss_std > 0 and loss > loss_mean + self.loss_spike_threshold * loss_std:
                issues.append(
                    TrainingIssue(
                        step=step,
                        issue_type='loss_spike',
                        severity='WARNING',
                        message=f'Loss spike detected: {loss:.4f} (mean: {loss_mean:.4f}, std: {loss_std:.4f})',
                        metrics={'loss': loss, 'mean': loss_mean, 'std': loss_std},
                        suggestions=[
                            'Review batch composition',
                            'Check for data outliers',
                            'Consider lowering learning rate',
                            'Enable gradient clipping',
                        ],
                    )
                )

        # Detect gradient explosions
        if 'grad_norm' in metrics and len(self.grad_norm_history) >= self.min_samples_for_detection:
            grad_norm = metrics['grad_norm']
            grad_mean = sum(self.grad_norm_history) / len(self.grad_norm_history)
            grad_std = (
                sum((x - grad_mean) ** 2 for x in self.grad_norm_history)
                / len(self.grad_norm_history)
            ) ** 0.5

            if (
                grad_std > 0
                and grad_norm > grad_mean + self.grad_explosion_threshold * grad_std
            ):
                issues.append(
                    TrainingIssue(
                        step=step,
                        issue_type='gradient_explosion',
                        severity='WARNING',
                        message=f'Gradient explosion detected: {grad_norm:.4f} (mean: {grad_mean:.4f}, std: {grad_std:.4f})',
                        metrics={'grad_norm': grad_norm, 'mean': grad_mean, 'std': grad_std},
                        suggestions=[
                            'Enable or lower gradient clipping threshold',
                            'Reduce learning rate',
                            'Check for numerical instability',
                        ],
                    )
                )

        return issues


class DiagnosticReport:
    """Generate rich-formatted diagnostic reports."""

    def __init__(self, console: Console):
        """Initialize report generator.

        Args:
            console: Rich console for output
        """
        self.console = console

    def generate_summary(self, issues: List[TrainingIssue], log_files: List[Path]) -> None:
        """Generate and display summary report.

        Args:
            issues: List of detected issues
            log_files: List of analyzed log files
        """
        # Create summary panel
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Training Diagnostic Report[/bold cyan]\n"
                f"Analyzed {len(log_files)} log file(s)\n"
                f"Found {len(issues)} issue(s)",
                border_style="cyan",
            )
        )
        self.console.print()

        if not issues:
            self.console.print("[green]✓ No issues detected! Training appears healthy.[/green]")
            return

        # Group issues by type
        issues_by_type = defaultdict(list)
        for issue in issues:
            issues_by_type[issue.issue_type].append(issue)

        # Create summary table
        summary_table = Table(title="Issue Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Issue Type", style="cyan", no_wrap=True)
        summary_table.add_column("Count", justify="right", style="yellow")
        summary_table.add_column("Severity", justify="center")

        for issue_type, type_issues in sorted(issues_by_type.items()):
            # Determine overall severity for this issue type
            severities = [issue.severity for issue in type_issues]
            if 'CRITICAL' in severities:
                severity_style = "[bold red]CRITICAL[/bold red]"
            elif 'WARNING' in severities:
                severity_style = "[yellow]WARNING[/yellow]"
            else:
                severity_style = "[blue]INFO[/blue]"

            summary_table.add_row(
                issue_type.replace('_', ' ').title(), str(len(type_issues)), severity_style
            )

        self.console.print(summary_table)
        self.console.print()

        # Create detailed timeline table
        timeline_table = Table(
            title="Issue Timeline", show_header=True, header_style="bold magenta"
        )
        timeline_table.add_column("Step", justify="right", style="cyan")
        timeline_table.add_column("Type", style="yellow")
        timeline_table.add_column("Severity", justify="center")
        timeline_table.add_column("Message", style="white", max_width=60)

        # Sort issues by step
        sorted_issues = sorted(issues, key=lambda x: x.step)

        for issue in sorted_issues[:50]:  # Limit to first 50 issues
            if issue.severity == 'CRITICAL':
                severity_style = "[bold red]CRITICAL[/bold red]"
            elif issue.severity == 'WARNING':
                severity_style = "[yellow]WARNING[/yellow]"
            else:
                severity_style = "[blue]INFO[/blue]"

            timeline_table.add_row(
                str(issue.step),
                issue.issue_type.replace('_', ' ').title(),
                severity_style,
                issue.message[:60],
            )

        if len(sorted_issues) > 50:
            timeline_table.add_row(
                "...", f"({len(sorted_issues) - 50} more issues)", "", "..."
            )

        self.console.print(timeline_table)
        self.console.print()

        # Display suggestions for critical issues
        critical_issues = [issue for issue in issues if issue.severity == 'CRITICAL']
        if critical_issues:
            self.console.print(
                Panel(
                    "[bold red]Critical Issues Require Immediate Attention[/bold red]",
                    border_style="red",
                )
            )
            self.console.print()

            # Show unique suggestions across all critical issues
            all_suggestions = set()
            for issue in critical_issues:
                all_suggestions.update(issue.suggestions)

            if all_suggestions:
                self.console.print("[bold yellow]Suggested Actions:[/bold yellow]")
                for i, suggestion in enumerate(sorted(all_suggestions), 1):
                    self.console.print(f"  {i}. {suggestion}")
                self.console.print()

    def generate_detailed_report(self, issues: List[TrainingIssue]) -> None:
        """Generate detailed report for each issue.

        Args:
            issues: List of detected issues
        """
        self.console.print("[bold cyan]Detailed Issue Analysis[/bold cyan]")
        self.console.print()

        # Group by issue type
        issues_by_type = defaultdict(list)
        for issue in issues:
            issues_by_type[issue.issue_type].append(issue)

        for issue_type, type_issues in sorted(issues_by_type.items()):
            self.console.print(f"[bold yellow]{issue_type.replace('_', ' ').title()}[/bold yellow]")
            self.console.print(f"Total occurrences: {len(type_issues)}")
            self.console.print()

            # Show first few examples
            for issue in type_issues[:3]:
                self.console.print(f"  Step {issue.step}: {issue.message}")
                if issue.metrics:
                    metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in issue.metrics.items())
                    self.console.print(f"  Metrics: {metrics_str}")

            if len(type_issues) > 3:
                self.console.print(f"  ... and {len(type_issues) - 3} more occurrences")

            # Show suggestions
            if type_issues[0].suggestions:
                self.console.print("  [cyan]Suggestions:[/cyan]")
                for suggestion in type_issues[0].suggestions:
                    self.console.print(f"    • {suggestion}")

            self.console.print()


def parse_args() -> Any:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = RichArgumentParser(
        description="Diagnose common training stability issues from log files",
        examples=[
            "python -m neuromanifold_gpt.cli.diagnose logs/training.log",
            "python -m neuromanifold_gpt.cli.diagnose logs/ --verbose",
            "python -m neuromanifold_gpt.cli.diagnose logs/ --issue-type loss_spike",
        ],
    )

    parser.add_argument(
        'log_path',
        type=str,
        help='Path to log file or directory containing log files',
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed report with all issues',
    )

    parser.add_argument(
        '--issue-type',
        type=str,
        default=None,
        choices=[
            'loss_spike',
            'gradient_explosion',
            'nan_detected',
            'inf_detected',
            'sdr_collapse',
            'divergence',
            'rollback',
        ],
        help='Filter by specific issue type',
    )

    parser.add_argument(
        '--loss-spike-threshold',
        type=float,
        default=3.0,
        help='Number of standard deviations for loss spike detection (default: 3.0)',
    )

    parser.add_argument(
        '--grad-explosion-threshold',
        type=float,
        default=3.0,
        help='Number of standard deviations for gradient explosion detection (default: 3.0)',
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=20,
        help='Minimum samples required before anomaly detection (default: 20)',
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for diagnose CLI.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()
    console = Console()

    # Determine log files to analyze
    log_path = Path(args.log_path)

    if not log_path.exists():
        console.print(f"[red]Error: Path does not exist: {log_path}[/red]")
        return 1

    if log_path.is_file():
        log_files = [log_path]
    elif log_path.is_dir():
        # Find all .log files in directory
        log_files = list(log_path.glob('*.log')) + list(log_path.glob('**/*.log'))
        if not log_files:
            console.print(f"[yellow]Warning: No .log files found in {log_path}[/yellow]")
            return 1
    else:
        console.print(f"[red]Error: Invalid path: {log_path}[/red]")
        return 1

    # Initialize parser
    parser = LogParser(
        loss_spike_threshold=args.loss_spike_threshold,
        grad_explosion_threshold=args.grad_explosion_threshold,
        min_samples_for_detection=args.min_samples,
    )

    # Parse all log files
    console.print(f"[cyan]Analyzing {len(log_files)} log file(s)...[/cyan]")
    all_issues = []

    for log_file in log_files:
        issues = parser.parse_log_file(log_file)
        all_issues.extend(issues)

    # Filter by issue type if requested
    if args.issue_type:
        all_issues = [issue for issue in all_issues if issue.issue_type == args.issue_type]

    # Generate report
    report = DiagnosticReport(console)
    report.generate_summary(all_issues, log_files)

    if args.verbose:
        report.generate_detailed_report(all_issues)

    # Return non-zero exit code if critical issues found
    critical_issues = [issue for issue in all_issues if issue.severity == 'CRITICAL']
    return 1 if critical_issues else 0


if __name__ == '__main__':
    sys.exit(main())
