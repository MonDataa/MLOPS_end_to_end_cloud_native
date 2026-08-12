"""Metrics helpers shared by the short-lived pipeline jobs.

Prometheus cannot reliably scrape a Kubernetes Job after it has completed, so
each job pushes its final state to Pushgateway.  The metric names are stable
and do not include run-specific labels to keep Prometheus cardinality bounded.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Mapping

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


PUSHGATEWAY_URL = os.environ.get('PUSHGATEWAY_URL', 'http://mlops-pushgateway:9091')


@dataclass
class JobMetrics:
    """Collect and publish the final result for one pipeline job."""

    job_name: str
    started_at: float = field(default_factory=time.time)

    def publish(
        self,
        *,
        success: bool,
        rows: int = 0,
        custom_metrics: Mapping[str, float] | None = None,
    ) -> None:
        registry = CollectorRegistry()
        labels = {'job': self.job_name}
        duration_seconds = max(time.time() - self.started_at, 0.0)

        Gauge(
            'mlops_job_last_run_success',
            'Whether the most recent pipeline job run succeeded (1 or 0).',
            ['job'],
            registry=registry,
        ).labels(**labels).set(1 if success else 0)
        Gauge(
            'mlops_job_last_run_duration_seconds',
            'Duration of the most recent pipeline job run.',
            ['job'],
            registry=registry,
        ).labels(**labels).set(duration_seconds)
        Gauge(
            'mlops_job_last_run_rows',
            'Rows processed by the most recent pipeline job run.',
            ['job'],
            registry=registry,
        ).labels(**labels).set(rows)
        Gauge(
            'mlops_job_last_run_timestamp_seconds',
            'Unix timestamp of the most recent pipeline job run.',
            ['job'],
            registry=registry,
        ).labels(**labels).set(time.time())

        metric_gauge = Gauge(
            'mlops_job_custom_metric',
            'Additional pipeline quality and Delta Lake metrics.',
            ['job', 'metric'],
            registry=registry,
        )
        for metric_name, value in (custom_metrics or {}).items():
            metric_gauge.labels(job=self.job_name, metric=metric_name).set(value)

        try:
            push_to_gateway(PUSHGATEWAY_URL, job=self.job_name, registry=registry)
        except Exception:
            # Monitoring outages must not discard a successfully written Delta table.
            logging.exception('Unable to push %s metrics to %s', self.job_name, PUSHGATEWAY_URL)
