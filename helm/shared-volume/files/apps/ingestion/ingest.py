"""Append synthetic events to the Delta Lake bronze table."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from apps.monitoring.metrics import JobMetrics


BRONZE_PATH = Path('/shared/lake/bronze/events')


def delta_version(path: Path) -> int:
    return DeltaTable(str(path)).version()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    metrics = JobMetrics('ingestion')
    now = datetime.now(timezone.utc)

    try:
        rows = [
            {
                'user_id': user_id,
                'event_value': user_id * 3.14,
                'event_time': now,
                'ingested_at': now,
            }
            for user_id in range(1, 11)
        ]
        events = pd.DataFrame(rows)
        BRONZE_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_deltalake(str(BRONZE_PATH), pa.Table.from_pandas(events, preserve_index=False), mode='append')
        version = delta_version(BRONZE_PATH)

        logging.info('Appended %d events to Delta bronze table %s (version=%d)', len(events), BRONZE_PATH, version)
        metrics.publish(
            success=True,
            rows=len(events),
            custom_metrics={'delta_version': version},
        )
    except Exception:
        metrics.publish(success=False)
        raise


if __name__ == '__main__':
    main()
