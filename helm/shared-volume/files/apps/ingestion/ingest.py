import os
from pathlib import Path
from datetime import datetime
import pandas as pd

RAW_DIR = Path('/shared/data/raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for user_id in range(1, 11):
    rows.append({
        'user_id': user_id,
        'event_value': user_id * 3.14,
        'event_time': datetime.utcnow().isoformat(),
    })

df = pd.DataFrame(rows)
file_path = RAW_DIR / f'raw_events_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(file_path, index=False)

print(f'Ingested data to {file_path}')
