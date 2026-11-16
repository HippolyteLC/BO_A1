
import json
from pathlib import Path

best_record = None

for file in Path(__file__).parent.rglob("log.jsonl"):
    best_record = None
    best_row = None

    with file.open("r", encoding="utf-8") as f:
        for row_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)

                if "fitness" not in record:
                    continue

                if best_record is None or record["fitness"] > best_record["fitness"]:
                    best_record = record
                    best_row = row_number

            except json.JSONDecodeError:
                continue

    if best_record:
        print(
            f"File: {file} | "
            f"Best fitness: {best_record['fitness']} | "
            f"Row: {best_row}"
        )
    else:
        print(f"File: {file} | No valid fitness scores")


