"""Print a compact scene/length catalog for the official test-scenes val episodes."""

import gzip
import json
from pathlib import Path


root = Path("/srv/navigation/data")
path = root / "datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz"
with gzip.open(path, "rt", encoding="utf-8") as source:
    episodes = json.load(source)["episodes"]
print("total_episodes", len(episodes))
for index, episode in enumerate(episodes[:8]):
    print(
        index,
        episode["scene_id"],
        episode.get("info", {}).get("geodesic_distance"),
    )
