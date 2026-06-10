from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

class ExperimentLogger:
    def __init__(self,log_path:str|Path):
        self.log_path=Path(log_path)
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.start_time=time.time()

    def log(self,data:dict[str,Any])-> None:
        record={
            "wallclock_time":time.time()-self.start_time,
            **data,
        }

        with self.log_path.open("a",encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
