import subprocess
import os
from pathlib import Path
import sys

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ[key.strip()] = val.strip()

from app.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5271"))
    uvicorn.run(app, host="0.0.0.0", port=port)
