# Installation

## Prerequisites

- Python 3.8 or higher (3.9+ recommended)
- pip and Git
- Redis (optional, for enhanced agent memory)

## Setup

```bash
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Editable install (`pip install -e .`) is required for `farm` imports.

## Optional: Redis

Redis is used for some agent memory features and can improve performance.

- Ubuntu/Debian: `sudo apt-get install redis-server`
- macOS: `brew install redis`
- Windows: [redis.io/download](https://redis.io/download)

Start the server with `redis-server`.

## Verify

```bash
pytest -q
```

## Next steps

- [Run your first simulation](first-simulation.md)
- [Contributing](../../CONTRIBUTING.md)
