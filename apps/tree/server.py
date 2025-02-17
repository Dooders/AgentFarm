import os
import sys

import uvicorn

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

if __name__ == "__main__":
    print("Starting server...")
    print("View the family tree at: http://localhost:8000")
    print("To force refresh: http://localhost:8000/refresh")
    uvicorn.run("apps.tree.app:app", host="0.0.0.0", port=8000, workers=1, reload=True)
