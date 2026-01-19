from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import KDTree
import pandas as pd
import numpy as np
import shutil
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
]


@app.get("/")
def read_root():
    return {"Hello": "World"}


app = FastAPI()
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Reading CSV for querying
metadata_df = pd.read_csv("dataset/millionSongs.csv")

# This is used for the identification purpose in the /i API
coords = np.fromfile("dataset/scattered_coords_baked.bin", dtype=np.float32).reshape(
    -1, 3
)
points_xy = coords[:, :2]
tree = KDTree(points_xy)


# Identify songs near a given 2D coordinate
#
# Description:
#   Returns metadata for the k-nearest songs around a given (x, y)
#   position. Used by the frontend to identify and "snap" to nearby
#   song points in the mesh.
#
# Query Params:
#   x (float)  : X coordinate in mesh space
#   y (float)  : Y coordinate in mesh space
#   k (int)    : Number of nearest points to fetch (default: 200)
#
# Notes:
#   Uses a KD-tree for fast nearest-neighbor lookup.
@app.get("/i")
async def identify_area(x: float, y: float, k: int = 200):
    # Find k nearest points
    distances, indices = tree.query([x, y], k=k)

    results = []
    for i, idx in enumerate(indices):
        # We include the baked X/Y so the frontend can "snap" to the dot
        results.append(
            {
                "id": int(idx),
                "x": float(points_xy[idx, 0]),
                "y": float(points_xy[idx, 1]),
                "data": metadata_df.iloc[idx].to_dict(),
            }
        )
    return results


# ------------------------------------------------------------
# Serves baked mesh coordinates for frontend rendering
#
# Description:
#   Serves the preprocessed and baked mesh coordinates used by the
#   frontend for rendering the song map.
#
# Notes:
#   - Coordinates are already transformed (baked)
@app.get("/load-mesh/")
async def load_mesh():
    return FileResponse(
        "dataset/scattered_coords_baked.bin", media_type="application/octet-stream"
    )


# ------------------------------------------------------------
# [Not in use] Serve raw coordinate data (no spiral applied)
#
# Description:
#   Serves the original coordinate binary file without spiral
#   transformation. Kept only for debugging and comparison.
#
# Usage:
#   Debugging, visual verification, testing
@app.get("/songs/coords")
def get_coords():
    return FileResponse("dataset/coords_new.bin", media_type="application/octet-stream")
