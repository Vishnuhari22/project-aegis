# server.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
from aegis_core import process_video # Import our core logic

app = FastAPI()

# Read the HTML file for the frontend
with open("index.html", "r") as f:
    html = f.read()

@app.get("/")
async def get():
    """Serves the main HTML page."""
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection."""
    await websocket.accept()
    print("Client connected")
    try:
        # Start the video processing task
        await process_video(websocket)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011, reason=f"An error occurred: {e}")
