# test_websocket_connection.py
"""
Simple test to verify WebSocket connection to Polygon server
"""
import asyncio
import websockets
import json

async def test_connection():
    """Test direct WebSocket connection"""
    url = "ws://localhost:8200/ws/test_client"
    
    print(f"Connecting to {url}...")
    
    try:
        async with websockets.connect(url) as websocket:
            print("Connected!")
            
            # Receive welcome message
            welcome = await websocket.recv()
            print(f"Welcome message: {welcome}")
            
            # Send subscription
            sub_msg = {
                "action": "subscribe",
                "symbols": ["AAPL"],
                "channels": ["T", "Q"]
            }
            
            print(f"Sending subscription: {sub_msg}")
            await websocket.send(json.dumps(sub_msg))
            
            # Listen for a few messages
            for i in range(10):
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                print(f"Received: {data.get('type', 'unknown')} - {data}")
                
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())