import json
import websocket


ws = websocket.WebSocket()


# Define the message payload
data = {
    "op": "publish",
    "topic": "/topic",
    "msg": {
        "data": "111"
    }
}

# Convert the message payload to JSON
json_data = json.dumps(data)

ws.connect("ws://127.0.0.1:9090")
ws.send(json_data)

# Print the response from the ROS Bridge
print(ws.recv())