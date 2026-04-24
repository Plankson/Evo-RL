import logging
import time
import socket
import numpy as np
import umsgpack
import websockets.sync.client
from typing import Dict, Optional, Tuple, Any

class MsgpackNumpy:
    """Simple helper to pack/unpack numpy arrays with msgpack."""
    @staticmethod
    def packb(obj: Any) -> bytes:
        def encode(o):
            if isinstance(o, np.ndarray):
                return {
                    "__nd__": True,
                    "type": str(o.dtype),
                    "shape": list(o.shape),
                    "data": o.tobytes(),
                }
            return o
        return umsgpack.packb(obj, default=encode)

    @staticmethod
    def unpackb(data: bytes) -> Any:
        def decode(obj):
            if "__nd__" in obj:
                return np.frombuffer(obj["data"], dtype=obj["type"]).reshape(obj["shape"])
            return obj
        return umsgpack.unpackb(data, ext_hook=decode)

class SimpleWebsocketClient:
    """
    Standalone Websocket Client that avoids external openpi dependency.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
            
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata_raw = conn.recv()
                metadata = MsgpackNumpy.unpackb(metadata_raw)
                return conn, metadata
            except (ConnectionRefusedError, socket.error):
                logging.info("Still waiting for server...")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error connecting to server: {e}")
                time.sleep(2)

    def infer(self, obs: Dict) -> Dict:
        data = MsgpackNumpy.packb(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return MsgpackNumpy.unpackb(response)
