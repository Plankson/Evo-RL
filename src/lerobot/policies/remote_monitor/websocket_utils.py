import logging
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import umsgpack
import websockets.sync.client


class MsgpackNumpy:
    """Simple helper to pack/unpack numpy arrays with msgpack."""

    @staticmethod
    def packb(obj: Any) -> bytes:
        def normalize(o: Any) -> Any:
            if isinstance(o, np.ndarray):
                return {
                    "__nd__": True,
                    "type": str(o.dtype),
                    "shape": list(o.shape),
                    "data": o.tobytes(),
                }
            if isinstance(o, dict):
                return {k: normalize(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [normalize(v) for v in o]
            return o

        return umsgpack.packb(normalize(obj))

    @staticmethod
    def unpackb(data: bytes) -> Any:
        def decode(obj):
            if "__nd__" in obj:
                return np.frombuffer(obj["data"], dtype=obj["type"]).reshape(obj["shape"])
            return obj

        return umsgpack.unpackb(data, ext_hook=decode)


class SimpleWebsocketClient:
    """
    Standalone websocket client for monitor endpoints.

    Some deployed servers send a metadata frame immediately after connect,
    while others only respond after the first inference request. Support both.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        metadata_timeout_s: float = 1.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._metadata_timeout_s = metadata_timeout_s
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._server_metadata: Dict[str, Any] = {}

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        logging.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=5.0,
                )

                try:
                    metadata_raw = conn.recv(timeout=self._metadata_timeout_s)
                    metadata = (
                        {"message": metadata_raw}
                        if isinstance(metadata_raw, str)
                        else MsgpackNumpy.unpackb(metadata_raw)
                    )
                except TimeoutError:
                    logging.info(
                        "Connected to %s without startup metadata; continuing with empty metadata.",
                        self._uri,
                    )
                    metadata = {}

                return conn, metadata
            except (ConnectionRefusedError, socket.error):
                logging.info("Still waiting for server...")
                time.sleep(2)
            except Exception as e:
                logging.error("Error connecting to server: %s", e)
                time.sleep(2)

    def _ensure_connected(self) -> None:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

    def _reset_connection(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_connected()
        data = MsgpackNumpy.packb(obs)

        try:
            self._ws.send(data)
            response = self._ws.recv()
        except Exception:
            self._reset_connection()
            self._ensure_connected()
            self._ws.send(data)
            response = self._ws.recv()

        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return MsgpackNumpy.unpackb(response)
