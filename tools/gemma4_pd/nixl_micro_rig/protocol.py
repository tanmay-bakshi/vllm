"""Length-prefixed JSON control protocol for the two micro-rig roles."""

import json
import socket
import struct
from dataclasses import dataclass


class ProtocolError(RuntimeError):
    """Report malformed or truncated control traffic."""


_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
_PROTOCOL_VERSION = 1


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    """Read exactly ``size`` bytes or fail.

    :param connection: Connected TCP socket.
    :param size: Required byte count.
    :returns: Received bytes.
    :raises ProtocolError: If the peer closes the stream early.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = connection.recv(remaining)
        if len(chunk) == 0:
            raise ProtocolError(
                f"control connection closed with {remaining} bytes outstanding"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class JsonChannel:
    """Exchange bounded, length-prefixed JSON objects.

    :ivar connection: Connected TCP socket owned by the channel.
    """

    connection: socket.socket

    def send(self, message: dict[str, object]) -> None:
        """Send one JSON object.

        :param message: Object to encode and send.
        :raises ProtocolError: If the encoded message exceeds the protocol limit.
        """
        if "protocol_version" in message:
            raise ProtocolError("callers must not override protocol_version")
        if not isinstance(message.get("type"), str):
            raise ProtocolError("control message requires a string type")
        envelope = {"protocol_version": _PROTOCOL_VERSION, **message}
        payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise ProtocolError(f"control message is too large: {len(payload)} bytes")
        self.connection.sendall(_HEADER.pack(len(payload)) + payload)

    def receive(self) -> dict[str, object]:
        """Receive and validate one JSON object.

        :returns: Decoded JSON object.
        :raises ProtocolError: If the frame is invalid or does not contain an object.
        """
        size = _HEADER.unpack(_receive_exact(self.connection, _HEADER.size))[0]
        if size > _MAX_MESSAGE_BYTES:
            raise ProtocolError(f"control message is too large: {size} bytes")
        payload = _receive_exact(self.connection, size)
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ProtocolError(f"invalid control JSON: {error}") from error
        if not isinstance(message, dict) or not all(
            isinstance(key, str) for key in message
        ):
            raise ProtocolError("control message must be a JSON object")
        version = message.pop("protocol_version", None)
        if version != _PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported control protocol version: {version!r}")
        if not isinstance(message.get("type"), str):
            raise ProtocolError("control message requires a string type")
        return message

    def close(self) -> None:
        """Close the owned socket."""
        self.connection.close()

    def __enter__(self) -> "JsonChannel":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def expect_message(channel: JsonChannel, message_type: str) -> dict[str, object]:
    """Receive one message and require its declared type.

    :param channel: Connected JSON channel.
    :param message_type: Required ``type`` field.
    :returns: Received message.
    :raises ProtocolError: If the type does not match.
    """
    message = channel.receive()
    actual = message.get("type")
    if actual != message_type:
        raise ProtocolError(f"expected message {message_type!r}, received {actual!r}")
    return message
