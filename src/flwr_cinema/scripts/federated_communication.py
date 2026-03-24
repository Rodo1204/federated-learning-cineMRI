"""Unified socket communication for federated learning supporting all training modes.

Handles chunked transfers with optional compression for:
- LoRA mode (small updates)
- Dual LoRA mode (medium updates)
- Full finetuning mode (large updates)
"""

import pickle
import socket
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_MESSAGE_SIZE = 2**31 - 1  # 2GB safety limit


def send_data_chunked(
    conn: socket.socket,
    data: dict,
    chunk_size: int = CHUNK_SIZE,
    use_compression: bool = False,
) -> None:
    """Send data in chunks with optional compression and progress logging.
    
    This is a drop-in replacement for the basic send_data method that handles large models.
    
    Args:
        conn: Socket connection.
        data: Data dictionary to send.
        chunk_size: Size of each chunk in bytes (default 1MB).
        use_compression: Whether to use zstd compression (auto-enabled for large messages).
    """
    # Serialize first
    serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    total_size = len(serialized)
    
    # Auto-detect compression if message is large
    if not use_compression and total_size > 50 * 1024 * 1024:  # Auto-compress > 50MB
        try:
            import zstandard as zstd
            use_compression = True
        except ImportError:
            use_compression = False
    
    # Optional compression
    if use_compression:
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor(level=10)
            serialized_compressed = cctx.compress(serialized)
            ratio = len(serialized_compressed) / len(serialized) * 100
            logger.info(f"Compressed {len(serialized) / (1024**2):.2f}MB → {len(serialized_compressed) / (1024**2):.2f}MB ({ratio:.1f}%)")
            serialized = serialized_compressed
            is_compressed = True
        except ImportError:
            logger.warning("zstandard not installed, skipping compression")
            is_compressed = False
    else:
        is_compressed = False
    
    if len(serialized) > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {len(serialized)} > {MAX_MESSAGE_SIZE}")
    
    # Send header: [total_size:4 bytes][is_compressed:1 byte]
    header = len(serialized).to_bytes(4, byteorder="big") + bytes([1 if is_compressed else 0])
    conn.sendall(header)
    
    # Send in chunks with progress logging
    bytes_sent = 0
    logger.info(f"Sending {len(serialized) / (1024**2):.2f}MB ({len(data)} keys)")
    
    while bytes_sent < len(serialized):
        chunk = serialized[bytes_sent : bytes_sent + chunk_size]
        conn.sendall(chunk)
        bytes_sent += len(chunk)
        
        # Log progress every 10%
        if bytes_sent % max(1, len(serialized) // 10) < chunk_size or bytes_sent == len(serialized):
            pct = (bytes_sent / len(serialized)) * 100
            logger.info(f"  Sent {bytes_sent / (1024**2):.2f}MB ({pct:.0f}%)")
    
    logger.info(f"✓ Sent {len(serialized) / (1024**2):.2f}MB successfully")


def receive_data_chunked(
    conn: socket.socket,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Receive data in chunks with automatic decompression and progress logging.
    
    This is a drop-in replacement for the basic receive_data method that handles large models.
    
    Args:
        conn: Socket connection.
        chunk_size: Size of each chunk in bytes (default 1MB).
        
    Returns:
        Received data dictionary.
    """
    # Receive header: [total_size:4 bytes][is_compressed:1 byte]
    header = conn.recv(5)
    if len(header) < 5:
        raise ConnectionError("Connection closed before header received")
    
    total_size = int.from_bytes(header[:4], byteorder="big")
    is_compressed = bool(header[4])
    
    if total_size > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size exceeded: {total_size} > {MAX_MESSAGE_SIZE}")
    
    logger.info(f"Expecting {total_size / (1024**2):.2f}MB (compressed={is_compressed})")
    
    # Receive in chunks with progress logging
    data = b""
    while len(data) < total_size:
        remaining = total_size - len(data)
        chunk_to_recv = min(chunk_size, remaining)
        
        try:
            chunk = conn.recv(chunk_to_recv)
        except socket.timeout:
            raise ConnectionError(f"Socket timeout after {len(data)}/{total_size} bytes")
        
        if not chunk:
            raise ConnectionError(f"Connection closed after {len(data)}/{total_size} bytes")
        
        data += chunk
        
        # Log progress every 10%
        if len(data) % max(1, total_size // 10) < len(chunk) or len(data) == total_size:
            pct = (len(data) / total_size) * 100
            logger.info(f"  Received {len(data) / (1024**2):.2f}MB ({pct:.0f}%)")
    
    # Decompress if needed
    if is_compressed:
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            data = dctx.decompress(data)
            logger.info(f"Decompressed to {len(data) / (1024**2):.2f}MB")
        except ImportError:
            logger.error("zstandard not installed but compression flag set!")
            raise
    
    logger.info(f"✓ Received {len(data) / (1024**2):.2f}MB. Unpickling...")
    obj = pickle.loads(data)
    logger.info(f"Unpickled {len(obj)} keys")
    
    return obj
