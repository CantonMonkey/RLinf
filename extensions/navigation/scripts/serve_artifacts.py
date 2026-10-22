#!/usr/bin/env python3
"""Serve an explicit artifact directory over loopback HTTP with byte ranges.

This is a read-only file server for offline videos and result files.  It has no
directory index, upload endpoint, or network-facing bind option.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
import stat
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit


_RANGE = re.compile(r"bytes=(\d*)-(\d*)\Z", re.ASCII)
_CHUNK_SIZE = 1024 * 1024


def parse_single_range(value: str, size: int) -> tuple[int, int]:
    """Return an inclusive byte range, or raise ValueError for invalid ranges."""
    match = _RANGE.fullmatch(value.strip())
    if match is None or not any(match.groups()) or size == 0:
        raise ValueError("invalid or unsatisfiable byte range")

    first, last = match.groups()
    try:
        if first:
            start = int(first)
            end = int(last) if last else size - 1
            if start >= size or end < start:
                raise ValueError("unsatisfiable byte range")
            return start, min(end, size - 1)

        suffix_length = int(last)
        if suffix_length == 0:
            raise ValueError("empty suffix range")
        return max(0, size - suffix_length), size - 1
    except ValueError as exc:
        raise ValueError("invalid or unsatisfiable byte range") from exc


class ArtifactHandler(BaseHTTPRequestHandler):
    """GET/HEAD only; all resolved files must stay inside the selected root."""

    def __init__(self, *args, root: Path, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def _empty_response(self, status: int, *, content_range: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Accept-Ranges", "bytes")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()

    def _resolve_file(self) -> tuple[Path | None, int]:
        url = urlsplit(self.path)
        if url.scheme or url.netloc or url.query or url.fragment:
            return None, 403
        try:
            path = unquote_to_bytes(url.path).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None, 403
        if not path.startswith("/") or path == "/" or "\\" in path:
            return None, 403
        if any(ord(char) < 32 or ord(char) == 127 for char in path):
            return None, 403
        parts = path[1:].split("/")
        if any(part in ("", ".", "..") or ":" in part for part in parts):
            return None, 403

        try:
            target = self.root.joinpath(*parts).resolve(strict=True)
        except FileNotFoundError:
            return None, 404
        except OSError:
            return None, 403
        if not target.is_relative_to(self.root):
            return None, 403
        return target, 200

    def _serve_file(self, *, send_body: bool) -> None:
        target, status = self._resolve_file()
        if target is None:
            self._empty_response(status)
            return
        try:
            source = target.open("rb")
        except FileNotFoundError:
            self._empty_response(404)
            return
        except OSError:
            self._empty_response(403)
            return

        with source:
            file_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                self._empty_response(403)
                return
            size = file_stat.st_size
            ranges = self.headers.get_all("Range", [])
            if ranges:
                try:
                    if len(ranges) != 1:
                        raise ValueError("multiple Range headers")
                    start, end = parse_single_range(ranges[0], size)
                except ValueError:
                    self._empty_response(416, content_range=f"bytes */{size}")
                    return
                response_status = 206
                length = end - start + 1
            else:
                start = 0
                length = size
                response_status = 200

            mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(response_status)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if response_status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()

            if send_body:
                source.seek(start)
                remaining = length
                try:
                    while remaining:
                        chunk = source.read(min(_CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def do_GET(self) -> None:
        self._serve_file(send_body=True)

    def do_HEAD(self) -> None:
        self._serve_file(send_body=False)

    def _method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _method_not_allowed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory to serve")
    parser.add_argument("--port", type=int, required=True, help="127.0.0.1 TCP port")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        root = args.root.expanduser().resolve(strict=True)
    except OSError as exc:
        parser.error(f"--root is unavailable: {exc}")
    if not root.is_dir():
        parser.error("--root must be a directory")

    handler = partial(ArtifactHandler, root=root)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Serving {root} at http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
