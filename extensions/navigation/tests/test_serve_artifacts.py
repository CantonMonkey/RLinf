"""Black-box HTTP checks for the loopback artifact server."""

import http.client
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts.serve_artifacts import ArtifactHandler


class ArtifactHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        temp = Path(self.temp_dir.name)
        self.root = temp / "shared"
        self.root.mkdir()
        (self.root / "nested").mkdir()
        (self.root / "clip.mp4").write_bytes(b"0123456789")
        (self.root / "nested" / "trace.jsonl").write_bytes(b"{}\n")
        (temp / "outside.txt").write_text("SECRET", encoding="utf-8")

        handler = partial(ArtifactHandler, root=self.root.resolve())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.server.server_close()

    def request(self, method, path, headers=None, body=None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=3
        )
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_full_get_and_head(self):
        status, headers, body = self.request("GET", "/clip.mp4")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Length"], "10")
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(body, b"0123456789")

        status, headers, body = self.request("HEAD", "/nested/trace.jsonl")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Length"], "3")
        self.assertEqual(body, b"")

    def test_explicit_open_ended_and_suffix_ranges(self):
        for range_header, expected_range, expected_body in (
            ("bytes=2-4", "bytes 2-4/10", b"234"),
            ("bytes=7-", "bytes 7-9/10", b"789"),
            ("bytes=-3", "bytes 7-9/10", b"789"),
            ("bytes=8-100", "bytes 8-9/10", b"89"),
        ):
            with self.subTest(range_header=range_header):
                status, headers, body = self.request(
                    "GET", "/clip.mp4", {"Range": range_header}
                )
                self.assertEqual(status, 206)
                self.assertEqual(headers["Content-Range"], expected_range)
                self.assertEqual(int(headers["Content-Length"]), len(expected_body))
                self.assertEqual(body, expected_body)

        status, headers, body = self.request(
            "HEAD", "/clip.mp4", {"Range": "bytes=2-4"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 2-4/10")
        self.assertEqual(headers["Content-Length"], "3")
        self.assertEqual(body, b"")

    def test_unsatisfiable_or_multiple_ranges_return_416(self):
        for value in ("bytes=10-", "bytes=8-2", "bytes=-0", "bytes=0-1,3-4"):
            with self.subTest(value=value):
                status, headers, body = self.request(
                    "GET", "/clip.mp4", {"Range": value}
                )
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], "bytes */10")
                self.assertEqual(headers["Content-Length"], "0")
                self.assertEqual(body, b"")

    def test_root_confinement_and_read_only_methods(self):
        for path in ("/", "/nested/", "/%2e%2e/outside.txt", "/..%5coutside.txt"):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 403)
                self.assertEqual(body, b"")

        outside = self.root.parent / "outside.txt"
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pass  # Windows accounts without symlink privilege cannot create this case.
        else:
            status, _, body = self.request("GET", "/link.txt")
            self.assertEqual(status, 403)
            self.assertEqual(body, b"")

        status, headers, body = self.request(
            "PUT", "/clip.mp4", body=b"overwritten"
        )
        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET, HEAD")
        self.assertEqual(body, b"")
        self.assertEqual((self.root / "clip.mp4").read_bytes(), b"0123456789")


if __name__ == "__main__":
    unittest.main()
