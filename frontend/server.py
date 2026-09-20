"""Sert le frontend statique PuLID sur un port distinct du backend."""

from __future__ import annotations

import argparse
from functools import partial
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Sequence
from urllib.parse import SplitResult, urlsplit


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8888
DEFAULT_BACKEND_URL = "http://127.0.0.1:12693"
PROXY_TIMEOUT_SECONDS = 5 * 60


class FrontendRequestHandler(SimpleHTTPRequestHandler):
    """Sert les fichiers sans cache afin de refléter immédiatement les changements."""

    backend: SplitResult

    def __init__(
        self,
        *args: object,
        backend: SplitResult,
        **kwargs: object,
    ) -> None:
        self.backend = backend
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/models/krea2":
            self._proxy("/models/krea2")
            return
        if self.path == "/api/capabilities":
            self._proxy("/capabilities")
            return
        if self.path == "/api/models":
            self._proxy("/models")
            return
        if self.path == "/frontend-config.json":
            self._send_json(
                200,
                {"backend_url": self.backend.geturl().rstrip("/")},
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/generate/krea2":
            self._proxy("/generate/krea2")
            return
        if self.path == "/api/generate":
            self._proxy("/generate")
            return
        self._send_json(404, {"detail": {"message": "Route frontend inconnue."}})

    def _proxy(self, backend_path: str) -> None:
        connection_type = HTTPSConnection if self.backend.scheme == "https" else HTTPConnection
        connection = connection_type(
            self.backend.hostname,
            self.backend.port,
            timeout=PROXY_TIMEOUT_SECONDS,
        )
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {}
        if content_type := self.headers.get("Content-Type"):
            headers["Content-Type"] = content_type

        try:
            connection.request(self.command, backend_path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for header in (
                "Content-Type",
                "Content-Disposition",
                "X-Generation-Seed",
                "X-SDXL-Model",
                "X-Generation-Model",
                "X-Krea2-Model",
                "X-Krea2-Text-Encoder",
                "X-Sampling-Method",
                "X-Sigma-Schedule",
            ):
                if value := response.getheader(header):
                    self.send_header(header, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
        except (HTTPException, OSError) as exc:
            self._send_json(
                502,
                {
                    "detail": {
                        "message": (
                            "Backend PuLID inaccessible à "
                            f"{self.backend.geturl().rstrip('/')}: {exc}"
                        )
                    }
                },
            )
        finally:
            connection.close()

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def frontend_directory() -> Path:
    """Retourne la racine statique indépendamment du répertoire courant."""

    return Path(__file__).resolve().parent


def build_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    backend_url: str = DEFAULT_BACKEND_URL,
    directory: Path | None = None,
) -> ThreadingHTTPServer:
    """Construit le serveur HTTP du frontend."""

    static_directory = (directory or frontend_directory()).resolve()
    if not static_directory.is_dir():
        raise FileNotFoundError(f"Dossier frontend introuvable : {static_directory}")
    backend = urlsplit(backend_url)
    if backend.scheme not in {"http", "https"} or not backend.hostname:
        raise ValueError(
            "L'adresse du backend doit être une URL HTTP(S) complète, "
            f"reçu : {backend_url!r}."
        )
    if backend.path not in {"", "/"} or backend.query or backend.fragment:
        raise ValueError(
            "L'adresse du backend ne doit contenir ni chemin, ni query string, "
            f"ni fragment : {backend_url!r}."
        )
    handler = partial(
        FrontendRequestHandler,
        backend=backend,
        directory=str(static_directory),
    )
    return ThreadingHTTPServer((host, port), handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serveur local du frontend de génération PuLID."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        server = build_server(args.host, args.port, backend_url=args.backend_url)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Impossible de démarrer le frontend : {exc}")
        return 1

    display_host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
    print(f"Frontend PuLID : http://{display_host}:{server.server_port}")
    print(f"Backend attendu : {args.backend_url.rstrip('/')}")
    print("Arrêt : Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du frontend.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
