"""浏览器控制插件：通过 Edge CDP（Chrome DevTools Protocol）启动游戏、
读取 JS 状态、发送真实按键。不依赖 Playwright/Selenium。
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlparse

from . import constants as C

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
GAME_URL = "file:///E:/deepseek/test/3d-tetris.html"

# 按键 -> CDP code / Windows 虚拟键码
KEY_MAP = {
    "a": ("KeyA", 65),
    "d": ("KeyD", 68),
    "w": ("KeyW", 87),
    "s": ("KeyS", 83),
    "x": ("KeyX", 88),
    "c": ("KeyC", 67),
    "p": ("KeyP", 80),
    "r": ("KeyR", 82),
    "m": ("KeyM", 77),
    " ": ("Space", 32),
    "ArrowUp": ("ArrowUp", 38),
    "ArrowDown": ("ArrowDown", 40),
    "ArrowLeft": ("ArrowLeft", 37),
    "ArrowRight": ("ArrowRight", 39),
}


class CdpError(Exception):
    pass


class CdpWebSocket:
    """极简 WebSocket 客户端，用于连接 Edge 的 CDP。"""

    def __init__(self, ws_url: str):
        self.sock: socket.socket | None = None
        self._buf = b""
        self._connect(ws_url)

    def _connect(self, ws_url: str):
        u = urlparse(ws_url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "wss" else 80)
        path = u.path or "/"
        if u.query:
            path += "?" + u.query

        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise CdpError("WebSocket 握手失败：连接被关闭")
            resp += chunk
            if len(resp) > 65536:
                raise CdpError("WebSocket 握手响应过大")
        head = resp.split(b"\r\n\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in head and " 101" not in head.split("\r\n")[0]:
            raise CdpError(f"WebSocket 握手失败：{head}")
        # 多余数据可能包含帧头，暂不处理（CDP 握手后通常无多余数据）

    def send_text(self, text: str):
        payload = text.encode("utf-8")
        self._send_frame(0x1, payload)

    def _send_frame(self, opcode: int, payload: bytes):
        n = len(payload)
        header = bytearray([0x80 | opcode])
        mask = os.urandom(4)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_frame(self, timeout: float = 15):
        self.sock.settimeout(timeout)
        h = self._recv_exact(2)
        if len(h) < 2:
            raise CdpError("WebSocket 连接关闭")
        b0, b1 = h[0], h[1]
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        masked = bool(b1 & 0x80)
        if length == 126:
            length = struct.unpack("!H", self._recv_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._recv_exact(8))[0]
        mask = self._recv_exact(4) if masked else None
        payload = self._recv_exact(length)
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def _recv_exact(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                break
            self._buf += chunk
        if len(self._buf) < n:
            out, self._buf = self._buf, b""
            return out
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


class CdpClient:
    def __init__(self, ws_url: str):
        self.ws = CdpWebSocket(ws_url)
        self._id = 0

    def call(self, method: str, params: dict | None = None, timeout: float = 15):
        self._id += 1
        cid = self._id
        msg = {"id": cid, "method": method, "params": params or {}}
        self.ws.send_text(json.dumps(msg))
        while True:
            opcode, payload = self.ws.recv_frame(timeout)
            if opcode == 0x8:  # close
                raise CdpError("CDP 连接被关闭")
            if opcode == 0x9:  # ping -> pong
                self.ws._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode != 0x1:
                continue
            try:
                data = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("id") == cid:
                if "error" in data:
                    err = data["error"]
                    raise CdpError(f"CDP {method} 错误: {err.get('message', err)}")
                return data.get("result", {})

    def close(self):
        self.ws.close()


class BrowserGameController:
    """负责启动 Edge、连接页面、读取状态、发送按键。"""

    def __init__(self, port: int = 9222, game_url: str = GAME_URL):
        self.port = port
        self.game_url = game_url
        self.proc: subprocess.Popen | None = None
        self.user_data_dir: str | None = None
        self.client: CdpClient | None = None
        self.page_ws_url: str | None = None

    # ---------- 启动/连接 ----------
    def connect_existing(self, timeout: float = 10):
        """连接一个已经以 --remote-debugging-port 启动的 Edge。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json", timeout=2
                ) as resp:
                    targets = json.loads(resp.read().decode("utf-8"))
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    game = [t for t in pages if "3d-tetris" in t.get("url", "")]
                    target = game[0] if game else pages[0]
                    self.page_ws_url = target.get("webSocketDebuggerUrl")
                    if self.page_ws_url:
                        self.client = CdpClient(self.page_ws_url)
                        self.client.call("Runtime.enable")
                        self.client.call("Page.enable")
                        return True
            except Exception:
                time.sleep(0.25)
        return False

    def start(self):
        profile_root = os.path.join(os.path.dirname(__file__), "edge_profiles")
        os.makedirs(profile_root, exist_ok=True)
        self.user_data_dir = os.path.join(profile_root, f"profile_{int(time.time() * 1000)}")
        cmd = [
            EDGE_PATH,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.user_data_dir}",
            "--new-window",
            "--window-size=1400,1000",
            "--no-first-run",
            "--no-default-browser-check",
            self.game_url,
        ]
        self.proc = subprocess.Popen(cmd)
        self._wait_and_connect()

    def _wait_and_connect(self, timeout: float = 20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json", timeout=2
                ) as resp:
                    targets = json.loads(resp.read().decode("utf-8"))
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    # 优先选游戏页面
                    game = [t for t in pages if "3d-tetris" in t.get("url", "")]
                    target = game[0] if game else pages[0]
                    self.page_ws_url = target.get("webSocketDebuggerUrl")
                    if self.page_ws_url:
                        self.client = CdpClient(self.page_ws_url)
                        self.client.call("Runtime.enable")
                        self.client.call("Page.enable")
                        return
            except Exception:
                time.sleep(0.25)
        raise CdpError("无法连接到 Edge 调试端口")

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        self.proc = None

    # ---------- JS 状态读取 ----------
    def eval_js(self, expression: str):
        if not self.client:
            raise CdpError("浏览器未连接")
        res = self.client.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return res.get("result", {}).get("value")

    def read_state(self) -> dict:
        expr = """
(() => {
  const b = [];
  for (let x = 0; x < 7; x++) {
    b[x] = [];
    for (let y = 0; y < 14; y++) {
      b[x][y] = [];
      for (let z = 0; z < 7; z++) b[x][y][z] = board[x][y][z] ? 1 : 0;
    }
  }
  return {
    status: state.status,
    score: state.score,
    level: state.level,
    lines: state.lines,
    planes: state.planes,
    combo: state.combo,
    canHold: state.canHold,
    piece: piece ? {type: piece.type, x: piece.x, y: piece.y, z: piece.z, cells: piece.cells} : null,
    hold: hold || null,
    queue: queue.slice(0, 6),
    board: b
  };
})()
"""
        return self.eval_js(expr)

    # ---------- 按键 ----------
    def press_key(self, key: str, delay: float = 0.02):
        code, vk = KEY_MAP.get(key, (key, 0))
        down = {
            "type": "keyDown",
            "key": key,
            "code": code,
            "windowsVirtualKeyCode": vk,
            "nativeVirtualKeyCode": vk,
        }
        up = {
            "type": "keyUp",
            "key": key,
            "code": code,
            "windowsVirtualKeyCode": vk,
            "nativeVirtualKeyCode": vk,
        }
        self.client.call("Input.dispatchKeyEvent", down)
        if delay > 0:
            time.sleep(delay)
        self.client.call("Input.dispatchKeyEvent", up)
        if delay > 0:
            time.sleep(delay * 0.5)

    def pause_game(self):
        """确保进入暂停。"""
        st = self.read_state()
        if st.get("status") == "playing":
            self.press_key(C.KEY_PAUSE, delay=0.02)
            time.sleep(0.05)

    def resume_game(self):
        st = self.read_state()
        if st.get("status") == "paused":
            self.press_key(C.KEY_PAUSE, delay=0.02)
            time.sleep(0.05)

    def start_game(self, timeout: float = 10):
        """开始一局新游戏（从 idle 或 over 状态）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.read_state()
            if st.get("status") == "playing":
                return True
            if st.get("status") in ("idle", "over"):
                self.press_key(" ", delay=0.05)
            time.sleep(0.1)
        return False
