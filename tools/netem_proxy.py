"""A fault-injecting TCP proxy ("netem in Python") placed between devices and the cloud, for tests and benchmarks.

Per connection it can: add latency (+ jitter) to every chunk, cap bandwidth, DROP the connection after a random
number of bytes (mid-request or mid-reply - the reply, i.e. the acknowledgement, can be lost after the cloud already
stored the events), STALL (accept, then never answer), or refuse everything during a BLACKOUT. Settings can change
while it runs (a flapping link). Pure asyncio, one background thread, no admin rights (unlike OS-level netem/clumsy).
"""
from __future__ import annotations

import asyncio
import random
import threading
from dataclasses import dataclass, field


@dataclass
class Faults:
    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    bandwidth_bps: float | None = None        # bytes per second per direction; None = unlimited
    drop_prob: float = 0.0                    # per request / reply: cut the connection inside it
    stall_prob: float = 0.0                   # per request: never forward it (the client times out)
    blackout: bool = False                    # refuse every connection


@dataclass
class Stats:
    connections: int = 0
    dropped: int = 0
    stalled: int = 0
    refused: int = 0
    bytes_up: int = 0
    bytes_down: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class NetemProxy:
    def __init__(self, target_host: str, target_port: int, faults: Faults | None = None, seed: int = 0):
        self.target = (target_host, target_port)
        self.faults = faults or Faults()
        self.stats = Stats()
        self._rng = random.Random(seed)
        self._loop = asyncio.new_event_loop()
        self._server = None
        self.port: int | None = None

    # ---- control ------------------------------------------------------------------------------------------
    def start(self) -> int:
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self._loop)
            self._server = self._loop.run_until_complete(asyncio.start_server(self._handle, "127.0.0.1", 0))
            self.port = self._server.sockets[0].getsockname()[1]
            ready.set()
            self._loop.run_forever()

        threading.Thread(target=run, name="netem", daemon=True).start()
        ready.wait(10)
        return self.port

    def set(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self.faults, k, v)

    def stop(self) -> None:
        async def _stop():
            if self._server:
                self._server.close()
            tasks = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task()]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._loop.stop()
        asyncio.run_coroutine_threadsafe(_stop(), self._loop)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ---- data path ----------------------------------------------------------------------------------------
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Faults are drawn per forwarded CHUNK, not per connection: HTTP clients keep one connection alive for many
        requests, so a per-connection decision would almost never fire. A chunk is one request or one reply (or a
        piece of a large one), so drop_prob / stall_prob act per request and per acknowledgement."""
        self.stats.connections += 1
        if self.faults.blackout:
            self.stats.refused += 1
            writer.close()
            return
        try:
            ur, uw = await asyncio.open_connection(*self.target)
        except OSError:
            writer.close()
            return
        state = {"cut": False, "stalled": False}

        async def pipe(src, dst, up: bool):
            try:
                while not state["cut"]:
                    data = await src.read(16384)
                    if not data:
                        break
                    f = self.faults
                    if f.blackout:                          # the link went down mid-connection
                        state["cut"] = True
                        break
                    if up and not state["stalled"] and self._rng.random() < f.stall_prob:
                        state["stalled"] = True            # from now on nothing is forwarded: the client times out
                        self.stats.stalled += 1
                    if state["stalled"]:
                        continue
                    if f.latency_ms or f.jitter_ms:
                        await asyncio.sleep(max(0.0, (f.latency_ms + self._rng.uniform(-f.jitter_ms, f.jitter_ms)) / 1000))
                    if f.bandwidth_bps:
                        await asyncio.sleep(len(data) / f.bandwidth_bps)
                    if self._rng.random() < f.drop_prob:    # cut inside this request / this reply
                        data = data[:self._rng.randint(0, len(data))]
                        state["cut"] = True
                    dst.write(data)
                    await dst.drain()
                    if up:
                        self.stats.bytes_up += len(data)
                    else:
                        self.stats.bytes_down += len(data)
            except (ConnectionError, OSError):
                pass
            finally:
                if state["cut"]:
                    for w in (writer, uw):
                        w.transport.abort()                # RST: both sides see a broken connection
                else:
                    try:
                        dst.close()
                    except Exception:
                        pass

        await asyncio.gather(pipe(reader, uw, True), pipe(ur, writer, False))
        if state["cut"]:
            self.stats.dropped += 1
