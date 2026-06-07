#!/usr/bin/env python3
"""
music_player_node.py — ROS 2 node that accepts song queries/URLs on /chosen_song,
resolves them with yt-dlp, and streams audio directly through ffmpeg → aplay.
No files are ever downloaded to disk.

Published:  /music/status   (std_msgs/String, JSON)
Subscribed: /chosen_song    (std_msgs/String)
Subscribed: /music/volume   (std_msgs/String)  — plain float string, e.g. "3.5"
"""

import json
import queue
import shutil
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yt_dlp

import subprocess

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
ALSA_DEVICE        = "plughw:2,0"
YTDLP_FORMAT       = "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio"
SAMPLE_RATE        = 44100
CHANNELS           = 2
DEFAULT_VOLUME     = 5.0               # ffmpeg dynaudnorm volume multiplier
STATUS_INTERVAL    = 0.05              # seconds between status queue drains
STOP_POLL_MS       = 0.2              # seconds between stop-event polls
PROC_KILL_TIMEOUT  = 3.0              # seconds before escalating terminate → kill
PROGRESS_INTERVAL  = 1.0              # seconds between current_time ticks


# ---------------------------------------------------------------------------
# YTDLPResolver
# ---------------------------------------------------------------------------
class YTDLPResolver:
    def resolve(self, query: str) -> dict:
        search_query = self._to_search(query)
        ydl_opts = {
            "format": YTDLP_FORMAT,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)

        if "entries" in info:
            entries = list(info["entries"])
            if not entries:
                raise RuntimeError("yt-dlp returned no results.")
            info = entries[0]

        if not info.get("url"):
            raise RuntimeError("yt-dlp did not return a stream URL.")

        return {
            "title":       info.get("title", "Unknown"),
            "duration":    float(info.get("duration") or 0.0),
            "webpage_url": info.get("webpage_url", ""),
            # Build a safe filename from the title
            "filename":    _safe_filename(info.get("title", "Unknown")),
        }

    @staticmethod
    def _to_search(query: str) -> str:
        if query.startswith("http://") or query.startswith("https://"):
            return query
        return f"ytsearch1:{query}"


def _safe_filename(title: str) -> str:
    """Produce a readable display name (no path sanitisation needed — display only)."""
    return title.strip()


# ---------------------------------------------------------------------------
# StatusPublisherThread
# ---------------------------------------------------------------------------
class StatusPublisherThread(threading.Thread):
    def __init__(self, ros_publisher):
        super().__init__(daemon=True, name="StatusPublisher")
        self._publisher = ros_publisher
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

    def enqueue(self, payload: dict) -> None:
        self._queue.put_nowait(payload)

    def run(self) -> None:
        while not self._stop_event.is_set():
            while not self._queue.empty():
                try:
                    payload = self._queue.get_nowait()
                    msg = String()
                    msg.data = json.dumps(payload)
                    self._publisher.publish(msg)
                except queue.Empty:
                    break
                except Exception:
                    pass
            time.sleep(STATUS_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# MusicPlayerNode
# ---------------------------------------------------------------------------
class MusicPlayerNode(Node):

    def __init__(self):
        super().__init__("music_player_node")

        self._publisher  = self.create_publisher(String, "/music/status", 10)
        self._subscriber = self.create_subscription(
            String, "/chosen_song", self._on_chosen_song, 10
        )
        self._vol_subscriber = self.create_subscription(
            String, "/music/volume", self._on_volume, 10
        )

        self._resolver   = YTDLPResolver()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._ytdlp_proc:  subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._aplay_proc:  subprocess.Popen | None = None
        self._state_lock = threading.Lock()

        # Volume (thread-safe via lock)
        self._volume_lock = threading.Lock()
        self._volume = DEFAULT_VOLUME

        # Playback progress state (set when playing, cleared on stop)
        self._play_start_time: float | None = None
        self._play_title: str = ""
        self._play_duration: float = 0.0
        self._play_filename: str = ""

        # Progress ticker
        self._ticker_stop = threading.Event()
        self._ticker_thread: threading.Thread | None = None

        self._status_thread = StatusPublisherThread(self._publisher)
        self._status_thread.start()

        self.get_logger().info("music_player_node started.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info("Shutting down…")
        self._stop_everything()
        self._status_thread.stop()
        self._status_thread.join(timeout=2.0)
        super().destroy_node()

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _on_chosen_song(self, msg: String) -> None:
        query = msg.data.strip()
        self.get_logger().info(f"Received /chosen_song: '{query}'")
        self._stop_everything()
        if not query:
            self._publish({"state": "finished"})
            return
        self._stop_event.clear()
        t = threading.Thread(
            target=self._worker_main, args=(query,), daemon=True, name="MusicWorker"
        )
        self._worker_thread = t
        t.start()

    def _on_volume(self, msg: String) -> None:
        try:
            val = float(msg.data.strip())
            val = max(0.0, min(val, 10.0))

            percent = int(val * 10)  # 0-10 -> 0-100%

            result = subprocess.run(
                [
                    "amixer",
                    "-c",
                    "2",
                    "sset",
                    "Speaker",
                    f"{percent}%"
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                self.get_logger().warning(
                    f"amixer failed: {result.stderr}"
                )
                return

            self.get_logger().info(
                f"Volume changed to {percent}%"
            )

        except ValueError:
            self.get_logger().warning(
                f"Invalid volume value: '{msg.data}'"
            )
    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------
    def _stop_everything(self) -> None:
        # Stop progress ticker
        self._ticker_stop.set()
        if self._ticker_thread and self._ticker_thread.is_alive():
            self._ticker_thread.join(timeout=1.0)
        self._ticker_thread = None
        self._ticker_stop.clear()

        with self._state_lock:
            self._stop_event.set()
            for attr in ("_ytdlp_proc", "_ffmpeg_proc", "_aplay_proc"):
                proc: subprocess.Popen | None = getattr(self, attr)
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass

        deadline = time.monotonic() + PROC_KILL_TIMEOUT
        for attr in ("_ytdlp_proc", "_ffmpeg_proc", "_aplay_proc"):
            proc: subprocess.Popen | None = getattr(self, attr)
            if proc:
                remaining = deadline - time.monotonic()
                try:
                    proc.wait(timeout=max(0.1, remaining))
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except OSError:
                        pass

        with self._state_lock:
            self._ytdlp_proc  = None
            self._ffmpeg_proc = None
            self._aplay_proc  = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=PROC_KILL_TIMEOUT + 1.0)
        self._worker_thread = None

        # Clear progress state
        self._play_start_time = None

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _worker_main(self, query: str) -> None:
        try:
            if self._stop_event.is_set():
                return

            self._publish({"state": "searching"})
            self.get_logger().info(f"Resolving: {query}")

            try:
                result = self._resolver.resolve(query)
            except Exception as exc:
                self._publish({"state": "error", "message": f"Search failed: {exc}"})
                self.get_logger().error(f"resolve error: {exc}")
                return

            if self._stop_event.is_set():
                return

            title       = result["title"]
            duration    = result["duration"]
            webpage_url = result["webpage_url"]
            filename    = result["filename"]

            self.get_logger().info(f"Streaming: {title}")

            # Store playback metadata for the progress ticker
            self._play_title    = title
            self._play_duration = duration
            self._play_filename = filename
            self._play_start_time = time.monotonic()

            self._publish({
                "state":        "playing",
                "title":        title,
                "duration":     duration,
                "filename":     filename,
                "current_time": 0.0,
            })

            # Start progress ticker
            self._ticker_stop.clear()
            self._ticker_thread = threading.Thread(
                target=self._progress_ticker, daemon=True, name="ProgressTicker"
            )
            self._ticker_thread.start()

            self._stream_audio(webpage_url)

        except Exception as exc:
            self._publish({"state": "error", "message": str(exc)})
            self.get_logger().error(f"Worker error: {exc}")

    # ------------------------------------------------------------------
    # Progress ticker — publishes current_time every second while playing
    # ------------------------------------------------------------------
    def _progress_ticker(self) -> None:
        while not self._ticker_stop.is_set():
            time.sleep(PROGRESS_INTERVAL)
            if self._ticker_stop.is_set():
                break
            if self._play_start_time is None:
                break
            elapsed = time.monotonic() - self._play_start_time
            self._publish({
                "state":        "playing",
                "title":        self._play_title,
                "duration":     self._play_duration,
                "filename":     self._play_filename,
                "current_time": round(elapsed, 1),
            })

    # ------------------------------------------------------------------
    # Streaming pipeline: yt-dlp → ffmpeg (with volume filter) → aplay
    # ------------------------------------------------------------------
    def _stream_audio(self, webpage_url: str) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH.")
        if shutil.which("aplay") is None:
            raise RuntimeError("aplay not found on PATH.")
        if shutil.which("yt-dlp") is None:
            raise RuntimeError("yt-dlp CLI not found on PATH.")

        with self._volume_lock:
            vol = self._volume

        ytdlp_cmd = [
            "yt-dlp",
            "--format", YTDLP_FORMAT,
            "--no-playlist",
            "--quiet",
            "-o", "-",
            webpage_url,
        ]
        ffmpeg_cmd = [
            "ffmpeg",
            "-loglevel", "warning",
            "-i", "pipe:0",
            "-af", f"dynaudnorm=g=12:f=250",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            "pipe:1",
        ]
        aplay_cmd = [
            "aplay",
            "-D", ALSA_DEVICE,
            "-f", "S16_LE",
            "-r", str(SAMPLE_RATE),
            "-c", str(CHANNELS),
        ]

        ytdlp_proc = ffmpeg_proc = aplay_proc = None
        try:
            ytdlp_proc = subprocess.Popen(
                ytdlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd, stdin=ytdlp_proc.stdout,
                stdout=subprocess.PIPE, stderr=None
            )
            aplay_proc = subprocess.Popen(
                aplay_cmd, stdin=ffmpeg_proc.stdout,
                stdout=subprocess.DEVNULL, stderr=None
            )
            ytdlp_proc.stdout.close()
            ffmpeg_proc.stdout.close()

            with self._state_lock:
                self._ytdlp_proc  = ytdlp_proc
                self._ffmpeg_proc = ffmpeg_proc
                self._aplay_proc  = aplay_proc

            while True:
                if self._stop_event.is_set():
                    return
                if aplay_proc.poll() is not None:
                    rc = ffmpeg_proc.poll()
                    if rc is not None and rc != 0:
                        raise RuntimeError(f"ffmpeg exited with code {rc}")
                    # Stop ticker before publishing finished
                    self._ticker_stop.set()
                    self._publish({"state": "finished"})
                    self.get_logger().info("Playback finished.")
                    return
                time.sleep(STOP_POLL_MS)

        finally:
            for proc in (ytdlp_proc, ffmpeg_proc, aplay_proc):
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=PROC_KILL_TIMEOUT)
                    except (OSError, subprocess.TimeoutExpired):
                        try:
                            proc.kill()
                        except OSError:
                            pass
            with self._state_lock:
                self._ytdlp_proc  = None
                self._ffmpeg_proc = None
                self._aplay_proc  = None

    # ------------------------------------------------------------------
    def _publish(self, payload: dict) -> None:
        self._status_thread.enqueue(payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = MusicPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()