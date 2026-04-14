from __future__ import annotations

import json
import socketserver
import sys
import threading
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Final

from agent_tools.queue_db import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PLAYING,
    STATUS_STOPPED,
    QueueItem,
    clear_queued_items,
    connect,
    get_current_item,
    get_next_queued_item,
    list_history_items,
    list_queue_items,
    normalize_inflight_items,
    update_status,
)
from agent_tools.runtime import CONTROLLER_HOST, CONTROLLER_PORT

SOCKET_TIMEOUT_SECONDS: Final = 0.5


@dataclass(frozen=True)
class ControllerCommand:
    action: str


def run_ui(*, hidden: bool) -> int:
    try:
        from PySide6.QtCore import Qt, QTimer, QUrl
        from PySide6.QtGui import QAction, QIcon, QKeySequence, QShortcut
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QMainWindow,
            QMenu,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QStyle,
            QSystemTrayIcon,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is required for the UI. Install with: pip install ai-nd-co-agent-tools[ui]"
        ) from exc

    class CommandServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    command_queue: SimpleQueue[ControllerCommand] = SimpleQueue()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            raw = self.rfile.readline().decode("utf-8", "replace").strip()
            if not raw:
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return
            action = payload.get("action")
            if isinstance(action, str):
                command_queue.put(ControllerCommand(action=action))

    try:
        server = CommandServer((CONTROLLER_HOST, CONTROLLER_PORT), Handler)
    except OSError as exc:
        raise RuntimeError("AgentTools UI controller is already running.") from exc

    threading.Thread(target=server.serve_forever, daemon=True).start()

    class ControllerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.conn = connect()
            normalize_inflight_items(self.conn)
            self.current_item_id: str | None = None
            self.current_duration_ms = 0

            self.audio_output = QAudioOutput()
            self.player = QMediaPlayer()
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.mediaStatusChanged.connect(self._on_media_status_changed)
            self.player.errorOccurred.connect(self._on_error_occurred)

            self.setWindowTitle("AgentTools Audio Queue")
            self.resize(560, 520)
            self._build_ui(
                QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MediaVolume)
            )
            self._setup_shortcuts()
            self._setup_tray()

            self.command_timer = QTimer(self)
            self.command_timer.timeout.connect(self._drain_commands)
            self.command_timer.start(150)

            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self.refresh_views)
            self.refresh_timer.start(1000)

            self.refresh_views()
            self._maybe_autoplay()

        def _build_ui(self, icon: QIcon) -> None:
            central = QWidget()
            layout = QVBoxLayout(central)

            self.now_label = QLabel("Now Playing: none")
            self.now_label.setWordWrap(True)
            self.progress = QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setValue(0)

            self.autoplay_checkbox = QCheckBox("Autoplay next item")
            self.autoplay_checkbox.setChecked(True)

            controls = QHBoxLayout()
            self.play_pause_button = QPushButton("Play/Pause")
            self.stop_button = QPushButton("Stop")
            self.replay_button = QPushButton("Replay")
            self.next_button = QPushButton("Next")
            self.clear_queue_button = QPushButton("Clear Queue")

            for button in (
                self.play_pause_button,
                self.stop_button,
                self.replay_button,
                self.next_button,
                self.clear_queue_button,
            ):
                controls.addWidget(button)

            self.queue_list = QListWidget()
            self.history_list = QListWidget()

            layout.addWidget(self.now_label)
            layout.addWidget(self.progress)
            layout.addWidget(self.autoplay_checkbox)
            layout.addLayout(controls)
            layout.addWidget(QLabel("Queue"))
            layout.addWidget(self.queue_list)
            layout.addWidget(QLabel("History"))
            layout.addWidget(self.history_list)

            self.setCentralWidget(central)
            self.setWindowIcon(icon)

            self.play_pause_button.clicked.connect(self.play_pause)
            self.stop_button.clicked.connect(self.stop_current)
            self.replay_button.clicked.connect(self.replay_current)
            self.next_button.clicked.connect(self.skip_next)
            self.clear_queue_button.clicked.connect(self.clear_queue)

        def _setup_shortcuts(self) -> None:
            self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
            self.space_shortcut.activated.connect(self.play_pause)

            self.escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
            self.escape_shortcut.activated.connect(self.stop_current)

            self.replay_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
            self.replay_shortcut.activated.connect(self.replay_current)

            self.next_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
            self.next_shortcut.activated.connect(self.skip_next)

        def _setup_tray(self) -> None:
            icon = self.windowIcon()
            tray_menu = QMenu()

            show_action = QAction("Show")
            show_action.triggered.connect(self.show_and_focus)
            tray_menu.addAction(show_action)

            play_pause_action = QAction("Play/Pause")
            play_pause_action.triggered.connect(self.play_pause)
            tray_menu.addAction(play_pause_action)

            stop_action = QAction("Stop")
            stop_action.triggered.connect(self.stop_current)
            tray_menu.addAction(stop_action)

            next_action = QAction("Next")
            next_action.triggered.connect(self.skip_next)
            tray_menu.addAction(next_action)

            tray_menu.addSeparator()

            quit_action = QAction("Quit")
            quit_action.triggered.connect(self.quit_app)
            tray_menu.addAction(quit_action)

            self.tray = QSystemTrayIcon(icon, self)
            self.tray.setToolTip("AgentTools Audio Queue")
            self.tray.setContextMenu(tray_menu)
            self.tray.activated.connect(self._on_tray_activated)
            self.tray.show()

        def _on_tray_activated(self, reason: object) -> None:
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                self.show_and_focus()

        def closeEvent(self, event) -> None:
            event.ignore()
            self.hide()
            self.tray.showMessage(
                "AgentTools",
                "Audio queue is still running in the tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

        def quit_app(self) -> None:
            self.player.stop()
            server.shutdown()
            server.server_close()
            self.conn.close()
            QApplication.quit()

        def show_and_focus(self) -> None:
            self.show()
            self.raise_()
            self.activateWindow()

        def refresh_views(self) -> None:
            current = get_current_item(self.conn)
            queue_items = list_queue_items(self.conn)
            history_items = list_history_items(self.conn, limit=25)

            if current is None:
                self.now_label.setText("Now Playing: none")
                if self.player.duration() <= 0:
                    self.progress.setValue(0)
            else:
                label = current.source_label or "unknown source"
                preview = _preview_text(current.tts_text)
                self.now_label.setText(f"Now Playing [{current.status}] ({label}): {preview}")
                self.current_item_id = current.item_id

            self.queue_list.clear()
            for item in queue_items:
                self.queue_list.addItem(_format_queue_item(item))

            self.history_list.clear()
            for item in history_items:
                self.history_list.addItem(_format_queue_item(item))

        def _drain_commands(self) -> None:
            saw_refresh = False
            while True:
                try:
                    cmd = command_queue.get_nowait()
                except Empty:
                    break
                if cmd.action == "show":
                    self.show_and_focus()
                    saw_refresh = True
                elif cmd.action == "refresh":
                    saw_refresh = True
                elif cmd.action == "shutdown":
                    self.quit_app()
                    return
            if saw_refresh:
                self.refresh_views()
                self._maybe_autoplay()

        def _maybe_autoplay(self) -> None:
            if not self.autoplay_checkbox.isChecked():
                return
            current = get_current_item(self.conn)
            if current is not None:
                return
            next_item = get_next_queued_item(self.conn)
            if next_item is None:
                return
            self._play_item(next_item)

        def _play_item(self, item: QueueItem) -> None:
            self.current_item_id = item.item_id
            self.current_duration_ms = item.duration_ms
            update_status(self.conn, item.item_id, STATUS_PLAYING)
            self.player.setSource(QUrl.fromLocalFile(item.audio_path))
            self.player.play()
            self.refresh_views()

        def play_pause(self) -> None:
            state = self.player.playbackState()
            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
                if self.current_item_id is not None:
                    update_status(self.conn, self.current_item_id, STATUS_PAUSED)
            elif state == QMediaPlayer.PlaybackState.PausedState:
                self.player.play()
                if self.current_item_id is not None:
                    update_status(self.conn, self.current_item_id, STATUS_PLAYING)
            else:
                current = get_current_item(self.conn)
                if current is not None:
                    self._play_item(current)
                else:
                    self._maybe_autoplay()
            self.refresh_views()

        def stop_current(self) -> None:
            if self.current_item_id is None:
                return
            self.player.stop()
            update_status(self.conn, self.current_item_id, STATUS_STOPPED)
            self.current_item_id = None
            self.progress.setValue(0)
            self.refresh_views()

        def replay_current(self) -> None:
            current = get_current_item(self.conn)
            if current is None:
                return
            self.player.stop()
            self._play_item(current)

        def skip_next(self) -> None:
            if self.current_item_id is not None:
                self.player.stop()
                update_status(self.conn, self.current_item_id, STATUS_STOPPED)
                self.current_item_id = None
            next_item = get_next_queued_item(self.conn)
            if next_item is not None:
                self._play_item(next_item)
            else:
                self.refresh_views()

        def clear_queue(self) -> None:
            clear_queued_items(self.conn)
            self.refresh_views()

        def _on_position_changed(self, position_ms: int) -> None:
            if self.current_duration_ms <= 0:
                self.progress.setValue(0)
                return
            value = int(min(1000, max(0, (position_ms / self.current_duration_ms) * 1000)))
            self.progress.setValue(value)

        def _on_duration_changed(self, duration_ms: int) -> None:
            if duration_ms > 0:
                self.current_duration_ms = duration_ms

        def _on_playback_state_changed(self, state) -> None:
            if self.current_item_id is None:
                return
            if state == QMediaPlayer.PlaybackState.PlayingState:
                update_status(self.conn, self.current_item_id, STATUS_PLAYING)
            elif state == QMediaPlayer.PlaybackState.PausedState:
                update_status(self.conn, self.current_item_id, STATUS_PAUSED)
            self.refresh_views()

        def _on_media_status_changed(self, status) -> None:
            if (
                status == QMediaPlayer.MediaStatus.EndOfMedia
                and self.current_item_id is not None
            ):
                update_status(self.conn, self.current_item_id, STATUS_COMPLETED)
                self.current_item_id = None
                self.progress.setValue(0)
                self.refresh_views()
                self._maybe_autoplay()

        def _on_error_occurred(self, *_args) -> None:
            if self.current_item_id is not None:
                message = self.player.errorString() or "Unknown playback error."
                update_status(self.conn, self.current_item_id, STATUS_FAILED, error_message=message)
                self.current_item_id = None
                self.progress.setValue(0)
                self.refresh_views()
                QMessageBox.warning(self, "Playback error", message)

    app = QApplication.instance() or QApplication(sys.argv)
    window = ControllerWindow()
    if not hidden:
        window.show_and_focus()
    return int(app.exec())


def _preview_text(text: str, limit: int = 96) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _format_queue_item(item: QueueItem) -> str:
    label = item.source_label or "unknown"
    preview = _preview_text(item.tts_text, limit=72)
    return f"[{item.status}] {label}: {preview}"
