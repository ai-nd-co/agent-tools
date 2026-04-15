from __future__ import annotations

import json
import socketserver
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, SimpleQueue

from agent_tools.queue_db import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PLAYING,
    STATUS_QUEUED,
    STATUS_STOPPED,
    QueueItem,
    connect,
    get_current_item,
    get_item_by_item_id,
    get_next_queued_item,
    list_all_items,
    normalize_inflight_items,
    update_status,
)
from agent_tools.runtime import CONTROLLER_HOST, CONTROLLER_PORT


@dataclass(frozen=True)
class ControllerCommand:
    action: str
    progress_id: str | None = None
    source_label: str | None = None
    preview_text: str | None = None
    detail_text: str | None = None
    stage: str | None = None


@dataclass(frozen=True)
class ProcessingItem:
    progress_id: str
    source_label: str | None
    preview_text: str
    detail_text: str
    stage: str
    order: int


@dataclass(frozen=True)
class FeedEntry:
    kind: str
    key: str
    queue_item: QueueItem | None = None
    processing_item: ProcessingItem | None = None


FEED_ROW_HEIGHT = 58
FEED_PREVIEW_LIMIT = 96
PLAYBACK_RATE_MIN = 0.5
PLAYBACK_RATE_MAX = 2.0
PLAYBACK_RATE_STEP = 0.1


def interrupted_status_for_switch(status: str) -> str:
    return STATUS_QUEUED if status == STATUS_QUEUED else status


def processing_stage_label(stage: str | None) -> str:
    text = (stage or "Processing audio").strip()
    return text or "Processing audio"


def restored_scroll_value(*, old_value: int, old_max: int, new_max: int) -> int:
    if old_value <= 0:
        return 0
    if old_value >= old_max:
        return new_max
    distance_from_bottom = max(0, old_max - old_value)
    return max(0, new_max - distance_from_bottom)


def clamp_playback_rate(rate: float) -> float:
    return max(PLAYBACK_RATE_MIN, min(PLAYBACK_RATE_MAX, round(rate, 2)))


def playback_rate_label(rate: float) -> str:
    return f"{rate:.1f}x"


def run_ui(*, hidden: bool) -> int:
    try:
        from PySide6.QtCore import QSize, Qt, QTimer, QUrl
        from PySide6.QtGui import QAction, QCursor, QIcon, QKeySequence, QShortcut
        from PySide6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
        from PySide6.QtWidgets import (
            QApplication,
            QFrame,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QSlider,
            QStyle,
            QSystemTrayIcon,
            QToolButton,
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
            if not isinstance(payload, dict):
                return
            action = payload.get("action")
            if isinstance(action, str):
                command_queue.put(
                    ControllerCommand(
                        action=action,
                        progress_id=_coerce_optional_str(payload.get("progress_id")),
                        source_label=_coerce_optional_str(payload.get("source_label")),
                        preview_text=_coerce_optional_str(payload.get("preview_text")),
                        detail_text=_coerce_optional_str(payload.get("detail_text")),
                        stage=_coerce_optional_str(payload.get("stage")),
                    )
                )

    try:
        server = CommandServer((CONTROLLER_HOST, CONTROLLER_PORT), Handler)
    except OSError as exc:
        raise RuntimeError("AgentTools UI controller is already running.") from exc

    threading.Thread(target=server.serve_forever, daemon=True).start()

    class AudioFeedRow(QFrame):
        def __init__(
            self,
            *,
            item: QueueItem,
            is_current: bool,
            is_playing: bool,
            play_icon: QIcon,
            pause_icon: QIcon,
            on_activate: Callable[[str], None],
        ) -> None:
            super().__init__()
            self.setObjectName("audioFeedRow")
            self._on_activate = on_activate
            self._item_id = item.item_id
            self.setFixedHeight(FEED_ROW_HEIGHT)

            root = QHBoxLayout(self)
            root.setContentsMargins(10, 6, 10, 6)
            root.setSpacing(10)

            self.play_button = QToolButton(self)
            self.play_button.setFixedSize(36, 36)
            self.play_button.setIconSize(QSize(18, 18))
            self.play_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.play_button.setIcon(pause_icon if is_current and is_playing else play_icon)
            self.play_button.clicked.connect(self._activate)

            button_style = (
                "QToolButton {"
                "border-radius: 18px;"
                "border: 1px solid rgba(255, 255, 255, 70);"
                "background: rgba(255, 255, 255, 22);"
                "padding: 0px;"
                "}"
                "QToolButton:hover { background: rgba(255, 255, 255, 36); }"
            )
            if is_current:
                button_style = (
                    "QToolButton {"
                    "border-radius: 18px;"
                    "border: 1px solid rgba(120, 180, 255, 150);"
                    "background: rgba(120, 180, 255, 42);"
                    "padding: 0px;"
                    "}"
                    "QToolButton:hover { background: rgba(120, 180, 255, 58); }"
                )
            self.play_button.setStyleSheet(button_style)

            content = QVBoxLayout()
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(0)

            preview_label = QLabel(_preview_text(item.tts_text, limit=FEED_PREVIEW_LIMIT), self)
            preview_label.setWordWrap(False)
            preview_label.setStyleSheet("font-size: 13px;")
            preview_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            content.addWidget(preview_label)

            root.addWidget(self.play_button, alignment=Qt.AlignmentFlag.AlignVCenter)
            root.addLayout(content, stretch=1)

            background = "transparent"
            border = "rgba(255, 255, 255, 18)"
            if item.status == STATUS_QUEUED and not is_current:
                background = "rgba(120, 180, 255, 16)"
            if is_current:
                background = "rgba(120, 180, 255, 24)"
                border = "rgba(120, 180, 255, 70)"

            self.setStyleSheet(
                "#audioFeedRow {"
                f"background: {background};"
                f"border: 1px solid {border};"
                "border-radius: 12px;"
                "}"
            )

        def _activate(self) -> None:
            self._on_activate(self._item_id)

    class ProcessingFeedRow(QFrame):
        def __init__(
            self,
            *,
            item: ProcessingItem,
        ) -> None:
            super().__init__()
            self.setObjectName("processingFeedRow")
            self.setFixedHeight(FEED_ROW_HEIGHT)

            root = QHBoxLayout(self)
            root.setContentsMargins(10, 6, 10, 6)
            root.setSpacing(10)

            badge = QToolButton(self)
            badge.setFixedSize(36, 36)
            badge.setEnabled(False)
            badge.setText("...")
            badge.setStyleSheet(
                "QToolButton {"
                "border-radius: 18px;"
                "border: 1px solid rgba(255, 215, 120, 120);"
                "background: rgba(255, 215, 120, 28);"
                "color: rgba(255, 255, 255, 210);"
                "font-size: 16px;"
                "font-weight: 600;"
                "padding: 0px;"
                "}"
            )

            content = QVBoxLayout()
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(0)

            preview_text = _preview_text(item.preview_text, limit=FEED_PREVIEW_LIMIT)
            preview_label = QLabel(
                f"{processing_stage_label(item.stage)}: {preview_text}",
                self,
            )
            preview_label.setWordWrap(False)
            preview_label.setStyleSheet("font-size: 13px;")
            preview_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            content.addWidget(preview_label)

            root.addWidget(badge, alignment=Qt.AlignmentFlag.AlignVCenter)
            root.addLayout(content, stretch=1)

            self.setStyleSheet(
                "#processingFeedRow {"
                "background: rgba(255, 215, 120, 14);"
                "border: 1px solid rgba(255, 215, 120, 80);"
                "border-radius: 12px;"
                "}"
            )

    class ControllerWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.conn = connect()
            normalize_inflight_items(self.conn)
            self.current_item_id: str | None = None
            self.current_duration_ms = 0
            self.current_audio_device_id: bytes = b""
            self.current_play_origin: str | None = None
            self.current_resume_status: str | None = None
            self.is_scrubbing = False
            self.processing_items: dict[str, ProcessingItem] = {}
            self.processing_sequence = 0
            self.playback_rate = 1.0

            self.audio_output = QAudioOutput(QMediaDevices.defaultAudioOutput())
            self.player = QMediaPlayer()
            self.player.setAudioOutput(self.audio_output)
            self.player.setPlaybackRate(self.playback_rate)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.mediaStatusChanged.connect(self._on_media_status_changed)
            self.player.errorOccurred.connect(self._on_error_occurred)

            style = QApplication.style()
            self.play_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.pause_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.volume_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaVolume)

            self.setWindowTitle("AgentTools Audio")
            self.resize(620, 680)
            self._build_ui()
            self._setup_shortcuts()
            self._setup_tray()

            self.command_timer = QTimer(self)
            self.command_timer.timeout.connect(self._drain_commands)
            self.command_timer.start(150)

            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self._on_refresh_tick)
            self.refresh_timer.start(1000)

            self._sync_default_audio_output(force=True)
            self.refresh_views()
            self._maybe_autoplay()

        def _build_ui(self) -> None:
            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)

            self.now_label = QLabel("No audio playing")
            self.now_label.setWordWrap(True)
            self.now_label.setStyleSheet("font-size: 14px; font-weight: 600;")

            slider_row = QHBoxLayout()
            slider_row.setContentsMargins(0, 0, 0, 0)
            slider_row.setSpacing(8)

            self.elapsed_label = QLabel("0:00")
            self.total_label = QLabel("0:00")
            self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
            self.timeline_slider.setRange(0, 0)
            self.timeline_slider.setEnabled(False)
            self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
            self.timeline_slider.sliderReleased.connect(self._on_slider_released)
            self.timeline_slider.sliderMoved.connect(self._on_slider_moved)

            slider_row.addWidget(self.elapsed_label)
            slider_row.addWidget(self.timeline_slider, stretch=1)
            slider_row.addWidget(self.total_label)

            controls_row = QHBoxLayout()
            controls_row.setContentsMargins(0, 0, 0, 0)
            controls_row.setSpacing(8)

            self.play_pause_button = QPushButton("Play/Pause")
            self.stop_button = QPushButton("Stop")
            self.next_button = QPushButton("Next queued")
            self.slower_button = QPushButton("Slower")
            self.faster_button = QPushButton("Faster")
            self.playback_rate_value = QLabel(playback_rate_label(self.playback_rate))
            self.playback_rate_value.setStyleSheet("font-size: 12px; font-weight: 600;")

            self.play_pause_button.clicked.connect(self.play_pause)
            self.stop_button.clicked.connect(self.stop_current)
            self.next_button.clicked.connect(self.skip_next)
            self.slower_button.clicked.connect(self.slower_playback)
            self.faster_button.clicked.connect(self.faster_playback)

            controls_row.addWidget(self.play_pause_button)
            controls_row.addWidget(self.stop_button)
            controls_row.addWidget(self.next_button)
            controls_row.addSpacing(12)
            controls_row.addWidget(self.slower_button)
            controls_row.addWidget(self.playback_rate_value)
            controls_row.addWidget(self.faster_button)
            controls_row.addStretch(1)

            self.feed_list = QListWidget()
            self.feed_list.setSpacing(6)
            self.feed_list.setAlternatingRowColors(False)

            current_text_title = QLabel("Current text")
            current_text_title.setStyleSheet("font-size: 12px; font-weight: 600;")

            self.current_meta_label = QLabel("")
            self.current_meta_label.setWordWrap(True)
            self.current_meta_label.setStyleSheet(
                "font-size: 11px; color: rgba(255, 255, 255, 150);"
            )

            self.current_text = QPlainTextEdit()
            self.current_text.setReadOnly(True)
            self.current_text.setPlaceholderText("Playback text will appear here.")
            self.current_text.setMinimumHeight(180)

            layout.addWidget(self.now_label)
            layout.addLayout(slider_row)
            layout.addLayout(controls_row)
            layout.addWidget(self.feed_list, stretch=1)
            layout.addWidget(current_text_title)
            layout.addWidget(self.current_meta_label)
            layout.addWidget(self.current_text)

            self.setCentralWidget(central)
            self.setWindowIcon(self.volume_icon)

        def _setup_shortcuts(self) -> None:
            self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
            self.space_shortcut.activated.connect(self.play_pause)

            self.escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
            self.escape_shortcut.activated.connect(self.stop_current)

            self.next_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
            self.next_shortcut.activated.connect(self.skip_next)

        def _setup_tray(self) -> None:
            self.tray_menu = QMenu(self)

            show_action = QAction("Show", self)
            show_action.triggered.connect(self.show_and_focus)
            self.tray_menu.addAction(show_action)

            play_pause_action = QAction("Play/Pause", self)
            play_pause_action.triggered.connect(self.play_pause)
            self.tray_menu.addAction(play_pause_action)

            stop_action = QAction("Stop", self)
            stop_action.triggered.connect(self.stop_current)
            self.tray_menu.addAction(stop_action)

            next_action = QAction("Next queued", self)
            next_action.triggered.connect(self.skip_next)
            self.tray_menu.addAction(next_action)

            self.tray_menu.addSeparator()

            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.quit_app)
            self.tray_menu.addAction(quit_action)

            self.tray = QSystemTrayIcon(self.windowIcon(), self)
            self.tray.setToolTip("AgentTools Audio")
            self.tray.setContextMenu(self.tray_menu)
            self.tray.activated.connect(self._on_tray_activated)
            self.tray.show()

        def _on_tray_activated(self, reason: object) -> None:
            if reason == QSystemTrayIcon.ActivationReason.Trigger:
                self.show_and_focus()
            elif reason == QSystemTrayIcon.ActivationReason.Context:
                self.tray_menu.popup(QCursor.pos())

        def closeEvent(self, event) -> None:
            event.ignore()
            self.hide()
            self.tray.showMessage(
                "AgentTools",
                "Audio playback continues in the tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )

        def quit_app(self) -> None:
            self.tray.hide()
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
            processing_items = self._sorted_processing_items()
            all_items = list_all_items(self.conn, limit=100)
            self._render_current_section(current, processing_items=processing_items)
            self._render_feed(processing_items, all_items)

        def _render_current_section(
            self,
            current: QueueItem | None,
            *,
            processing_items: list[ProcessingItem],
        ) -> None:
            if current is None:
                if processing_items:
                    count = len(processing_items)
                    label = "item" if count == 1 else "items"
                    self.now_label.setText(f"Processing {count} {label}")
                    self.current_meta_label.setText(_format_processing_meta(processing_items[0]))
                    self.current_text.setPlainText(processing_items[0].detail_text)
                else:
                    self.now_label.setText("No audio playing")
                    self.current_meta_label.setText("")
                    self.current_text.setPlainText("")
                self.timeline_slider.setRange(0, 0)
                self.timeline_slider.setValue(0)
                self.timeline_slider.setEnabled(False)
                self.elapsed_label.setText("0:00")
                self.total_label.setText("0:00")
                if self.player.duration() <= 0:
                    self.current_duration_ms = 0
                return

            label = current.source_label or "unknown source"
            self.now_label.setText(label)
            self.current_meta_label.setText(_format_item_meta(current))
            self.current_text.setPlainText(current.tts_text)
            self.timeline_slider.setEnabled(True)
            if self.current_duration_ms <= 0:
                self.current_duration_ms = current.duration_ms
            if self.current_duration_ms > 0:
                self.timeline_slider.setRange(0, self.current_duration_ms)
                self.total_label.setText(_format_duration_ms(self.current_duration_ms))

        def _render_feed(
            self,
            processing_items: list[ProcessingItem],
            queue_items: list[QueueItem],
        ) -> None:
            scroll_bar = self.feed_list.verticalScrollBar()
            old_value = scroll_bar.value()
            old_max = scroll_bar.maximum()
            self.feed_list.clear()
            playback_state = self.player.playbackState()
            is_playing = playback_state == QMediaPlayer.PlaybackState.PlayingState

            entries = merged_feed_entries(processing_items, queue_items)
            for entry in entries:
                list_item = QListWidgetItem()
                list_item.setData(Qt.ItemDataRole.UserRole, entry.key)
                row_widget: QWidget
                if entry.processing_item is not None:
                    row_widget = ProcessingFeedRow(
                        item=entry.processing_item,
                    )
                else:
                    queue_item = entry.queue_item
                    if queue_item is None:
                        continue
                    row_widget = AudioFeedRow(
                        item=queue_item,
                        is_current=queue_item.item_id == self.current_item_id,
                        is_playing=queue_item.item_id == self.current_item_id and is_playing,
                        play_icon=self.play_icon,
                        pause_icon=self.pause_icon,
                        on_activate=self._activate_item,
                    )
                list_item.setSizeHint(QSize(0, FEED_ROW_HEIGHT))
                self.feed_list.addItem(list_item)
                self.feed_list.setItemWidget(list_item, row_widget)

            scroll_bar.setValue(
                restored_scroll_value(
                    old_value=old_value,
                    old_max=old_max,
                    new_max=scroll_bar.maximum(),
                )
            )

        def _sorted_processing_items(self) -> list[ProcessingItem]:
            return sorted(
                self.processing_items.values(),
                key=lambda item: item.order,
                reverse=True,
            )

        def _on_refresh_tick(self) -> None:
            self._sync_default_audio_output()
            self.refresh_views()

        def _sync_default_audio_output(self, *, force: bool = False) -> None:
            default_device = QMediaDevices.defaultAudioOutput()
            device_id = default_device.id().data()
            if not force and device_id == self.current_audio_device_id:
                return

            was_muted = self.audio_output.isMuted()
            volume = self.audio_output.volume()
            self.audio_output.setDevice(default_device)
            self.audio_output.setMuted(was_muted)
            self.audio_output.setVolume(volume)
            self.current_audio_device_id = device_id

        def _drain_commands(self) -> None:
            saw_refresh = False
            queue_refresh = False
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
                    queue_refresh = True
                elif cmd.action == "shutdown":
                    self.quit_app()
                    return
                elif cmd.action == "processing-start":
                    self._start_processing(cmd)
                    saw_refresh = True
                elif cmd.action == "processing-update":
                    self._update_processing(cmd)
                    saw_refresh = True
                elif cmd.action == "processing-finish":
                    self._finish_processing(cmd)
                    saw_refresh = True
            if queue_refresh and self._should_advance_from_paused_current():
                self.skip_next()
                return
            if saw_refresh:
                self.refresh_views()
                self._maybe_autoplay()


        def _start_processing(self, cmd: ControllerCommand) -> None:
            if not cmd.progress_id:
                return
            existing = self.processing_items.get(cmd.progress_id)
            if existing is None:
                self.processing_sequence += 1
                order = self.processing_sequence
            else:
                order = existing.order
            self.processing_items[cmd.progress_id] = ProcessingItem(
                progress_id=cmd.progress_id,
                source_label=cmd.source_label,
                preview_text=(cmd.preview_text or "Processing audio").strip() or "Processing audio",
                detail_text=(cmd.detail_text or cmd.preview_text or "").strip(),
                stage=processing_stage_label(cmd.stage),
                order=order,
            )

        def _update_processing(self, cmd: ControllerCommand) -> None:
            if not cmd.progress_id:
                return
            existing = self.processing_items.get(cmd.progress_id)
            if existing is None:
                return
            self.processing_items[cmd.progress_id] = ProcessingItem(
                progress_id=existing.progress_id,
                source_label=cmd.source_label or existing.source_label,
                preview_text=(cmd.preview_text or existing.preview_text).strip()
                or existing.preview_text,
                detail_text=(cmd.detail_text or existing.detail_text).strip()
                or existing.detail_text,
                stage=processing_stage_label(cmd.stage or existing.stage),
                order=existing.order,
            )

        def _finish_processing(self, cmd: ControllerCommand) -> None:
            if not cmd.progress_id:
                return
            self.processing_items.pop(cmd.progress_id, None)

        def _should_advance_from_paused_current(self) -> bool:
            if self.current_item_id is None:
                return False
            if self.player.playbackState() != QMediaPlayer.PlaybackState.PausedState:
                return False
            return get_next_queued_item(self.conn) is not None

        def _maybe_autoplay(self) -> None:
            if self.current_item_id is not None:
                return
            next_item = get_next_queued_item(self.conn)
            if next_item is None:
                return
            self._start_item(next_item, origin="queue")

        def _activate_item(self, item_id: str) -> None:
            try:
                item = get_item_by_item_id(self.conn, item_id)
            except KeyError:
                self.refresh_views()
                return

            if self.current_item_id == item_id:
                self.play_pause()
                return

            origin = "queue" if item.status == STATUS_QUEUED else "history"
            self._interrupt_current_for_switch()
            self._start_item(item, origin=origin)

        def _interrupt_current_for_switch(self) -> None:
            if self.current_item_id is None:
                return
            self.player.stop()
            if self.current_resume_status is not None:
                update_status(self.conn, self.current_item_id, self.current_resume_status)
            self._clear_current_playback_state(reset_slider=False)

        def _start_item(self, item: QueueItem, *, origin: str) -> None:
            self.current_item_id = item.item_id
            self.current_play_origin = origin
            self.current_resume_status = interrupted_status_for_switch(item.status)
            self.current_duration_ms = item.duration_ms
            update_status(self.conn, item.item_id, STATUS_PLAYING)
            self.timeline_slider.setRange(0, max(0, item.duration_ms))
            self.timeline_slider.setValue(0)
            self.elapsed_label.setText("0:00")
            self.total_label.setText(_format_duration_ms(item.duration_ms))
            self.player.setSource(QUrl.fromLocalFile(item.audio_path))
            self.player.setPosition(0)
            self.player.setPlaybackRate(self.playback_rate)
            self.player.play()
            self.refresh_views()

        def play_pause(self) -> None:
            state = self.player.playbackState()
            if self.current_item_id is None:
                self._maybe_autoplay()
                self.refresh_views()
                return

            if state == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
                update_status(self.conn, self.current_item_id, STATUS_PAUSED)
            elif state == QMediaPlayer.PlaybackState.PausedState:
                self.player.play()
                update_status(self.conn, self.current_item_id, STATUS_PLAYING)
            else:
                current = get_current_item(self.conn)
                if current is not None:
                    self._start_item(current, origin=self.current_play_origin or "queue")
                else:
                    self._maybe_autoplay()
            self.refresh_views()

        def stop_current(self) -> None:
            if self.current_item_id is None:
                return
            self.player.stop()
            update_status(self.conn, self.current_item_id, STATUS_STOPPED)
            self._clear_current_playback_state(reset_slider=True)
            self.refresh_views()

        def slower_playback(self) -> None:
            self._set_playback_rate(self.playback_rate - PLAYBACK_RATE_STEP)

        def faster_playback(self) -> None:
            self._set_playback_rate(self.playback_rate + PLAYBACK_RATE_STEP)

        def _set_playback_rate(self, rate: float) -> None:
            self.playback_rate = clamp_playback_rate(rate)
            self.player.setPlaybackRate(self.playback_rate)
            self.playback_rate_value.setText(playback_rate_label(self.playback_rate))

        def skip_next(self) -> None:
            if self.current_item_id is not None:
                self.player.stop()
                update_status(self.conn, self.current_item_id, STATUS_STOPPED)
                self._clear_current_playback_state(reset_slider=True)
            next_item = get_next_queued_item(self.conn)
            if next_item is not None:
                self._start_item(next_item, origin="queue")
            else:
                self.refresh_views()

        def _clear_current_playback_state(self, *, reset_slider: bool) -> None:
            self.current_item_id = None
            self.current_play_origin = None
            self.current_resume_status = None
            self.current_duration_ms = 0
            if reset_slider:
                self.timeline_slider.setRange(0, 0)
                self.timeline_slider.setValue(0)
                self.timeline_slider.setEnabled(False)
                self.elapsed_label.setText("0:00")
                self.total_label.setText("0:00")

        def _on_slider_pressed(self) -> None:
            self.is_scrubbing = True

        def _on_slider_released(self) -> None:
            self.is_scrubbing = False
            if self.current_item_id is None:
                return
            self.player.setPosition(self.timeline_slider.value())
            self.elapsed_label.setText(_format_duration_ms(self.timeline_slider.value()))

        def _on_slider_moved(self, position_ms: int) -> None:
            self.elapsed_label.setText(_format_duration_ms(position_ms))

        def _on_position_changed(self, position_ms: int) -> None:
            if not self.is_scrubbing:
                self.timeline_slider.setValue(position_ms)
            self.elapsed_label.setText(_format_duration_ms(position_ms))

        def _on_duration_changed(self, duration_ms: int) -> None:
            if duration_ms <= 0:
                return
            self.current_duration_ms = duration_ms
            self.timeline_slider.setRange(0, duration_ms)
            self.total_label.setText(_format_duration_ms(duration_ms))

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
                self._clear_current_playback_state(reset_slider=True)
                self.refresh_views()
                self._maybe_autoplay()

        def _on_error_occurred(self, *_args) -> None:
            if self.current_item_id is not None:
                message = self.player.errorString() or "Unknown playback error."
                update_status(self.conn, self.current_item_id, STATUS_FAILED, error_message=message)
                self._clear_current_playback_state(reset_slider=True)
                self.refresh_views()
                self._maybe_autoplay()
                QMessageBox.warning(self, "Playback error", message)

    app = QApplication.instance() or QApplication(sys.argv)
    window = ControllerWindow()
    if not hidden:
        window.show_and_focus()
    return int(app.exec())


def merged_feed_entries(
    processing_items: list[ProcessingItem],
    queue_items: list[QueueItem],
) -> list[FeedEntry]:
    entries = [
        FeedEntry(
            kind="processing",
            key=_processing_entry_key(item.progress_id),
            processing_item=item,
        )
        for item in processing_items
    ]
    entries.extend(
        FeedEntry(kind="queue", key=_queue_entry_key(item.item_id), queue_item=item)
        for item in queue_items
    )
    return entries


def _queue_entry_key(item_id: str) -> str:
    return f"queue:{item_id}"


def _processing_entry_key(progress_id: str) -> str:
    return f"processing:{progress_id}"


def _coerce_optional_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _preview_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _format_duration_ms(duration_ms: int) -> str:
    total_seconds = max(0, duration_ms // 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_item_meta(item: QueueItem) -> str:
    parts = []
    if item.source_label:
        parts.append(item.source_label)
    parts.append(_format_duration_ms(item.duration_ms))
    if item.voice:
        parts.append(f"voice {item.voice}")
    if item.model:
        parts.append(item.model)
    if item.language:
        parts.append(f"lang {item.language}")
    return " | ".join(parts)


def _format_processing_meta(item: ProcessingItem) -> str:
    parts = [processing_stage_label(item.stage)]
    if item.source_label:
        parts.append(item.source_label)
    detail = _preview_text(item.detail_text, limit=220)
    if detail and detail != _preview_text(item.preview_text, limit=220):
        parts.append(detail)
    return " | ".join(part for part in parts if part)
