import sys
import os

# 1. 解決 PyInstaller --noconsole 模式下 static_ffmpeg / whisper 輸出 NoneType 崩潰問題
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import static_ffmpeg
static_ffmpeg.add_paths()

import opencc
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit, QComboBox
)
from PySide6.QtCore import QThread, Signal, Qt


# 2. 背景轉譯線程（支援 暫停/恢復/停止、繁體轉換與模型快取目錄重定向）
class TranscribeWorker(QThread):
    progress_signal = Signal(int)
    log_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, video_path, output_dir, model_size="base"):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.model_size = model_size
        
        # 控制狀態標記
        self._is_paused = False
        self._is_stopped = False
        self.cc = opencc.OpenCC('s2twp')  # 簡體轉臺灣繁體

    def pause(self):
        self._is_paused = True
        self.log_signal.emit("⏸️ 已按下暫停...")

    def resume(self):
        self._is_paused = False
        self.log_signal.emit("▶️ 恢復轉譯任務...")

    def stop(self):
        self._is_stopped = True
        self.log_signal.emit("⏹️ 正在停止任務...")

    def run(self):
        try:
            # 指定模型下載快取目錄至 D 槽，避免佔用或擠爆 C 槽
            cache_dir = r"D:\whisper_cache"
            os.makedirs(cache_dir, exist_ok=True)

            self.log_signal.emit(f"正在載入 Faster-Whisper 模型 ({self.model_size})...")
            self.log_signal.emit(f"模型快取位置: {cache_dir}")
            
            from faster_whisper import WhisperModel
            model = WhisperModel(
                self.model_size, 
                device="cpu", 
                compute_type="int8", 
                download_root=cache_dir
            )

            self.log_signal.emit(f"開始分析影片: {os.path.basename(self.video_path)}")
            segments, info = model.transcribe(self.video_path, beam_size=5)

            total_duration = info.duration if info.duration > 0 else 1.0
            
            # 設定輸出檔名 (.srt)
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            srt_filename = f"{base_name}.srt"
            srt_path = os.path.join(self.output_dir, srt_filename)

            self.log_signal.emit(f"字幕將儲存至: {srt_path}")

            srt_lines = []
            for i, segment in enumerate(segments, start=1):
                # 檢查是否請求停止
                if self._is_stopped:
                    self.log_signal.emit("❌ 任務已被使用者手動取消。")
                    return

                # 檢查是否暫停（迴圈等待）
                while self._is_paused:
                    if self._is_stopped:
                        self.log_signal.emit("❌ 任務已被使用者手動取消。")
                        return
                    self.msleep(200)

                # 轉換時間格式 hh:mm:ss,mss
                start_time = self.format_timestamp(segment.start)
                end_time = self.format_timestamp(segment.end)
                
                # 自動轉換為繁體中文
                traditional_text = self.cc.convert(segment.text.strip())

                srt_block = f"{i}\n{start_time} --> {end_time}\n{traditional_text}\n\n"
                srt_lines.append(srt_block)

                # 計算並發送進度
                progress = int((segment.end / total_duration) * 100)
                progress = min(progress, 100)
                self.progress_signal.emit(progress)
                self.log_signal.emit(f"[{start_time} -> {end_time}] {traditional_text}")

            # 寫入 SRT 檔案
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.writelines(srt_lines)

            self.progress_signal.emit(100)
            self.finished_signal.emit(srt_path)

        except Exception as e:
            self.error_signal.emit(str(e))

    def format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


# 3. 主 UI 視窗介面
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoToSRT 影片轉字幕工具 (繁體中文版)")
        self.resize(650, 500)

        self.video_path = ""
        self.output_dir = os.path.expanduser("~/Desktop")  # 預設輸出至桌面
        self.worker = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # ---- 1. 選擇影片檔案 ----
        file_layout = QHBoxLayout()
        self.lbl_file = QLabel("未選擇影片檔案")
        btn_select_file = QPushButton("選擇影片")
        btn_select_file.clicked.connect(self.select_video)
        file_layout.addWidget(self.lbl_file, 1)
        file_layout.addWidget(btn_select_file)
        main_layout.addLayout(file_layout)

        # ---- 2. 選擇輸出資料夾 ----
        output_layout = QHBoxLayout()
        self.lbl_output = QLabel(f"輸出目錄: {self.output_dir}")
        btn_select_output = QPushButton("更改目錄")
        btn_select_output.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.lbl_output, 1)
        output_layout.addWidget(btn_select_output)
        main_layout.addLayout(output_layout)

        # ---- 3. 模型大小選擇 ----
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Whisper 模型大小:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["tiny", "base", "small", "medium"])
        self.combo_model.setCurrentText("base")
        model_layout.addWidget(self.combo_model, 1)
        main_layout.addLayout(model_layout)

        # ---- 4. 進度條 ----
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # ---- 5. 控制按鈕 (開始 / 暫停 / 停止) ----
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶️ 開始轉譯")
        self.btn_pause = QPushButton("⏸️ 暫停")
        self.btn_stop = QPushButton("⏹️ 停止")

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_transcription)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_stop.clicked.connect(self.stop_transcription)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        main_layout.addLayout(btn_layout)

        # ---- 6. 日誌顯示區域 ----
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        main_layout.addWidget(self.txt_log)

        # ---- 7. 開發者署名標籤 (右下角) ----
        lbl_developer = QLabel("Developed by Daniel 趙偉智")  # 請在此替換為你的名字
        lbl_developer.setAlignment(Qt.AlignRight)
        lbl_developer.setStyleSheet("color: #888888; font-size: 11px; padding-top: 2px;")
        main_layout.addWidget(lbl_developer)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片或音訊", "", "Video/Audio Files (*.mp4 *.mkv *.avi *.mov *.mp3 *.wav *.m4a)"
        )
        if path:
            self.video_path = path
            self.lbl_file.setText(os.path.basename(path))
            self.log(f"已選取檔案: {path}")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇 SRT 字幕檔輸出路徑", self.output_dir)
        if dir_path:
            self.output_dir = dir_path
            self.lbl_output.setText(f"輸出目錄: {self.output_dir}")
            self.log(f"已更新輸出目錄: {dir_path}")

    def start_transcription(self):
        if not self.video_path:
            self.log("❌ 請先選擇影片檔案！")
            return

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setText("⏸️ 暫停")
        self.progress_bar.setValue(0)
        self.txt_log.clear()

        # 啟動 Worker 線程
        model_size = self.combo_model.currentText()
        self.worker = TranscribeWorker(self.video_path, self.output_dir, model_size)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def toggle_pause(self):
        if self.worker and self.worker.isRunning():
            if not self.worker._is_paused:
                self.worker.pause()
                self.btn_pause.setText("▶️ 繼續")
            else:
                self.worker.resume()
                self.btn_pause.setText("⏸️ 暫停")

    def stop_transcription(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.reset_btn_states()

    def on_finished(self, srt_path):
        self.log(f"\n🎉 轉換完成！字幕已儲存至: {srt_path}")
        self.reset_btn_states()

    def on_error(self, err_msg):
        self.log(f"\n❌ 發生錯誤: {err_msg}")
        self.reset_btn_states()

    def reset_btn_states(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setText("⏸️ 暫停")

    def log(self, text):
        self.txt_log.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())