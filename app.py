import static_ffmpeg
static_ffmpeg.add_paths()
import os
import sys
import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar, QTextEdit
)
from PySide6.QtCore import QThread, Signal, Qt
from faster_whisper import WhisperModel

def format_timestamp(seconds: float) -> str:
    """將秒數轉換為 SRT 的時間戳格式 (00:00:00,000)"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

class TranscribeThread(QThread):
    """背景語音辨識線程，防止 GUI 凍結"""
    progress_signal = Signal(str)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self, video_path: str, output_path: str, model_size: str, language: str):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.model_size = model_size
        self.language = None if language == "自動偵測 (Auto)" else language

    def run(self):
        try:
            self.progress_signal.emit("正在載入語音辨識模型（首次執行會自動下載）...")
            
            # 使用 CPU 模式（預設 compute_type="int8" 增強 CPU 處理速度）
            # 若使用 Nvidia 顯卡可調成 device="cuda", compute_type="float16"
            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

            self.progress_signal.emit("開始掃描並辨識影片聲音內容...")
            segments, info = model.transcribe(
                self.video_path,
                language=self.language,
                beam_size=5
            )

            self.progress_signal.emit(f"偵測到語言：{info.language} (信心度：{info.language_probability:.2f})")

            srt_content = []
            for i, segment in enumerate(segments, start=1):
                start_str = format_timestamp(segment.start)
                end_str = format_timestamp(segment.end)
                text = segment.text.strip()
                
                # 組合 SRT 單一區塊
                srt_block = f"{i}\n{start_str} --> {end_str}\n{text}\n"
                srt_content.append(srt_block)
                
                # 即時回傳進度日誌
                self.progress_signal.emit(f"[{start_str} -> {end_str}] {text}")

            # 寫入 SRT 檔案 (UTF-8 編碼)
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_content))

            self.finished_signal.emit(self.output_path)

        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("影片自動生成字幕工具 (SRT Generator)")
        self.resize(700, 500)
        
        self.selected_video_path = ""
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 1. 檔案選擇區塊
        file_layout = QHBoxLayout()
        self.file_label = QLabel("尚未選擇影片檔案")
        self.btn_select_file = QPushButton("匯入影片")
        self.btn_select_file.clicked.connect(self.select_video)
        file_layout.addWidget(self.file_label, stretch=1)
        file_layout.addWidget(self.btn_select_file)
        layout.addLayout(file_layout)

        # 2. 設定選擇區塊（模型大小與語言）
        config_layout = QHBoxLayout()
        
        config_layout.addWidget(QLabel("模型精準度:"))
        self.combo_model = QComboBox()
        # tiny/base/small/medium/large-v3，small 適合大多數中文/英文辨識
        self.combo_model.addItems(["base", "small", "medium"])
        self.combo_model.setCurrentText("small")
        config_layout.addWidget(self.combo_model)

        config_layout.addWidget(QLabel("影片講話語言:"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(["自動偵測 (Auto)", "zh", "en", "ja", "ko"])
        config_layout.addWidget(self.combo_lang)

        layout.addLayout(config_layout)

        # 3. 開始執行按鈕
        self.btn_start = QPushButton("開始轉換並匯出 SRT 字幕")
        self.btn_start.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_start.clicked.connect(self.start_transcription)
        layout.addWidget(self.btn_start)

        # 4. 日誌顯示區塊
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇影片", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.wmv)"
        )
        if file_path:
            self.selected_video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.log_area.append(f"已載入檔案：{file_path}")

    def start_transcription(self):
        if not self.selected_video_path:
            self.log_area.append("錯誤：請先點擊「匯入影片」選擇檔案！")
            return

        # 自動生成輸出的 srt 路徑（與影片同檔名、同資料夾）
        base_name = os.path.splitext(self.selected_video_path)[0]
        output_srt_path = f"{base_name}.srt"

        model_size = self.combo_model.currentText()
        language = self.combo_lang.currentText()

        # UI 鎖定
        self.btn_start.setEnabled(False)
        self.btn_select_file.setEnabled(False)
        self.log_area.append("\n================ 開始處理 ================")

        # 啟動背景工作線程
        self.thread = TranscribeThread(
            self.selected_video_path, output_srt_path, model_size, language
        )
        self.thread.progress_signal.connect(self.update_log)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.error_signal.connect(self.on_error)
        self.thread.start()

    def update_log(self, text: str):
        self.log_area.append(text)

    def on_finished(self, output_path: str):
        self.log_area.append("\n================ 處理完成 ================")
        self.log_area.append(f"SRT 字幕檔已成功儲存於：\n{output_path}")
        self.btn_start.setEnabled(True)
        self.btn_select_file.setEnabled(True)

    def on_error(self, error_msg: str):
        self.log_area.append(f"\n發生錯誤：{error_msg}")
        self.btn_start.setEnabled(True)
        self.btn_select_file.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())