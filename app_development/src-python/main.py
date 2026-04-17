import os
import sys

# Thêm đường dẫn để có thể import các module cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from engine import AppEngine

KV = '''
BoxLayout:
    orientation: 'vertical'
    padding: 20
    spacing: 10
    canvas.before:
        Color:
            rgba: 0.07, 0.07, 0.07, 1
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        text: 'SECURITY VOICE EMBEDDED'
        font_size: '24sp'
        size_hint_y: None
        height: 50
        color: 0.9, 0.9, 0.9, 1

    Label:
        id: status_label
        text: 'Trạng thái: Sẵn sàng'
        color: 0.3, 0.8, 0.3, 1
        size_hint_y: None
        height: 30

    ScrollView:
        Label:
            id: transcription_label
            text: '...'
            size_hint_y: None
            height: self.texture_size[1]
            text_size: self.width, None
            halign: 'left'
            valign: 'top'
            padding: 10, 10

    BoxLayout:
        size_hint_y: None
        height: 60
        spacing: 10
        
        Button:
            id: start_btn
            text: 'BẮT ĐẦU LẮNG NGHE'
            background_color: 0.1, 0.6, 0.1, 1
            on_press: app.toggle_listening()

        Button:
            text: 'XÁC THỰC: TẮT'
            id: auth_btn
            background_color: 0.6, 0.1, 0.1, 1
            on_press: app.toggle_auth()
'''

class VoiceApp(App):
    def build(self):
        # Đường dẫn mô hình (Điều chỉnh cho phù hợp với môi trường hiện tại)
        WHISPER_MODEL = "quantization/ggml-whispersmall-multilingual/q5_1/ggml-model-q5_1.bin"
        WHISPER_EXE = "whisper.cpp/build/bin/main"
        
        self.engine = AppEngine(WHISPER_MODEL, WHISPER_EXE)
        self.root = Builder.load_string(KV)
        self.is_listening = False
        return self.root

    def toggle_listening(self):
        if not self.is_listening:
            self.engine.start(self.on_engine_event)
            self.root.ids.start_btn.text = 'DỪNG LẮNG NGHE'
            self.root.ids.start_btn.background_color = (0.6, 0.1, 0.1, 1)
            self.root.ids.status_label.text = 'Trạng thái: Đang lắng nghe...'
            self.is_listening = True
        else:
            self.engine.stop()
            self.root.ids.start_btn.text = 'BẮT ĐẦU LẮNG NGHE'
            self.root.ids.start_btn.background_color = (0.1, 0.6, 0.1, 1)
            self.root.ids.status_label.text = 'Trạng thái: Sẵn sàng'
            self.is_listening = False

    def toggle_auth(self):
        new_state = not self.engine.is_auth_enabled
        self.engine.set_auth(new_state)
        self.root.ids.auth_btn.text = f'XÁC THỰC: {"BẬT" if new_state else "TẮT"}'
        self.root.ids.auth_btn.background_color = (0.1, 0.1, 0.6, 1) if new_state else (0.6, 0.1, 0.1, 1)

    def on_engine_event(self, event_type, data):
        # Kivy UI phải được cập nhật từ main thread
        Clock.schedule_once(lambda dt: self._update_ui(event_type, data))

    def _update_ui(self, event_type, data):
        if event_type == "transcription":
            current_text = self.root.ids.transcription_label.text
            if current_text == "...": current_text = ""
            self.root.ids.transcription_label.text = current_text + "\n> " + data
        elif event_type == "auth":
            self.root.ids.status_label.text = f'Xác thực: {data}'

if __name__ == '__main__':
    VoiceApp().run()
