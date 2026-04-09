[app]
title = Security Voice Embedded
package.name = securityvoice
package.domain = org.newt
source.dir = src-python
source.include_exts = py,png,jpg,kv,atlas,html,js,css,onnx
version = 1.0

# Các thư viện Python cần nạp vào app mobile (Hardcore: chỉ chọn những thứ thật sự cần)
requirements = python3, kivy, numpy, onnxruntime, pyjnius, requests

# Chế độ chạy ngầm cho Android (Background Service)
services = MyBackgroundService:service.py

# Quyền hạn cần thiết (Ghi âm để nghe giọng nói)
android.permissions = RECORD_AUDIO, WAKE_LOCK, FOREGROUND_SERVICE

# Giao diện chính (Sử dụng Webview để load UI Vue.js từ file index.html)
orientation = portrait
fullscreen = 1

# Cấu hình cụ thể cho Mobile Build
android.api = 33
android.minapi = 21
android.archs = arm64-v8a, armeabi-v7a

[buildozer]
log_level = 2
warn_on_root = 1
