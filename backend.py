import os, base64, binascii, logging, time, re, ast
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from io import BytesIO
import shutil, tempfile
from flask import send_file, after_this_request
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()
logging.basicConfig(level=logging.INFO)
app = Flask(__name__, static_folder="static")
# Render đứng sau proxy: lấy IP thật của người dùng để rate-limit không bị gộp chung
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
limiter = Limiter(get_remote_address, app=app, default_limits=["30 per minute"], storage_uri="memory://")
CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image").strip()
client = genai.Client(api_key=API_KEY) if API_KEY else None
_client_cache = {}

def get_client():
    """Ưu tiên API key người dùng nhập trong trang web (header X-Api-Key); nếu không có thì dùng key trên máy chủ."""
    user_key = (request.headers.get("X-Api-Key") or "").strip()
    if user_key and 20 <= len(user_key) <= 200:
        c = _client_cache.get(user_key)
        if c is None:
            if len(_client_cache) > 50: _client_cache.clear()
            c = _client_cache[user_key] = genai.Client(api_key=user_key)
        return c
    return client
SYSTEM = os.getenv("SYSTEM_INSTRUCTION",
    "Bạn là trợ lý AI của Nguyễn Anh Huy, nói tiếng Việt thân thiện, chính xác, trả lời rõ ràng có cấu trúc. "
    "Nếu thiếu dữ kiện hoặc không chắc chắn, hãy nói rõ. Không bịa nguồn.")

CODE_SYSTEM = ("Bạn là lập trình viên Python chuyên nghiệp. Hãy viết MỘT file Python hoàn chỉnh, chạy được ngay theo yêu cầu. "
    "CHỈ trả về mã nguồn thô: KHÔNG dùng khối markdown ```, KHÔNG giải thích, KHÔNG chữ thừa trước hoặc sau code. "
    "Giải thích cần thiết hãy viết bằng comment tiếng Việt trong code.")

MAX_FILE_BYTES = 10 * 1024 * 1024   # file code tối đa 10 MB

def strip_fences(t):
    t = (t or "").strip("\r\n")
    t = re.sub(r"^\s*```[a-zA-Z0-9_+-]*[ \t]*\n", "", t)
    t = re.sub(r"\n?[ \t]*```\s*$", "", t)
    return t

def build_code_file(client_, contents, model_name):
    """Sinh code Python; nếu bị cắt giữa chừng thì tự yêu cầu viết tiếp (tối đa 4 lần), giới hạn 10 MB."""
    cfg = types.GenerateContentConfig(system_instruction=CODE_SYSTEM, temperature=0.3, max_output_tokens=60000)
    code, used, rounds = "", model_name, 0
    convo = list(contents)
    while True:
        resp, used = generate_with_retry(model_name, convo, cfg)
        piece = strip_fences(resp.text or "")
        code += (("\n" if code and not code.endswith("\n") else "") + piece) if rounds == 0 else piece
        rounds += 1
        fin = ""
        try: fin = str(resp.candidates[0].finish_reason)
        except Exception: pass
        if "MAX_TOKENS" not in fin or rounds >= 5 or len(code.encode("utf-8")) >= MAX_FILE_BYTES:
            break
        convo = convo + [types.Content(role="model", parts=[types.Part(text=piece)]),
                         types.Content(role="user", parts=[types.Part(text="Tiếp tục viết tiếp CHÍNH XÁC từ ký tự cuối cùng ở trên. Chỉ code thô, không lặp lại phần đã viết, không dùng ```.")])]
    raw = code.encode("utf-8")
    truncated = False
    if len(raw) > MAX_FILE_BYTES:
        raw = raw[:MAX_FILE_BYTES]; truncated = True
        raw = raw[:raw.rfind(b"\n") + 1] or raw
    return raw, truncated, used

@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
    return response

@app.get("/")
@limiter.exempt
def home():
    return send_from_directory("static", "index.html")

@app.get("/favicon.ico")
@limiter.exempt
def favicon():
    return "", 204

@app.get("/api/health")
@limiter.exempt
def health():
    return jsonify({"ok": True, "app": "Nguyễn Anh Huy AI",
                    "configured": bool(client)})

def require_client():
    if get_client() is None:
        return jsonify({"error": "Chưa có API key. Bấm biểu tượng ⚙ (Cài đặt) để dán Gemini API key của bạn."}), 503
    return None

def decode_attachment(att):
    if not isinstance(att, dict) or not att.get("data"):
        return None, None, None
    mime = str(att.get("mimeType", ""))
    if mime not in ("image/jpeg", "image/png", "image/webp", "application/pdf"):
        raise ValueError("Chỉ hỗ trợ ảnh JPG, PNG, WEBP và PDF.")
    try:
        raw = base64.b64decode(att["data"], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Dữ liệu tệp đính kèm không hợp lệ.")
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("Tệp tối đa 10 MB.")
    name = str(att.get("name", "attachment"))[:120]
    return mime, raw, name

# Model dự phòng khi model chính quá tải (503/500/504). Có thể đổi bằng biến GEMINI_FALLBACK_MODELS (cách nhau dấu phẩy)
FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",") if m.strip()]

def is_overloaded(exc):
    m = str(exc).lower()
    return any(x in m for x in ("503", "unavailable", "overloaded", "high demand", "500", "internal", "504", "deadline"))

def generate_with_retry(primary, contents, config, fallbacks=None):
    """Thử model chính 3 lần (chờ 1s, 2s), nếu vẫn quá tải thì chuyển sang model dự phòng."""
    models = [primary] + [m for m in (FALLBACKS if fallbacks is None else fallbacks) if m != primary]
    last = None
    for model in models:
        for attempt in range(3 if model == primary else 2):
            try:
                resp = get_client().models.generate_content(model=model, contents=contents, config=config)
                return resp, model
            except Exception as exc:
                last = exc
                if not is_overloaded(exc):
                    raise
                logging.warning("Model %s quá tải (lần %d): %s", model, attempt + 1, str(exc)[:120])
                time.sleep(1.0 * (attempt + 1))
    raise last

@app.post("/api/chat")
def chat():
    err = require_client()
    if err: return err
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    raw_images = data.get("images") if isinstance(data.get("images"), list) else []
    if not message and not raw_images and not data.get("attachment"):
        return jsonify({"error": "Bạn chưa nhập nội dung."}), 400
    if not message: message = "Hãy đọc và giải bài trong tệp đính kèm."

    if len(message) > 18000: return jsonify({"error": "Tin nhắn quá dài (tối đa 18.000 ký tự)."}), 413

    contents = []
    history = data.get("history", [])
    if isinstance(history, list):
        for item in history[-16:]:
            if not isinstance(item, dict): continue
            role = "user" if item.get("role") == "user" else "model"
            txt = str(item.get("content", "")).strip()
            if txt: contents.append(types.Content(role=role, parts=[types.Part(text=txt[:10000])]))

    parts = [types.Part(text=message)]
    try:
        for im in raw_images[:6]:
            m2, r2, _n = decode_attachment(im)
            if m2 and m2.startswith("image/"):
                parts.append(types.Part(inline_data=types.Blob(mime_type=m2, data=r2)))
        if raw_images:
            parts.append(types.Part(text="Hãy quan sát kỹ TẤT CẢ các ảnh đính kèm."))
        mime, raw, name = decode_attachment(data.get("attachment"))
        if mime and mime.startswith("image/"):
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=raw)))
            parts.append(types.Part(text="Hãy quan sát và phân tích hình ảnh đính kèm."))
        elif mime == "application/pdf":
            reader = PdfReader(BytesIO(raw))
            if len(reader.pages) > 35:
                return jsonify({"error": "PDF tối đa 35 trang trong một lần hỏi."}), 413
            extracted = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            if not extracted.strip():
                return jsonify({"error": "Không trích xuất được chữ trong PDF. Có thể đây là PDF scan; hãy gửi ảnh trang hoặc PDF có OCR."}), 422
            parts.append(types.Part(text="Nội dung trích xuất từ PDF ("+name+"):\n"+extracted[:90000]))
        contents.append(types.Content(role="user", parts=parts))

        if data.get("mode") == "code":
            raw, truncated, _used = build_code_file(None, contents, CHAT_MODEL)
            text = raw.decode("utf-8", errors="replace")
            note = ""
            try: ast.parse(text)
            except SyntaxError as se:
                note = f" ⚠ Phát hiện lỗi cú pháp quanh dòng {se.lineno}" + (" (file bị cắt ở mốc 10 MB)." if truncated else ", bạn nhờ mình sửa nhé.")
            if truncated and not note: note = " ⚠ File đã chạm mốc 10 MB nên bị cắt."
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return jsonify({"reply": f"✅ Đã tạo file Python: {len(text.splitlines())} dòng, {len(raw)/1024:.1f} KB.{note}",
                            "file": {"name": f"huy_code_{stamp}.py", "size": len(raw),
                                     "b64": base64.b64encode(raw).decode("ascii")}})

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM, temperature=0.65, max_output_tokens=(16000 if data.get("mode") == "fix" else 5000))
        if data.get("googleSearch") is True:
            config.tools = [types.Tool(google_search=types.GoogleSearch())]
        response, used = generate_with_retry(CHAT_MODEL, contents, config)
        return jsonify({"reply": response.text or "Chưa có nội dung trả về.", })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Gemini chat failed")
        msg = str(exc).lower()
        if any(x in msg for x in ("quota", "429", "resource_exhausted")):
            return jsonify({"error": "Đã chạm hạn mức AI. Hãy kiểm tra trang quản lý API key và thử lại sau."}), 429
        if any(x in msg for x in ("api key", "unauthorized", "permission_denied", "401")):
            return jsonify({"error": "API key không hợp lệ hoặc chưa được cấp quyền."}), 401
        if data.get("googleSearch") is True and ("google_search" in msg or "tool" in msg):
            return jsonify({"error": "Tài khoản hiện chưa hỗ trợ tìm kiếm web. Hãy tắt nút Tìm web hoặc đổi model."}), 502
        if is_overloaded(exc):
            return jsonify({"error": "Máy chủ AI đang quá tải. Mình đã tự thử lại và đổi model dự phòng nhưng chưa được, bạn đợi 1-2 phút rồi gửi lại nhé."}), 503
        return jsonify({"error": "Trợ lý AI gặp lỗi: " + str(exc)[:350]}), 502


# ---------------- TẢI VIDEO / MP3 (YouTube, TikTok) ----------------
URL_RE = re.compile(r"^(https?://)?([a-zA-Z0-9-]+\.)?(youtube\.com|youtu\.be|tiktok\.com)/\S*$", re.I)
MAX_DL_BYTES = int(os.getenv("MAX_DOWNLOAD_MB", "150")) * 1024 * 1024
HAS_FFMPEG = bool(shutil.which("ffmpeg"))

def run_download(url, kind, outdir):
    import yt_dlp
    tmpl = os.path.join(outdir, "%(title).80s_%(id)s.%(ext)s")
    opts = {"outtmpl": tmpl, "quiet": True, "no_warnings": True, "noplaylist": True,
            "retries": 3, "fragment_retries": 3, "geo_bypass": True,
            "max_filesize": MAX_DL_BYTES,
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}}}
    if kind == "audio":
        if HAS_FFMPEG:
            opts["format"] = "bestaudio/best"
            opts["postprocessors"] = [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}]
        else:
            opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
    else:
        if HAS_FFMPEG:
            opts["format"] = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best"
            opts["merge_output_format"] = "mp4"
        else:
            opts["format"] = "best[height<=720][ext=mp4]/best[ext=mp4]/best"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    files = [f for f in os.listdir(outdir) if not f.endswith((".part", ".ytdl"))]
    if not files: return None
    files.sort(key=lambda f: os.path.getsize(os.path.join(outdir, f)), reverse=True)
    return os.path.join(outdir, files[0])

@app.post("/api/download")
@limiter.limit("5 per minute")
def download():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    kind = "audio" if data.get("kind") == "audio" else "video"
    if not URL_RE.match(url):
        return jsonify({"error": "Chỉ hỗ trợ link YouTube hoặc TikTok."}), 400
    if not url.lower().startswith("http"): url = "https://" + url
    outdir = tempfile.mkdtemp(prefix="dl_")
    @after_this_request
    def cleanup(resp):
        shutil.rmtree(outdir, ignore_errors=True)
        return resp
    try:
        path = run_download(url, kind, outdir)
        if not path or not os.path.exists(path):
            return jsonify({"error": "Không tải được (video không khả dụng hoặc vượt giới hạn dung lượng)."}), 502
        if os.path.getsize(path) > MAX_DL_BYTES:
            return jsonify({"error": "File quá lớn so với giới hạn máy chủ."}), 413
        resp = send_file(path, as_attachment=True, download_name=os.path.basename(path))
        resp.headers["X-File-Name"] = base64.b64encode(os.path.basename(path).encode("utf-8")).decode("ascii")
        resp.headers["Access-Control-Expose-Headers"] = "X-File-Name"
        return resp
    except Exception as exc:
        logging.exception("Download failed")
        shutil.rmtree(outdir, ignore_errors=True)
        return jsonify({"error": "yt-dlp không tải được link này. Thử cập nhật yt-dlp hoặc link khác. Chi tiết: " + str(exc)[:200]}), 502

@app.post("/api/image")
def image():
    err = require_client()
    if err: return err
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt: return jsonify({"error": "Hãy nhập mô tả ảnh."}), 400
    if len(prompt) > 3000: return jsonify({"error": "Mô tả ảnh tối đa 3.000 ký tự."}), 413
    try:
        response, _ = generate_with_retry(
            IMAGE_MODEL, prompt,
            types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]), fallbacks=[])
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if getattr(part, "inline_data", None):
                blob = part.inline_data
                return jsonify({"image": base64.b64encode(blob.data).decode("ascii"),
                                "mimeType": blob.mime_type or "image/png"})
        return jsonify({"error": "Model không trả về ảnh. Kiểm tra model tạo ảnh và quyền API."}), 502
    except Exception as exc:
        logging.exception("Gemini image generation failed")
        return jsonify({"error": "Không tạo được ảnh: " + str(exc)[:350]}), 502

@app.errorhandler(429)
def rate_limited(_):
    return jsonify({"error": "Bạn gửi quá nhanh. Vui lòng đợi một phút rồi thử lại."}), 429

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Yêu cầu quá lớn. Giới hạn tải lên của máy chủ là 40 MB."}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
