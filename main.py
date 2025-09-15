import os
import uuid
import datetime
import subprocess
from typing import List
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from supabase import create_client, Client
import stripe

load_dotenv()

# ---- CORS (liberar acesso do seu site) ----
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- IA de transcrição (modelo leve) ----
model = WhisperModel("base", device="cpu", compute_type="int8")

# ---- Supabase (banco) ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# ---- Stripe (pagamentos) ----
stripe.api_key = os.getenv("STRIPE_SECRET")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# ---- helpers ----
def ensure_temp():
    os.makedirs("temp", exist_ok=True)

def save_temp_file(upload: UploadFile, ext: str) -> str:
    ensure_temp()
    name = f"{uuid.uuid4()}{ext}"
    path = os.path.join("temp", name)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path

def extract_audio(video_path: str) -> str:
    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path], check=True)
    return audio_path

def transcribe_segments(audio_path: str):
    segments, _ = model.transcribe(audio_path, beam_size=5)
    return list(segments)

def ts(seconds: float) -> str:
    """Transforma segundos em string tipo 00:00:05,123"""
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(segments, srt_path: str):
    Path(srt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, start=1):
            start = ts(s.start)
            end = ts(s.end)
            text = s.text.strip().replace("\n", " ")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def burn_subs(video_path: str, srt_path: str) -> str:
    # criar caminho de saída
    base = Path(video_path)
    out_path = str(base.with_name(base.stem + "_final.mp4"))

    # converter caminhos para formato que o FFmpeg aceite melhor
    video_posix = Path(video_path).as_posix()
    srt_posix = Path(srt_path).as_posix()
    out_posix = Path(out_path).as_posix()

    # verificar se o arquivo SRT existe
    if not os.path.exists(srt_path):
        raise RuntimeError(f"SRT não encontrado: {srt_path}")

    vf = f"subtitles={srt_posix}:force_style='FontName=Arial,FontSize=40,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3'"

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_posix, "-vf", vf, "-c:a", "copy", out_posix],
            check=True
        )
    except subprocess.CalledProcessError:
        # fallback se o estilo der problema
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_posix, "-vf", f"subtitles={srt_posix}", "-c:a", "copy", out_posix],
            check=True
        )

    return out_path

# ----- regras de plano -----
def limit_for(plan: str) -> int:
    return {"gratuito": 3, "criador": 30, "pro": 9999}.get(plan, 0)

def get_or_create_user(email: str) -> str:
    if not supabase:
        return "gratuito"
    r = supabase.table("usuarios").select("*").eq("email", email).execute()
    if len(r.data) == 0:
        supabase.table("usuarios").insert({"email": email, "plano": "gratuito"}).execute()
        return "gratuito"
    return r.data[0]["plano"]

def count_uses_this_month(email: str) -> int:
    if not supabase:
        return 0
    today = datetime.datetime.utcnow()
    start = datetime.datetime(today.year, today.month, 1)
    r = supabase.table("usos").select("*") \
        .eq("email", email) \
        .gte("data", start.isoformat()) \
        .execute()
    return len(r.data)

def add_use(email: str, video_name: str, plan: str):
    if supabase:
        supabase.table("usos").insert({"email": email, "video_nome": video_name, "plano": plan}).execute()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/full_process")
def full_process(file: UploadFile = File(...), email: str = Form(...)):
    if not email or "@" not in email:
        raise HTTPException(400, "Informe um e-mail válido.")

    plan = get_or_create_user(email)
    used = count_uses_this_month(email)
    limit = limit_for(plan)
    if used >= limit:
        raise HTTPException(403, f"Limite do plano '{plan}' atingido ({limit}/mês). Faça upgrade.")

    video_path = save_temp_file(file, ".mp4")
    audio_path = extract_audio(video_path)
    segs = transcribe_segments(audio_path)

    # imprimir debug para ver o que o modelo captou
    print("Segmentos de texto:", [s.text for s in segs])

    srt_path = video_path.rsplit(".", 1)[0] + ".srt"
    write_srt(segs, srt_path)
    final_path = burn_subs(video_path, srt_path)

    add_use(email, os.path.basename(video_path), plan)

    headers = {"Content-Disposition": f'attachment; filename="video_legenda.mp4"'}
    return StreamingResponse(open(final_path, "rb"), media_type="video/mp4", headers=headers)

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    if not STRIPE_WEBHOOK_SECRET or not supabase:
        return {"status": "ignored"}
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(400, "Webhook inválido")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        email = session["customer_details"]["email"]
        supabase.table("usuarios").update({"plano": "criador"}).eq("email", email).execute()
    return {"status": "ok"}
