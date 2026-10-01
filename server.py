"""스피치 발음 카드 — 로컬 백엔드

- /api/tts    : Kokoro(오픈소스 신경망 TTS)로 미국식/영국식 음성 생성
- /api/score  : faster-whisper(로컬 Whisper)로 녹음을 받아 적고 원문과 비교해 점수 계산
- /           : index.html 제공 (마이크는 localhost에서만 허용되므로 이 주소로 열어야 함)

실행:  .venv/bin/python server.py   →  http://localhost:8000
"""
import io
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small.en")
VOICES = {"en-US": ("af_heart", "en-us"), "en-GB": ("bf_emma", "en-gb")}
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")

@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Speech Pronunciation Card", lifespan=lifespan)
status ={"tts": "loading", "asr": "loading", "tts_error": None, "asr_error": None}
kokoro = None
whisper = None
tts_lock = threading.Lock()
asr_lock = threading.Lock()


def load_models():
    """서버 시작을 막지 않도록 모델은 백그라운드에서 불러온다."""
    global kokoro, whisper
    try:
        from kokoro_onnx import Kokoro
        kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
        status["tts"] = "ready"
    except Exception as e:  # 모델 파일 누락 등
        status["tts"], status["tts_error"] = "error", str(e)
    try:
        from faster_whisper import WhisperModel
        whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        status["asr"] = "ready"
    except Exception as e:
        status["asr"], status["asr_error"] = "error", str(e)


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {**status, "voices": {k: v[0] for k, v in VOICES.items()}, "whisper": WHISPER_MODEL}


# ---------------------------------------------------------------- TTS
class TTSReq(BaseModel):
    text: str
    accent: str = "en-US"
    slow: bool = False


tts_cache: "OrderedDict[tuple, bytes]" = OrderedDict()


@app.post("/api/tts")
def tts(req: TTSReq):
    if status["tts"] != "ready":
        raise HTTPException(503, f"음성 모델 상태: {status['tts']}")
    text = req.text.strip()[:1000]
    if not text:
        raise HTTPException(400, "읽을 문장이 비어 있어요.")
    voice, lang = VOICES.get(req.accent, VOICES["en-US"])
    speed = 0.5 if req.slow else 1.0
    key = (text, voice, speed)
    if key not in tts_cache:
        with tts_lock:
            samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
        tts_cache[key] = buf.getvalue()
        if len(tts_cache) > 200:
            tts_cache.popitem(last=False)
    tts_cache.move_to_end(key)
    return Response(tts_cache[key], media_type="audio/wav")


# ---------------------------------------------------------------- 채점
NUMS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def norm(w: str) -> str:
    w = w.strip().lower().replace("’", "'").strip(".,!?;:\"'()-")
    return NUMS[int(w)] if w.isdigit() and int(w) <= 20 else w


def align(target: list[str], heard: list[str]):
    """편집 거리로 원문 단어와 들린 단어를 짝지음. 비슷한 철자(ratio≥0.75)는 대체 비용을 낮춘다."""
    n, m = len(target), len(heard)
    sim = [[SequenceMatcher(None, t, h).ratio() for h in heard] for t in target]
    cost = lambda i, j: 0 if target[i] == heard[j] else (0.5 if sim[i][j] >= 0.75 else 1.2)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost(i - 1, j - 1))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + cost(i - 1, j - 1):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1], sim


def decode_audio(data: bytes) -> np.ndarray:
    """브라우저 녹음(webm/opus, mp4/aac, wav)을 Whisper 입력인 16kHz 모노 float32로 변환."""
    import av
    chunks = []
    with av.open(io.BytesIO(data)) as box:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in box.decode(audio=0):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("empty audio")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def prob_score(p: float) -> int:
    # Whisper 단어 확률: 또렷한 발음은 보통 0.85 이상, 애매하면 0.5 아래로 떨어짐
    return int(round(np.clip((p - 0.3) / 0.6, 0, 1) * 100))


@app.post("/api/score")
async def score(audio: UploadFile = File(...), text: str = Form(...)):
    if status["asr"] != "ready":
        raise HTTPException(503, f"인식 모델 상태: {status['asr']}")
    data = await audio.read()
    if len(data) < 2000:
        raise HTTPException(400, "녹음이 너무 짧아요. 버튼을 누르고 문장을 끝까지 읽어 주세요.")
    t0 = time.time()
    try:
        pcm = decode_audio(data)
    except Exception:
        raise HTTPException(400, "녹음 파일을 읽지 못했어요. 다시 녹음해 주세요.")
    with asr_lock:
        segs, info = whisper.transcribe(pcm, language="en", word_timestamps=True,
                                        beam_size=5, condition_on_previous_text=False, vad_filter=True)
        words = [w for s in segs for w in (s.words or [])]

    target_raw = WORD_RE.findall(text)
    target = [norm(w) for w in target_raw]
    heard = [norm(w.word) for w in words]
    keep = [k for k, h in enumerate(heard) if h]
    words, heard = [words[k] for k in keep], [heard[k] for k in keep]
    if not target:
        raise HTTPException(400, "채점할 영어 문장이 없어요.")

    pairs, sim = align(target, heard)
    out = [None] * len(target)
    extra = []
    for ti, hi in pairs:
        if ti is None:
            extra.append(words[hi].word.strip())
        elif hi is None:
            out[ti] = {"word": target_raw[ti], "status": "missed", "score": 0, "heard": None}
        else:
            w = words[hi]
            if target[ti] == heard[hi]:
                sc = prob_score(w.probability)
                st = "good" if sc >= 80 else "fair" if sc >= 60 else "poor"
            else:  # 다른 단어로 들림
                sc = int(round(min(sim[ti][hi], 0.9) * 50))
                st = "wrong"
            out[ti] = {"word": target_raw[ti], "status": st, "score": sc, "heard": w.word.strip(),
                       "prob": round(w.probability, 3), "start": round(w.start, 2), "end": round(w.end, 2)}

    accuracy = float(np.mean([o["score"] for o in out]))
    completeness = 100 * sum(o["status"] != "missed" for o in out) / len(out)
    # 유창성: 문장 중간의 긴 멈춤(0.6초↑)과 말하기 속도로 감점
    timed = [o for o in out if o.get("start") is not None]
    pauses = sum(1 for a, b in zip(timed, timed[1:]) if b["start"] - a["end"] > 0.6)
    span = (timed[-1]["end"] - timed[0]["start"]) if len(timed) > 1 else 0
    wpm = len(timed) / span * 60 if span > 0 else 0
    rate_pen = 0 if wpm == 0 or wpm >= 90 else (90 - wpm) * 0.6
    fluency = float(np.clip(100 - pauses * 10 - rate_pen - len(extra) * 5, 0, 100))
    if len(target) == 1:  # 단어 하나는 유창성을 따지지 않음
        fluency = accuracy
    overall = 0.6 * accuracy + 0.2 * completeness + 0.2 * fluency

    return {
        "overall": round(overall), "accuracy": round(accuracy), "completeness": round(completeness),
        "fluency": round(fluency), "words": out, "extra": extra,
        "transcript": " ".join(w.word.strip() for w in words), "wpm": round(wpm), "pauses": pauses,
        "seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
