"""Transcription pipeline: YouTube URL -> audio -> (optional piano isolation) -> MIDI -> MusicXML.

Runs as a subprocess of the web server. Progress is reported by rewriting
status.json inside the job directory after each stage.

Usage: python pipeline.py <url> <job_dir> [--separate]
"""

import json
import re
import shutil
import subprocess
import sys
import traceback
import urllib.parse
import urllib.request
from pathlib import Path

MAX_DURATION_SECONDS = 15 * 60
GRID = 0.25  # quantization grid in quarterLengths (a 16th note)
TREBLE_SPLIT = 60  # notes at/above middle C go to the treble staff


class Job:
    def __init__(self, job_dir: Path):
        self.dir = job_dir
        self.status = {
            "stage": "starting",
            "detail": "",
            "done": False,
            "error": None,
            "title": None,
            "thumbnail": None,
            "duration": None,
            "tempo": None,
            "files": {},
        }

    def update(self, **kwargs):
        self.status.update(kwargs)
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status))
        tmp.replace(self.dir / "status.json")

    def fail(self, message: str):
        self.update(stage="error", error=message, done=True)


def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "replace")


def resolve_spotify(url: str) -> str:
    """Spotify's audio can't be downloaded; return 'title artist' to find the
    same song on YouTube. Uses public no-auth endpoints (oEmbed + embed page)."""
    oembed = json.loads(_http_get(
        "https://open.spotify.com/oembed?url=" + urllib.parse.quote(url, safe="")))
    title = (oembed.get("title") or "").strip()
    artist = ""
    track = re.search(r"/track/([A-Za-z0-9]+)", url)
    if track:
        try:
            html = _http_get(f"https://open.spotify.com/embed/track/{track.group(1)}")
            m = re.search(r'"artists"\s*:\s*\[\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', html)
            if m:
                artist = m.group(1)
        except Exception:
            pass  # artist is a nice-to-have; title alone still searches fine
    if not title:
        raise ValueError("Couldn't read that Spotify link — make sure it's a track link.")
    return f"{title} {artist}".strip()


def normalize_source(job: Job, url: str) -> str:
    """Map non-downloadable sources onto something yt-dlp can fetch."""
    host = urllib.parse.urlparse(url).netloc.lower()
    if "spotify.com" in host:
        job.update(stage="downloading", detail="Reading the track from Spotify…")
        query = resolve_spotify(url)
        job.update(detail=f"Finding “{query}” on YouTube…")
        return f"ytsearch1:{query} audio"
    return url


def friendly_download_error(url: str, exc: Exception) -> str:
    host = urllib.parse.urlparse(url).netloc.lower()
    if "instagram" in host:
        return ("Instagram blocked the download — it often requires a login. "
                "Public Reels sometimes work on a retry; otherwise try the same "
                "song from YouTube or TikTok.")
    if "tiktok" in host:
        return ("TikTok download failed — the clip may be private, region-locked, "
                "or TikTok changed something. Try again in a moment.")
    detail = str(exc).strip().splitlines()[-1][:300]
    return f"Download failed: {detail}"


def download_audio(job: Job, url: str) -> Path:
    import yt_dlp

    url = normalize_source(job, url)
    job.update(stage="downloading", detail="Fetching the audio…")
    ffmpeg_dir = str(Path(shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg").parent)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(job.dir / "audio.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "ffmpeg_location": ffmpeg_dir,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist":  # search strings resolve to a playlist
                entries = info.get("entries") or []
                if not entries:
                    raise ValueError("No matching video found for that track.")
                info = entries[0]
            duration = info.get("duration") or 0
            if duration > MAX_DURATION_SECONDS:
                raise ValueError(
                    f"Video is {duration // 60} min long; the prototype caps at "
                    f"{MAX_DURATION_SECONDS // 60} min. Try a shorter video."
                )
            job.update(
                title=info.get("title"),
                thumbnail=info.get("thumbnail"),
                duration=duration,
            )
            ydl.download([info["webpage_url"]])
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(friendly_download_error(url, exc)) from exc

    wav = job.dir / "audio.wav"
    if not wav.exists():
        raise RuntimeError("Audio download failed (no wav produced).")
    return wav


def separate_piano(job: Job, wav: Path, stem: str = "piano") -> Path:
    """Isolate one instrument stem with Demucs (htdemucs_6s). Slow on CPU."""
    job.update(stage="separating", detail=f"Isolating the {stem} track (this is the slow part)…")
    out = job.dir / "separated"
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs_6s",
        "--two-stems", stem,
        "-o", str(out),
        str(wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-800:]}")
    stem_wav = out / "htdemucs_6s" / wav.stem / f"{stem}.wav"
    if not stem_wav.exists():
        raise RuntimeError(f"Demucs did not produce a {stem} stem.")
    return stem_wav


def estimate_tempo(wav: Path) -> int:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav), sr=22050, mono=True, duration=120)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0]) or 120.0
    # Fold extreme estimates back into a sane range
    while bpm > 180:
        bpm /= 2
    while bpm < 50:
        bpm *= 2
    return round(bpm)


def transcribe(job: Job, wav: Path, bpm: int, instrument: str = "piano") -> Path:
    """Piano: Transkun (best note-length/offset accuracy, F1 ≈ 0.90 with
    offsets vs ≈ 0.82 for ByteDance), then ByteDance high-res, then Basic
    Pitch. Guitar: the piano models are piano-only — Basic Pitch (instrument-
    agnostic) constrained to the guitar's frequency range."""
    mid = job.dir / "transcription.mid"
    if instrument == "guitar":
        job.update(stage="transcribing", detail="Listening for guitar notes (Basic Pitch)…")
        return transcribe_basic_pitch(job, wav, bpm, mid, instrument="guitar")
    job.update(
        stage="transcribing",
        detail="Listening closely (Transkun, best-accuracy piano model)…",
    )
    try:
        return transcribe_transkun(wav, mid)
    except Exception as exc:
        print(f"Transkun failed ({exc}); trying ByteDance high-res model.", file=sys.stderr)
    try:
        return transcribe_bytedance(job, wav, mid)
    except Exception as exc:
        print(f"High-res model failed ({exc}); falling back to Basic Pitch.", file=sys.stderr)
        return transcribe_basic_pitch(job, wav, bpm, mid)


def transcribe_transkun(wav: Path, mid: Path) -> Path:
    transkun_bin = Path(sys.executable).parent / "transkun"
    if not transkun_bin.exists():
        raise FileNotFoundError("transkun CLI not found in venv")
    # Run through the interpreter: the script's shebang bakes in an absolute
    # path that breaks if the project folder is ever renamed.
    result = subprocess.run(
        [sys.executable, str(transkun_bin), str(wav), str(mid), "--device", "cpu"],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"transkun exited {result.returncode}: {result.stderr[-400:]}")
    if not mid.exists() or mid.stat().st_size < 100:
        raise RuntimeError("transkun produced no notes")
    return mid


def transcribe_bytedance(job: Job, wav: Path, mid: Path) -> Path:
    import librosa
    from piano_transcription_inference import PianoTranscription, sample_rate

    job.update(stage="transcribing", detail="Listening closely (high-accuracy piano model)…")
    # The package's own load_audio uses librosa internals removed in 0.11
    audio, _ = librosa.load(str(wav), sr=sample_rate, mono=True)
    PianoTranscription(device="cpu").transcribe(audio, str(mid))
    if not mid.exists() or mid.stat().st_size < 100:
        raise RuntimeError("high-res model produced no notes")
    return mid


def transcribe_basic_pitch(job: Job, wav: Path, bpm: int, mid: Path,
                           instrument: str = "piano") -> Path:
    import scipy.signal
    if not hasattr(scipy.signal, "gaussian"):  # removed in scipy>=1.13; basic-pitch 0.3 still uses it
        scipy.signal.gaussian = scipy.signal.windows.gaussian

    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import predict

    # Prefer CoreML on macOS; the bundled TF SavedModel does not load on TF>=2.16.
    model_path = None
    for suffix in (FilenameSuffix.coreml, FilenameSuffix.onnx, FilenameSuffix.tf):
        candidate = build_icassp_2022_model_path(suffix)
        if candidate.exists():
            model_path = candidate
            break

    kwargs = {"midi_tempo": bpm}
    if instrument == "guitar":
        # Standard-tuning guitar spans E2 (~82 Hz) to ~E6; constraining the
        # range suppresses phantom bass rumble and cymbal-like overtones.
        # Shorter minimum length catches fast picking runs.
        kwargs.update(minimum_frequency=75.0, maximum_frequency=2100.0,
                      minimum_note_length=90.0)
    _, midi_data, _ = predict(str(wav), model_path, **kwargs)
    if instrument == "guitar":
        clean_guitar_notes(midi_data)
        for inst in midi_data.instruments:
            inst.program = 25  # General MIDI: acoustic steel guitar, for playback
    midi_data.write(str(mid))
    return mid


def clean_guitar_notes(midi_data) -> None:
    """Remove Basic Pitch's classic guitar artifacts in place.

    1. Harmonic ghosts: a weak note starting together with a stronger note an
       octave / twelfth / two octaves below is an overtone, not a played note.
    2. Noise floor: very quiet, very short notes are pick/string noise.
    """
    for inst in midi_data.instruments:
        notes = sorted(inst.notes, key=lambda n: n.start)
        keep = []
        for i, n in enumerate(notes):
            dur = n.end - n.start
            if n.velocity < 18 and dur < 0.09:
                continue
            ghost = False
            j = i - 1
            while j >= 0 and n.start - notes[j].start <= 0.035:
                j -= 1
            for m in notes[j + 1:]:
                if m.start - n.start > 0.035:
                    break
                if m is n:
                    continue
                if (n.pitch - m.pitch) in (12, 19, 24) and \
                        m.velocity >= n.velocity * 1.3 and \
                        (m.end - m.start) >= dur * 0.8:
                    ghost = True
                    break
            if not ghost:
                keep.append(n)
        inst.notes = keep


STRING_OPEN = [64, 59, 55, 50, 45, 40]  # standard tuning, string 1 (high E) → 6 (low E)
MAX_FRET = 20


def assign_frets(job: Job, mid: Path, bpm: int, capo: int = 0) -> tuple:
    """Assign each note a (string, fret) position, fret 0 = capo (or nut).

    Heuristic: notes sounding together go on distinct strings; prefer
    positions near the current hand position and lower frets; open strings
    are always cheap. Out-of-range pitches are octave-shifted into range.
    """
    import pretty_midi

    open_pitches = [p + capo for p in STRING_OPEN]
    max_fret = MAX_FRET - capo
    pm = pretty_midi.PrettyMIDI(str(mid))
    notes = sorted((n for i in pm.instruments for n in i.notes),
                   key=lambda n: (n.start, -n.pitch))

    # group into chords: notes within 50 ms share an onset
    chords, current = [], []
    for n in notes:
        if current and n.start - current[0].start > 0.05:
            chords.append(current)
            current = []
        current.append(n)
    if current:
        chords.append(current)

    hand = 3.0  # rolling estimate of hand position (fret above the capo)
    result = []
    lowest = open_pitches[5]
    highest = open_pitches[0] + max_fret
    for chord in chords:
        used = set()
        placed_frets = []
        for n in sorted(chord, key=lambda x: -x.pitch):
            pitch = n.pitch
            while pitch < lowest:
                pitch += 12
            while pitch > highest:
                pitch -= 12
            best, best_cost = None, None
            for s, open_pitch in enumerate(open_pitches):
                if s in used:
                    continue
                fret = pitch - open_pitch
                if not 0 <= fret <= max_fret:
                    continue
                cost = 0.0 if fret == 0 else abs(fret - hand) + fret * 0.12
                spans = [abs(fret - f) for f in placed_frets if f > 0 and fret > 0]
                if spans and max(spans) > 4:  # unplayable stretch within a chord
                    cost += (max(spans) - 4) * 3.0
                if best_cost is None or cost < best_cost:
                    best, best_cost = (s, fret), cost
            if best is None:
                continue  # more simultaneous notes than strings: drop the extra
            used.add(best[0])
            placed_frets.append(best[1])
            result.append({
                "start": round(n.start, 3),
                "end": round(n.end, 3),
                "pitch": pitch,
                "string": best[0] + 1,
                "fret": best[1],
            })
        fretted = [f for f in placed_frets if f > 0]
        if fretted:
            hand = 0.7 * hand + 0.3 * (sum(fretted) / len(fretted))

    result.sort(key=lambda x: x["start"])
    tabs = job.dir / "tabs.json"
    tabs.write_text(json.dumps({"tuning": "EADGBE", "bpm": bpm, "capo": capo,
                                "notes": result}))
    txt = job.dir / "tabs.txt"
    txt.write_text(tab_text(result, bpm, job.status.get("title") or "Transcription",
                            capo=capo))
    return tabs, txt


NOTE_NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]
SHARP_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_NAMES = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
# Krumhansl-Schmuckler key profiles
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
# sharps in the key signature for each major/minor tonic pitch class
MAJOR_SHARPS = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
MINOR_SHARPS = {0: -3, 1: 4, 2: -1, 3: 6, 4: 1, 5: -4, 6: 3, 7: -2, 8: 5, 9: 0, 10: -5, 11: 2}


def detect_keys(mid: Path, bpm: int) -> list:
    """Windowed Krumhansl-Schmuckler key finding with hysteresis.

    Returns [{"start": sec, "tonic": pc, "mode": "major"|"minor", "sharps": n}];
    a new entry appears only when a different key persists for two windows,
    so a real modulation registers but a chromatic phrase doesn't.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(mid))
    notes = [n for i in pm.instruments for n in i.notes]
    if not notes:
        return []
    window = 8 * 4 * 60.0 / bpm  # 8 bars of 4/4
    hop = window / 2
    song_end = max(n.end for n in notes)

    def correlate(weights, profile):
        n = 12
        mw, mp = sum(weights) / n, sum(profile) / n
        num = sum((weights[i] - mw) * (profile[i] - mp) for i in range(n))
        den = (sum((w - mw) ** 2 for w in weights) * sum((p - mp) ** 2 for p in profile)) ** 0.5
        return num / den if den else 0.0

    raw = []
    t = 0.0
    while t < song_end:
        w = [0.0] * 12
        for n in notes:
            overlap = min(n.end, t + window) - max(n.start, t)
            if overlap > 0:
                w[n.pitch % 12] += overlap
        if sum(w) > 1.0:
            best, best_r = None, -2
            for tonic in range(12):
                rotated = [w[(pc + tonic) % 12] for pc in range(12)]
                for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
                    r = correlate(rotated, profile)
                    if r > best_r:
                        best, best_r = (tonic, mode), r
            raw.append((t, best))
        t += hop

    segments = []
    current = None
    for i, (start, key) in enumerate(raw):
        if key == current:
            continue
        # hysteresis: a new key must hold for two consecutive windows
        if i + 1 < len(raw) and raw[i + 1][1] != key:
            continue
        current = key
        sharps = (MAJOR_SHARPS if key[1] == "major" else MINOR_SHARPS)[key[0]]
        segments.append({"start": round(start, 2), "tonic": key[0],
                         "mode": key[1], "sharps": sharps})
    return segments


def key_at(key_segments: list, t: float) -> dict:
    active = None
    for seg in key_segments:
        if seg["start"] <= t:
            active = seg
        else:
            break
    return active or {"tonic": 0, "mode": "major", "sharps": 0}


def spell(pc: int, key_segments: list, t: float) -> str:
    """Name a pitch class according to the key in force at time t."""
    return (FLAT_NAMES if key_at(key_segments, t)["sharps"] < 0 else SHARP_NAMES)[pc]


def detect_capo(mid: Path) -> int:
    """Guess a capo position: if the pitch floor is raised and the music keeps
    landing on what would be capo'd open strings, the player is likely fretted
    up. Returns 0 (no capo) unless the evidence is clearly better than open."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(mid))
    notes = [n for i in pm.instruments for n in i.notes]
    if not notes:
        return 0
    lowest = min(n.pitch for n in notes)
    max_capo = min(7, lowest - STRING_OPEN[5])
    if max_capo <= 0:
        return 0
    ring_frac, scores = {}, {}
    total = sum(n.end - n.start for n in notes)
    for c in range(0, max_capo + 1):
        opens = {p + c for p in STRING_OPEN}
        ring_frac[c] = sum((n.end - n.start) for n in notes if n.pitch in opens) / total
        scores[c] = ring_frac[c] - c * 0.015  # slight preference for lower capo
    best = max(scores, key=scores.get)
    # claim a capo only on real evidence: lots of open-string ringing that
    # clearly beats the no-capo reading — barre-heavy songs stay capo 0
    if best > 0 and ring_frac[best] >= 0.12 and scores[best] > scores[0] * 1.3 + 0.02:
        return best
    return 0
CHORD_TEMPLATES = {  # quality suffix -> pitch classes relative to root
    "": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10),
    "maj7": (0, 4, 7, 11), "sus4": (0, 5, 7), "sus2": (0, 2, 7),
}
OPEN_SHAPES = {  # common open-position shapes; frets low E → high e, -1 = mute
    "C": [-1, 3, 2, 0, 1, 0], "A": [-1, 0, 2, 2, 2, 0], "G": [3, 2, 0, 0, 0, 3],
    "E": [0, 2, 2, 1, 0, 0], "D": [-1, -1, 0, 2, 3, 2], "F": [-1, -1, 3, 2, 1, 1],
    "Am": [-1, 0, 2, 2, 1, 0], "Em": [0, 2, 2, 0, 0, 0], "Dm": [-1, -1, 0, 2, 3, 1],
    "A7": [-1, 0, 2, 0, 2, 0], "B7": [-1, 2, 1, 2, 0, 2], "C7": [-1, 3, 2, 3, 1, 0],
    "D7": [-1, -1, 0, 2, 1, 2], "E7": [0, 2, 0, 1, 0, 0], "G7": [3, 2, 0, 0, 0, 1],
    "Am7": [-1, 0, 2, 0, 1, 0], "Em7": [0, 2, 0, 0, 0, 0], "Dm7": [-1, -1, 0, 2, 1, 1],
    "Cmaj7": [-1, 3, 2, 0, 0, 0], "Amaj7": [-1, 0, 2, 1, 2, 0], "Dmaj7": [-1, -1, 0, 2, 2, 2],
    "Fmaj7": [-1, -1, 3, 2, 1, 0], "Gmaj7": [3, 2, 0, 0, 0, 2],
    "Asus2": [-1, 0, 2, 2, 0, 0], "Dsus2": [-1, -1, 0, 2, 3, 0],
    "Asus4": [-1, 0, 2, 2, 3, 0], "Dsus4": [-1, -1, 0, 2, 3, 3], "Esus4": [0, 2, 2, 2, 0, 0],
}
E_BARRE = {"": [0, 2, 2, 1, 0, 0], "m": [0, 2, 2, 0, 0, 0], "7": [0, 2, 0, 1, 0, 0],
           "m7": [0, 2, 0, 0, 0, 0], "maj7": [0, 2, 1, 1, 0, 0], "sus4": [0, 2, 2, 2, 0, 0]}
A_BARRE = {"": [-1, 0, 2, 2, 2, 0], "m": [-1, 0, 2, 2, 1, 0], "7": [-1, 0, 2, 0, 2, 0],
           "m7": [-1, 0, 2, 0, 1, 0], "maj7": [-1, 0, 2, 1, 2, 0],
           "sus2": [-1, 0, 2, 2, 0, 0], "sus4": [-1, 0, 2, 2, 3, 0]}


def chord_shape(root_pc: int, quality: str) -> dict:
    """Return a playable diagram for the chord: frets (low E→high e), base fret."""
    name = NOTE_NAMES[root_pc] + quality
    plain = name.replace("♯", "#").replace("♭", "b")
    for key in (name, plain):
        if key in OPEN_SHAPES:
            return {"frets": OPEN_SHAPES[key], "base": 1}
    # movable barre shapes: root on string 6 (E shape) or string 5 (A shape)
    candidates = []
    fe = (root_pc - 4) % 12
    if quality in E_BARRE and fe > 0:
        candidates.append((fe, E_BARRE[quality]))
    fa = (root_pc - 9) % 12
    if quality in A_BARRE and fa > 0:
        candidates.append((fa, A_BARRE[quality]))
    if not candidates:  # last resort: root-position power chord, E shape
        base = max(1, (root_pc - 4) % 12)
        return {"frets": [base, base + 2, base + 2, -1, -1, -1], "base": base}
    base, shape = min(candidates, key=lambda c: c[0])
    return {"frets": [f + base if f >= 0 else -1 for f in shape], "base": base}


def assign_fingers(frets: list, base: int) -> list:
    """Songbook-style finger numbers (1=index … 4=pinky); 0 for open/mute."""
    fingers = [0] * 6
    fretted = [(f, s) for s, f in enumerate(frets) if f > 0]
    if not fretted:
        return fingers
    min_fret = min(f for f, _ in fretted)
    barre_strings = [s for f, s in fretted if f == min_fret]
    next_finger = 1
    # index-finger barre: any movable shape, or several strings on the 1st fret (F, B♭…)
    if (base > 1 or min_fret == 1) and len(barre_strings) >= 2:
        for s in barre_strings:
            fingers[s] = 1
        next_finger = 2
        fretted = [(f, s) for f, s in fretted if f != min_fret]
    for f, s in sorted(fretted):
        fingers[s] = min(4, next_finger)
        next_finger += 1
    return fingers


def detect_chords(job: Job, mid: Path, bpm: int, capo: int = 0,
                  key_segments: list | None = None) -> Path:
    """Segment the song and label each segment with the best-matching chord.

    Chord names are spelled for the key in force (F♯ in sharp keys, G♭ in
    flat keys). With a capo, diagrams show the shape you actually grab
    (sounding B♭ with capo 3 = a G shape).
    """
    import pretty_midi

    key_segments = key_segments or []

    pm = pretty_midi.PrettyMIDI(str(mid))
    notes = [n for i in pm.instruments for n in i.notes]
    seg_len = 2 * 60.0 / bpm  # half a bar of 4/4 per segment
    n_segs = int(max(n.end for n in notes) / seg_len) + 1 if notes else 0

    weights = [dict() for _ in range(n_segs)]  # pitch class -> summed duration
    bass = [None] * n_segs                     # lowest pitch heard in the segment
    for n in notes:
        first, last = int(n.start / seg_len), int(n.end / seg_len)
        for i in range(first, min(last, n_segs - 1) + 1):
            lo, hi = i * seg_len, (i + 1) * seg_len
            overlap = min(n.end, hi) - max(n.start, lo)
            if overlap <= 0.02:
                continue
            pc = n.pitch % 12
            weights[i][pc] = weights[i].get(pc, 0.0) + overlap
            if bass[i] is None or n.pitch < bass[i]:
                bass[i] = n.pitch

    labels = [None] * n_segs  # (root, quality) per segment
    for i, w in enumerate(weights):
        total = sum(w.values())
        if total < seg_len * 0.5:  # mostly silence
            continue
        best, best_score = None, 0.0
        for root in range(12):
            for quality, template in CHORD_TEMPLATES.items():
                pcs = {(root + t) % 12 for t in template}
                hit = sum(v for pc, v in w.items() if pc in pcs)
                miss = total - hit
                score = hit - 0.8 * miss
                if bass[i] is not None and bass[i] % 12 == root:
                    score *= 1.25  # bass note on the root is strong evidence
                if score > best_score:
                    best, best_score = (root, quality), score
        if best is not None and best_score >= total * 0.45:
            labels[i] = best

    # melody notes flash false chords for half a bar; absorb single-segment
    # islands whose neighbours agree with each other
    for i in range(1, n_segs - 1):
        if (labels[i - 1] is not None and labels[i - 1] == labels[i + 1]
                and labels[i] != labels[i - 1]):
            labels[i] = labels[i - 1]

    sequence = []
    for i, label in enumerate(labels):
        if label is None:
            continue
        t = i * seg_len
        name = spell(label[0], key_segments, t) + label[1]
        if sequence and sequence[-1]["name"] == name:
            continue
        sequence.append({"name": name, "root": label[0], "quality": label[1],
                         "start": round(t, 2)})

    counts = {}
    for item in sequence:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
    # the chord sheet shows the chords that carry the song: recurring ones,
    # capped at 12 and ordered by how often they come around
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    keep = {n for n, c in ranked[:12] if c >= 2} or {n for n, _ in ranked[:6]}

    by_name = {}
    for item in sequence:
        if item["name"] in by_name or item["name"] not in keep:
            continue
        shape_root = (item["root"] - capo) % 12
        shape = chord_shape(shape_root, item["quality"])
        entry = {
            "name": item["name"],
            "frets": shape["frets"],
            "base": shape["base"],
            "fingers": assign_fingers(shape["frets"], shape["base"]),
            "count": counts[item["name"]],
            "first": item["start"],
        }
        if capo:
            entry["shape"] = spell(shape_root, key_segments, item["start"]) + item["quality"]
        by_name[item["name"]] = entry
    chords = sorted(by_name.values(), key=lambda c: -c["count"])

    out = job.dir / "chords.json"
    out.write_text(json.dumps({
        "chords": chords,
        "capo": capo,
        "sequence": [{"name": s["name"], "start": s["start"]} for s in sequence],
    }))
    return out


def tab_text(tab_notes: list, bpm: int, title: str, capo: int = 0) -> str:
    """Render classic monospace tablature on a 16th-note grid, 4 bars per line."""
    sec16 = 60.0 / bpm / 4.0
    slots = {}  # (string, slot) -> fret
    last_slot = 0
    for n in tab_notes:
        slot = round(n["start"] / sec16)
        last_slot = max(last_slot, slot)
        slots.setdefault((n["string"], slot), n["fret"])

    labels = "eBGDAE"
    measures = last_slot // 16 + 1
    tuning = "standard tuning" + (f" · CAPO {capo}" if capo else "")
    out = [title, f"~{bpm} BPM · {tuning} · 16th-note grid", ""]
    for block_start in range(0, measures, 4):
        block = range(block_start, min(block_start + 4, measures))
        for s in range(1, 7):
            row = [labels[s - 1], "|"]
            for m in block:
                for slot in range(m * 16, (m + 1) * 16):
                    cell = slots.get((s, slot))
                    row.append("--" if cell is None else str(cell).ljust(2, "-"))
                row.append("|")
            out.append("".join(row))
        out.append("")
    return "\n".join(out)


def make_performance_midi(job: Job, mid: Path) -> Path:
    """Bake sustain-pedal (CC64) into note durations.

    The transcription models detect the pedal, but most MIDI players (including
    the in-browser one) ignore pedal events — notes cut off at key release and
    the piece sounds dry and choppy. Extending each note to the pedal release
    (or the next strike of the same key) makes playback match what you hear.
    """
    import bisect

    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(mid))
    song_end = pm.get_end_time() + 0.1
    for inst in pm.instruments:
        pedal = sorted((c for c in inst.control_changes if c.number == 64),
                       key=lambda c: c.time)
        intervals = []
        down = None
        for c in pedal:
            if c.value >= 64 and down is None:
                down = c.time
            elif c.value < 64 and down is not None:
                intervals.append((down, c.time))
                down = None
        if down is not None:
            intervals.append((down, song_end))
        if not intervals:
            continue

        onsets_by_pitch = {}
        for n in inst.notes:
            onsets_by_pitch.setdefault(n.pitch, []).append(n.start)
        for starts in onsets_by_pitch.values():
            starts.sort()

        for n in inst.notes:
            for d, u in intervals:
                if d <= n.end < u:  # key released while pedal held
                    new_end = u
                    same = onsets_by_pitch[n.pitch]
                    i = bisect.bisect_right(same, n.start + 1e-4)
                    if i < len(same):  # same key struck again: damper resets
                        new_end = min(new_end, same[i])
                    # A real string decays; the sampled piano doesn't. Cap the
                    # ring so long pedal holds don't drone.
                    new_end = min(new_end, n.end + 3.0)
                    n.end = max(n.end, new_end)
                    break

    out = job.dir / "performance.mid"
    pm.write(str(out))
    return out


def build_score(job: Job, mid: Path, bpm: int, instrument_name: str = "piano",
                key_segments: list | None = None) -> Path:
    """Quantize the raw MIDI onto a 16th grid and engrave the score.

    Piano gets a grand staff split at middle C; guitar gets the conventional
    single treble staff sounding an octave lower (Treble8vb clef).
    """
    import pretty_midi
    from music21 import (chord, clef, instrument, key, layout, metadata,
                         meter, note, stream, tempo)

    job.update(stage="engraving", detail="Quantizing and engraving the score…")
    pm = pretty_midi.PrettyMIDI(str(mid))
    sec_per_16th = 60.0 / bpm / 4.0
    guitar = instrument_name == "guitar"

    # staff -> onset grid index -> list of (pitch, length in grid steps)
    staves = {"treble": {}} if guitar else {"treble": {}, "bass": {}}
    for inst in pm.instruments:
        for n in inst.notes:
            start = round(n.start / sec_per_16th)
            end = max(start + 1, round(n.end / sec_per_16th))
            staff = "treble" if guitar or n.pitch >= TREBLE_SPLIT else "bass"
            staves[staff].setdefault(start, []).append((n.pitch, end - start))

    score = stream.Score()
    score.metadata = metadata.Metadata(title=job.status.get("title") or "Transcription")
    score.metadata.composer = "trans. Adam Cross"

    if guitar:
        staff_defs = (("treble", clef.Treble8vbClef()),)
        m21_instrument = instrument.Guitar
    else:
        staff_defs = (("treble", clef.TrebleClef()), ("bass", clef.BassClef()))
        m21_instrument = instrument.Piano

    parts = {}
    for name, clef_obj in staff_defs:
        part = stream.Part(id=name)
        part.insert(0, m21_instrument())
        part.insert(0, clef_obj)
        part.insert(0, meter.TimeSignature("4/4"))
        if name == "treble":
            part.insert(0, tempo.MetronomeMark(number=bpm))
        parts[name] = part

    for name, onsets in staves.items():
        part = parts[name]
        starts = sorted(onsets)
        for i, start in enumerate(starts):
            notes_here = onsets[start]
            longest = max(length for _, length in notes_here)
            # Piano-reduction style: cut each chord off when the next one lands
            # so every staff stays a single clean voice.
            if i + 1 < len(starts):
                longest = min(longest, starts[i + 1] - start)
            longest = min(longest, 16)  # cap at a whole note
            ql = longest * GRID
            pitches = sorted({p for p, _ in notes_here})
            element = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            element.quarterLength = ql
            part.insert(start * GRID, element)
        if not starts:
            part.insert(0, note.Rest(quarterLength=4.0))

    score.insert(0, parts["treble"])
    if not guitar:
        score.insert(0, parts["bass"])
        score.insert(0, layout.StaffGroup([parts["treble"], parts["bass"]], symbol="brace"))

    if key_segments:
        # honour modulations: a fresh signature wherever the key changes,
        # snapped to the nearest bar line
        prev_sharps = None
        for seg in key_segments:
            if seg["sharps"] == prev_sharps:
                continue
            prev_sharps = seg["sharps"]
            offset = round(seg["start"] * bpm / 60.0 / 4.0) * 4.0
            for part in parts.values():
                part.insert(offset, key.KeySignature(seg["sharps"]))
    else:
        try:
            detected = score.analyze("key")
            for part in parts.values():
                part.insert(0, key.KeySignature(detected.sharps))
        except Exception:
            pass  # key detection is best-effort

    xml = job.dir / "score.musicxml"
    score.write("musicxml", fp=str(xml))
    return xml


def main():
    url = sys.argv[1]
    job_dir = Path(sys.argv[2])
    extra = sys.argv[3:]
    separate = "--separate" in extra
    instrument = "guitar" if ("--instrument" in extra and
                              extra[extra.index("--instrument") + 1] == "guitar") else "piano"
    job = Job(job_dir)
    try:
        job.update(stage="starting", instrument=instrument)
        wav = download_audio(job, url)
        source = separate_piano(job, wav, stem=instrument) if separate else wav
        bpm = estimate_tempo(source)
        job.update(tempo=bpm)
        mid = transcribe(job, source, bpm, instrument=instrument)
        keys = detect_keys(mid, bpm)
        main_key = keys[0] if keys else None
        job.update(keys=[{"name": spell(k["tonic"], keys, k["start"]) + (" minor" if k["mode"] == "minor" else ""),
                          "start": k["start"]} for k in keys])
        perf = make_performance_midi(job, mid)
        # Engrave from the pedal-applied version: the quantizer caps every note
        # at the next chord anyway, so sustained notes come out as legato
        # phrases instead of staccato fragments followed by rests.
        xml = build_score(job, perf, bpm, instrument_name=instrument, key_segments=keys)
        files = {
            "midi": mid.name,
            "performance": perf.name,
            "musicxml": xml.name,
            "audio": "audio.wav",
        }
        if instrument == "guitar":
            capo = detect_capo(mid)
            job.update(capo=capo)
            tabs, txt = assign_frets(job, mid, bpm, capo=capo)
            files["tabs"] = tabs.name
            files["tabtxt"] = txt.name
            files["chords"] = detect_chords(job, mid, bpm, capo=capo,
                                            key_segments=keys).name
        job.update(stage="done", detail="", done=True, files=files)
    except Exception as exc:
        traceback.print_exc()
        job.fail(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
