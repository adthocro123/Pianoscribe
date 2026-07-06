# PianoScribe 🎹

Paste a YouTube link → get piano **sheet music** and an interactive **piano roll**.

Works in any browser (phone, laptop, PC) — the heavy lifting runs on the Python backend.

## How it works

```
YouTube URL ──yt-dlp──▶ audio.wav ──(optional) Demucs──▶ piano stem
                                          │
                                          ▼
        Transkun (best note-length accuracy; fallbacks: ByteDance
                    high-res model, then Basic Pitch)
                                          │
                                          ▼
        sustain pedal baked into note durations (performance.mid) —
        players ignore MIDI pedal events, so holds/releases sound
        wrong without this; playback + piano roll use this version
                                          │
                                          ▼
              transcription.mid ──quantize + music21──▶ score.musicxml
                                          │
                          browser: OpenSheetMusicDisplay (sheet)
                                   custom canvas piano roll (Synthesia-style
                                   falling notes onto an 88-key keyboard),
                                   html-midi-player for audio playback/seek
```

- **Isolate piano** checkbox: runs Demucs `htdemucs_6s` source separation first.
  Use it for full-band songs with vocals/drums; skip it for solo piano videos (much faster).
- Notes are quantized to a 16th-note grid at the detected tempo and split at middle C
  into a grand staff (treble/bass).

## Features

- **Library** — every transcription is saved and reopens with one click; hover a card to delete.
- **Practice tools** (piano roll): playback speed 50–150% (pitch unchanged), A–B section
  looping, adjustable lookahead, space bar = play/pause.
- **Sheet tools**: zoom, MusicXML/MIDI downloads, Print/PDF via a paginated A4 print view.
- **Guitar mode**: animated fretboard (string-colored fingering dots with fret numbers,
  approach rings for upcoming notes) plus classic text tablature with a .txt download.
  Fret assignment keeps chords on distinct strings and the hand position stable.
- Paste a URL anywhere on the page to start transcribing instantly.
- Web-app manifest: "Add to Home Screen" on a phone for an app-like experience.

## Run it

```bash
.venv/bin/python -m uvicorn server:app --port 8321
# open http://localhost:8321
```

First-time setup was: `python3.12 -m venv .venv` then
`pip install -r requirements.txt` (plus `brew install ffmpeg`).

## Prototype limits

- Sources: YouTube and TikTok download directly; Spotify links resolve to the same song
  on YouTube (Spotify audio cannot be downloaded — "isolate piano" auto-enables since
  those are full mixes); Instagram usually requires a login and fails gracefully.
- Videos capped at 15 minutes.
- Transcription is an approximation — solo piano recordings come out much cleaner than
  full mixes. Expect to tidy the score in MuseScore for anything complex.
- Downloading YouTube audio is against YouTube's ToS; keep this for personal use.
