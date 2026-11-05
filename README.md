# F.R.I.D.A.Y. — local desktop assistant

Runs on your machine. No API keys, no accounts, no subscription. Wake it with
**"Hey Friday"**, or type commands if you'd rather skip voice entirely.

## What it can actually do
- Open and close apps on your system ("open Chrome", "close Spotify")
- Open files and folders ("open ~/Documents/report.pdf")
- Run a shell command and read back the result ("run command: dir")
- Open websites and run web searches
- Quick math, time, date
- Take and save notes, set countdown timers
- Lock your screen, take a screenshot
- Chat naturally with a local Ollama model, and ask it to create new text
  files or folders. It will not overwrite or delete files. Notes saved with
  "remember that ..." can personalize future chats.
- Generate images locally from text prompts with Stable Diffusion Turbo.

## 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Notes per OS:
- **Windows**: if `pip install pyaudio` fails, run `pip install pipwin` then
  `pipwin install pyaudio`.
- **macOS**: install PortAudio first — `brew install portaudio` — then
  `pip install pyaudio`.
- **Linux**: install PortAudio's dev headers first —
  `sudo apt install portaudio19-dev python3-pyaudio` (Debian/Ubuntu) — then
  `pip install pyaudio`.

If you don't want microphone input, skip `pyaudio` and `SpeechRecognition` and
just run in text mode (see below). Voice and UI modes load Silero TTS in the
background during startup; its model is downloaded by PyTorch Hub on the first
run and cached locally. Text-only mode skips this warm-up. `pyttsx3` remains
available as a fallback.

## 2. Run it

On Garuda or Arch Linux, install Ollama and start its service:

```bash
sudo pacman -S ollama
sudo systemctl enable --now ollama.service
```

On other systems, install Ollama from <https://ollama.com/download> and start
the Ollama app or server. Download a free local model once. This does not
require an Ollama account:

```bash
ollama pull llama3.2:3b
```

For screen inspection, also download a vision-capable Ollama model:

```bash
ollama pull moondream
```

Keep the Ollama app running (or start its server with `ollama serve`), then run
Friday. It uses `llama3.2:3b` by default. To use another model already pulled,
set `OLLAMA_MODEL` before starting Friday, for example
`OLLAMA_MODEL=qwen2.5:3b python friday.py --text`.
For an Ollama server on another host or port, set `OLLAMA_HOST`, for example
`OLLAMA_HOST=http://192.168.1.20:11434 python friday.py --text`.

Image creation uses Diffusers and PyTorch, not Ollama. The first image request
downloads the default `stabilityai/sd-turbo` model to the local Hugging Face
cache; later requests reuse it. A CUDA GPU is recommended, while CPU generation
is supported but slower. Ask Friday to "create an image of a red fox in a snowy
forest". Images are saved as PNG files in `FRIDAY Images` beside `friday.py`.
To select another compatible Diffusers text-to-image model, set
`FRIDAY_IMAGE_MODEL` before starting Friday.

Ollama chat and file/folder creation run on your computer. Friday saves the
most recent 20 chat exchanges locally and restores them on the next launch.
New file and folder paths must be under your home folder or the directory
where Friday was started; existing files are never overwritten.

FRIDAY can prepare a source-code improvement proposal using its selected local
model. Name one top-level function or class method, for example
`propose source edit Friday.model_for_task: improve coding model selection`.
FRIDAY shows a diff and does not change any file until you enter the exact
`approve source edit <ID>` command. Use `review source edit` to show the pending
diff again, or `reject source edit <ID>` to discard it. Proposals are limited to
one function or method in `friday.py`; FRIDAY checks that the file has not
changed, creates a unique backup, and applies the edit atomically. It does not
run edited code or tests automatically. Restart FRIDAY for approved changes to
take effect.

```bash
FRIDAY_MODE=ui python friday.py      # text-first chat UI, default when Tk is available
python friday.py --ui                # open the chat UI (voice is optional)
python friday.py --text              # terminal text mode, no GUI or mic needed
```

The desktop UI has **Text** and **Voice** modes. Text mode is a chat interface
and does not start the microphone. Voice mode opens the voice console and
starts listening; switch back to Text to stop listening. The model menu defaults
to **Auto (task-based)**: Friday prefers an installed coder model for coding,
an installed reasoning model for analysis, and the configured default for
general chat. Screen images use an installed vision model when available. Pick
a specific model in the menu to override automatic routing. The refresh button
re-reads the locally installed models. Only models pulled with
`ollama pull <model>` are available offline; Friday does not download models
automatically or load every model at once.

Friday allows up to 2048 generated tokens for a normal answer and 4096 when you
ask for a detailed explanation. To change those limits, set
`FRIDAY_REPLY_TOKENS` and `FRIDAY_DETAIL_TOKENS` before starting Friday; for
example, `FRIDAY_REPLY_TOKENS=3072 FRIDAY_DETAIL_TOKENS=6144`.

Set a built-in female Silero voice with `FRIDAY_SILERO_VOICE=en_1` (or another
available speaker such as `en_3` or `en_5`) before launching the assistant.

Voice mode calibrates to your room's background noise for a second, then
listens continuously for the wake word. Say "Hey Friday" and then your
request, or say them together: "Hey Friday, open Chrome."

Voice mode prefers a named physical analog microphone over generic PipeWire,
monitor, and HDMI devices, then calibrates background noise for two seconds.
Stay quiet during the calibration prompt. To choose another input explicitly,
list device indices and set `FRIDAY_MICROPHONE_INDEX`:

```bash
python -c "import speech_recognition as sr; print(list(enumerate(sr.Microphone.list_microphone_names())))"
FRIDAY_MICROPHONE_INDEX=1 .venv/bin/python friday.py --voice
```

Speech recognition uses Google's Web Speech endpoint and therefore requires
internet. It defaults to Indian English (`en-IN`); set
`FRIDAY_SPEECH_LANGUAGE` to another SpeechRecognition language code if needed,
for example `en-GB` or `hi-IN`. Say “Goodbye Friday” to close Friday from voice
mode; plain “bye” does not exit the assistant.

## Control OBS Studio

FRIDAY can check OBS status, start or stop streaming and recording, pause or
resume recording, switch to an existing scene, and mute or unmute an existing
audio input. Install the added dependency by rerunning `python -m pip install
-r requirements.txt`, then enable **Tools > WebSocket Server Settings** in OBS.
OBS WebSocket is built into OBS Studio 28 and newer; its default port is
`4455`. Set a WebSocket password in OBS and configure the same value before
starting FRIDAY:

```bash
OBS_HOST=127.0.0.1 OBS_PORT=4455 OBS_PASSWORD='your-password' python friday.py --text
```

`OBS_PASSWORD` can be left unset if the OBS server has no password. The
connection defaults to `127.0.0.1:4455`. Example requests include “Is OBS
streaming?”, “Start recording”, “Switch OBS to Gameplay”, and “Mute the
Desktop Audio input”. Say “OBS mode” to keep subsequent commands focused on
OBS; say “exit OBS mode” to leave it. In OBS mode, requests can also list
scenes or inputs, enable or disable Studio Mode, and show or hide a named
source within a scene. OBS must already be running for these controls; “open
OBS” still launches the app.

Ask “what's on my screen?” or “describe my screen” to capture the current
desktop and have the local `OLLAMA_VISION_MODEL` describe it (defaults to
`moondream:latest`). On KDE Wayland, FRIDAY captures the desktop with
Spectacle and sends a resized JPEG to your local Ollama server. The screenshot
is not sent to a cloud service or kept after analysis. Screen analysis can
take several seconds on CPU; text-only Ollama replies stream as they are
generated.

For a screen-guided scene decision, ask “look at my screen and choose the best
OBS scene”. FRIDAY supplies the current scene and available scene/source names;
the model may switch scenes or adjust source visibility, but cannot start or
stop streaming or recording from a screen-based guess. Use direct commands such
as “start streaming” or “start recording” when you intend those actions. These
direct OBS commands go over WebSocket without waiting for Ollama.

## Dictation and research documents

On Linux Wayland, install a keyboard input helper so FRIDAY can type into the
currently focused field. On Arch use `sudo pacman -S wtype`; on Debian or
Ubuntu use:

```bash
sudo apt install wtype
```

Say "start dictation mode", focus the text field where you want the words, then
speak naturally. Say "stop dictation" to exit. Dictation types recognized text
directly into the focused field without sending it to the chat model.

Ask FRIDAY to research a topic and create either format:

```text
Research renewable energy and make a Word document
Research renewable energy and make a PowerPoint
```

Research PowerPoints use several web searches, gather up to 12 distinct
sources, and include readable source-page text when available. Each factual
slide bullet is tied to numbered source references, with source URLs listed
at the end of the deck. For example:

```text
Research deeply on renewable energy and make a fact-based PowerPoint
```

The local Ollama model creates the cited outline. Word research briefs keep
the shorter search-summary workflow. Documents are saved in the same folder
as `friday.py`.

To turn the current assistant conversation into a PowerPoint, say "make a
presentation of our chat" or "give me PPT". The `.pptx` is also saved beside
`friday.py`.

## 3. Teach it your apps

Open `apps_config.json`. For any app that doesn't launch correctly by its
plain name, add an entry with the exact command your OS uses to launch it
(you can test the command yourself in a terminal first). This file is also
where you tell FRIDAY the process name to use when you ask it to close
something.

On Windows, FRIDAY can also open apps registered in the Start menu by name,
even when their executable isn't on PATH. Say "show apps" to list configured
apps and installed desktop apps. On macOS and Linux, apps outside the config
are still attempted by their system command or registered display name.

## How the voice pipeline works, honestly

- **Speech-to-text** uses `SpeechRecognition`'s free Google Web Speech
  endpoint by default. This voice transcription sends audio to Google and
  needs an internet connection, but no sign-up or key. If you want fully
  offline recognition, swap
  `recognizer.recognize_google(audio)` for
  `recognizer.recognize_sphinx(audio)` after `pip install pocketsphinx`.
- **Text-to-speech** prefers Silero TTS locally through PyTorch and
  `sounddevice` when available. The default English female voice is `en_1`; set
  `FRIDAY_SILERO_VOICE` to override it. If Silero cannot load or play audio,
  FRIDAY falls back to `pyttsx3` and then the platform speech tools (eSpeak on
  Linux).

## A note on `run command:`

This lets voice or text input execute a real shell command on your machine.
That's what makes it genuinely useful for open-ended tasks, but it also
means a misheard phrase could run something you didn't intend. It's off by
default in the sense that it only fires on the exact phrase `run command:` —
consider removing that block in `friday.py` if you'd rather it never touch
your shell directly, or restrict it to a fixed set of safe commands.

## Extending it

Every command lives in `Friday.handle()` in `friday.py`. Add a new `if`
branch, return a string reply, and it's live next run — no build step, no
redeploy.
