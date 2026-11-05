import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import friday


class VoiceDefaultsTests(unittest.TestCase):
    def test_default_mode_is_gui_when_requested(self):
        original = os.environ.get("FRIDAY_MODE")
        os.environ["FRIDAY_MODE"] = "ui"
        try:
            args = friday.parse_cli_args([])
            self.assertEqual(args.mode, "ui")
        finally:
            if original is None:
                os.environ.pop("FRIDAY_MODE", None)
            else:
                os.environ["FRIDAY_MODE"] = original

    def test_default_silero_voice_is_female(self):
        original = os.environ.get("FRIDAY_SILERO_VOICE")
        os.environ.pop("FRIDAY_SILERO_VOICE", None)
        try:
            voice = friday.Voice()
            self.assertEqual(voice.silero_speaker, "en_1")
        finally:
            if original is None:
                os.environ.pop("FRIDAY_SILERO_VOICE", None)
            else:
                os.environ["FRIDAY_SILERO_VOICE"] = original

    def test_linux_prefers_silero_over_espeak_when_available(self):
        with patch.object(friday, "SYSTEM", "Linux"), patch.object(friday, "HAVE_SILERO", True):
            voice = friday.Voice()
        self.assertEqual(voice.speech_backend, "silero")
        self.assertEqual(voice.silero_speaker, "en_1")

    def test_incomplete_request_prompts_for_missing_subject(self):
        self.assertTrue(friday.needs_request_clarification("tell me about"))
        self.assertFalse(friday.needs_request_clarification("tell me about servers"))

    def test_voice_bye_and_goodbye_are_exit_commands(self):
        for phrase in ("bye", "goodbye", "good bye", "goodby"):
            with self.subTest(phrase=phrase):
                self.assertTrue(friday.is_voice_exit_command(phrase))
        self.assertFalse(friday.is_voice_exit_command("good night Friday"))
        self.assertFalse(friday.is_voice_exit_command("bye everyone"))

    def test_goodbye_friday_is_an_explicit_close_phrase(self):
        for phrase in ("goodbye Friday", "good bye Friday", "bye Friday!"):
            with self.subTest(phrase=phrase):
                self.assertTrue(friday.is_friday_goodbye_phrase(phrase))
        self.assertFalse(friday.is_friday_goodbye_phrase("bye"))
        self.assertFalse(friday.is_friday_goodbye_phrase("goodbye everyone"))

    def test_microphone_selection_prefers_hardware_input_by_name(self):
        devices = [
            "HD-Audio HDMI 0",
            "HD-Audio Generic: ALC3204 Analog",
            "hdmi",
            "pipewire",
            "default",
            "Ryzen HD Audio Controller Analog Stereo",
        ]
        self.assertEqual(friday.preferred_microphone_index(devices), 1)

    def test_microphone_selection_uses_pipewire_only_as_fallback(self):
        self.assertEqual(
            friday.preferred_microphone_index(["HDMI output", "pipewire", "default"]),
            1,
        )

    def test_microphone_selection_prefers_alsa_hardware_over_jack_alias(self):
        devices = [
            "HD-Audio Generic: HDMI 0 (hw:0,3)",
            "HD-Audio Generic: ALC3204 Analog (hw:1,0)",
            "hdmi",
            "pipewire",
            "default",
            "Ryzen HD Audio Controller Analog Stereo",
        ]
        details = [
            {"maxInputChannels": 0, "hostApiName": "ALSA"},
            {"maxInputChannels": 2, "hostApiName": "ALSA"},
            {"maxInputChannels": 0, "hostApiName": "ALSA"},
            {"maxInputChannels": 128, "hostApiName": "ALSA"},
            {"maxInputChannels": 128, "hostApiName": "ALSA"},
            {"maxInputChannels": 4, "hostApiName": "JACK Audio Connection Kit"},
        ]
        self.assertEqual(friday.preferred_microphone_index(devices, details), 1)

    def test_microphone_selection_prefers_valid_portaudio_default(self):
        devices = ["ALSA hardware input", "PipeWire", "default"]
        details = [
            {"maxInputChannels": 2, "defaultSampleRate": 44100},
            {"maxInputChannels": 128, "defaultSampleRate": 44100},
            {"maxInputChannels": 128, "defaultSampleRate": 44100},
        ]
        self.assertEqual(friday.preferred_microphone_index(devices, details, 2), 2)

    def test_microphone_sample_rate_uses_device_default(self):
        details = [{"defaultSampleRate": 44100.0}]
        self.assertEqual(friday.microphone_sample_rate(0, details), 44100)
        self.assertEqual(friday.microphone_sample_rate(4, details), 16000)

    def test_voice_capture_waits_for_speech_before_recording(self):
        calls = []

        class VoiceStub:
            def wait_until_idle(self):
                calls.append("speech_idle")

        class RecognizerStub:
            def listen(self, source, timeout, phrase_time_limit):
                calls.append(("listen", timeout, phrase_time_limit))
                return "audio"

        with patch.object(friday, "quiet_audio_backend", return_value=nullcontext()):
            result = friday.listen_after_speech(RecognizerStub(), "source", VoiceStub())

        self.assertEqual(result, "audio")
        self.assertEqual(calls, ["speech_idle", ("listen", 20, 15)])

    def test_voice_loop_routes_recognized_command_and_reply(self):
        class Audio:
            def get_default_input_device_info(self):
                return {"index": 0}

            def get_device_count(self):
                return 1

            def get_device_info_by_index(self, index):
                return {"hostApi": 0, "maxInputChannels": 1, "defaultSampleRate": 44100}

            def get_host_api_info_by_index(self, index):
                return {"name": "ALSA"}

            def terminate(self):
                pass

        class PyAudio:
            def PyAudio(self):
                return Audio()

        class Microphone:
            created = []

            def __init__(self, device_index=None, sample_rate=None):
                self.device_index = device_index
                self.sample_rate = sample_rate
                self.created.append((device_index, sample_rate))

            @staticmethod
            def get_pyaudio():
                return PyAudio()

            def __enter__(self):
                return "mock source"

            def __exit__(self, *args):
                return False

            @staticmethod
            def list_microphone_names():
                return ["mock ALSA mic"]

        class Recognizer:
            pause_threshold = 0
            operation_timeout = 0
            non_speaking_duration = 0
            dynamic_energy_threshold = False
            energy_threshold = 0

            def adjust_for_ambient_noise(self, source, duration):
                self.calibrated = (source, duration)

            def listen(self, source, timeout, phrase_time_limit):
                self.listen_options = (timeout, phrase_time_limit)
                return "mock audio"

            def recognize_google(self, audio, language):
                self.language = language
                return "what time is it"

        class Voice:
            speech_in_progress = False

            def say(self, text):
                pass

            def wait_until_idle(self):
                pass

        class Assistant:
            running = True
            dictation_mode = False

            def respond(self, command):
                self.command = command
                self.running = False
                return "The current time is noon."

        class UI:
            listening = True
            voice_session_id = 7

            def __init__(self):
                self.messages = []
                self.destroy_requested = False

            def set_listening_state(self, *args):
                pass

            def queue_message(self, sender, message):
                self.messages.append((sender, message))

            def request_destroy(self):
                self.destroy_requested = True

        recognizer = Recognizer()
        speech_module = SimpleNamespace(
            Recognizer=lambda: recognizer,
            Microphone=Microphone,
            UnknownValueError=type("UnknownValueError", (Exception,), {}),
            RequestError=type("RequestError", (Exception,), {}),
            WaitTimeoutError=type("WaitTimeoutError", (Exception,), {}),
        )
        assistant = Assistant()
        ui = UI()
        with (
            patch.object(friday, "HAVE_SR", True),
            patch.object(friday, "sr", speech_module),
            patch.object(friday, "quiet_audio_backend", return_value=nullcontext()),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("FRIDAY_MICROPHONE_INDEX", None)
            friday.voice_loop(assistant, Voice(), ui=ui, session_id=7)

        self.assertEqual(assistant.command, "what time is it")
        self.assertIn(("You", "what time is it"), ui.messages)
        self.assertIn(("FRIDAY", "The current time is noon."), ui.messages)
        self.assertTrue(ui.destroy_requested)
        self.assertEqual(recognizer.listen_options, (20, 15))
        self.assertEqual(recognizer.language, "en-IN")
        self.assertEqual(Microphone.created[0], (0, 44100))

    def test_ollama_models_are_discovered_from_local_server(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"models":[{"name":"qwen2.5:3b"},{"name":"llama3.2:3b"}]}'

        with patch.object(friday.urllib.request, "urlopen", return_value=Response()):
            self.assertEqual(friday.list_ollama_models(), ["llama3.2:3b", "qwen2.5:3b"])

    def test_ollama_uses_explicit_model_selection(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"hello"}}'

        def capture_request(request, timeout):
            import json
            captured.update(json.loads(request.data.decode("utf-8")))
            return Response()

        with patch.object(friday.urllib.request, "urlopen", side_effect=capture_request):
            self.assertEqual(friday.ask_ollama("hello", model="qwen2.5:3b"), "hello")
        self.assertEqual(captured["model"], "qwen2.5:3b")

    def test_chat_history_is_restored_and_sent_as_context(self):
        import json
        import tempfile

        history = [
            {"role": "user", "content": "My project is called Atlas."},
            {"role": "assistant", "content": "Got it, your project is Atlas."},
        ]
        captured = {}
        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "chat.json")
            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(history, history_file)
            with (
                patch.object(friday, "CHAT_HISTORY_PATH", history_path),
                patch.object(friday, "load_notes", return_value=[]),
                patch.object(friday, "load_routines", return_value={}),
                patch.object(friday, "load_reminders", return_value=[]),
                patch.object(friday.threading, "Thread"),
                patch.object(
                    friday,
                    "ask_ollama",
                    side_effect=lambda prompt, supplied_history, **kwargs: (
                        captured.update(history=list(supplied_history)) or "Atlas."
                    ),
                ),
            ):
                assistant = friday.Friday(None)
                assistant.model_for_task = lambda prompt, images=False: "test-model"
                assistant.response_context = SimpleNamespace(on_token=None)
                assistant._ask_ollama("What is my project called?")

                with open(history_path, "r", encoding="utf-8") as history_file:
                    saved_history = json.load(history_file)

        self.assertEqual(captured["history"], history)
        self.assertEqual(saved_history[-2:], [
            {"role": "user", "content": "What is my project called?"},
            {"role": "assistant", "content": "Atlas."},
        ])

    def test_chat_history_loader_ignores_invalid_entries_and_limits_context(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            history_path = os.path.join(directory, "chat.json")
            history = [
                {"role": "user", "content": f"turn {index}"}
                for index in range(45)
            ] + [{"role": "tool", "content": "ignore"}, "invalid"]
            with open(history_path, "w", encoding="utf-8") as history_file:
                json.dump(history, history_file)
            with patch.object(friday, "CHAT_HISTORY_PATH", history_path):
                restored = friday.load_chat_history()

        self.assertEqual(len(restored), 40)
        self.assertEqual(restored[0]["content"], "turn 5")

    def test_model_router_selects_installed_model_for_task(self):
        models = ["llama3.2:3b", "qwen2.5-coder:3b", "deepseek-r1:1.5b"]
        self.assertEqual(
            friday.choose_ollama_model("Write a Python script", models, fallback=models[0]),
            "qwen2.5-coder:3b",
        )
        self.assertEqual(
            friday.choose_ollama_model("Analyze these tradeoffs", models, fallback=models[0]),
            "deepseek-r1:1.5b",
        )
        self.assertEqual(
            friday.choose_ollama_model("Tell me a joke", models, fallback=models[0]),
            "llama3.2:3b",
        )

    def test_model_router_uses_only_available_vision_model(self):
        models = ["llama3.2:3b", "llava:7b"]
        self.assertEqual(
            friday.choose_ollama_model("Describe this", models, fallback=models[0], images=True),
            "llava:7b",
        )
        self.assertIsNone(
            friday.choose_ollama_model("Describe this", ["llama3.2:3b"], fallback="llama3.2:3b", images=True)
        )

    def test_friday_auto_mode_routes_but_manual_mode_overrides(self):
        assistant = object.__new__(friday.Friday)
        assistant.model = "llama3.2:3b"
        assistant.model_mode = "auto"
        models = ["llama3.2:3b", "qwen2.5-coder:3b"]
        with patch.object(friday, "list_ollama_models", return_value=models):
            self.assertEqual(assistant.model_for_task("Write a Python function"), "qwen2.5-coder:3b")
            assistant.model_mode = "manual"
            assistant.model = "llama3.2:3b"
            self.assertEqual(assistant.model_for_task("Write a Python function"), "llama3.2:3b")

    def test_ollama_reply_budget_allows_complete_answers(self):
        budgets = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"message":{"content":"complete"}}'

        def capture_request(request, timeout):
            import json
            budgets.append(json.loads(request.data.decode("utf-8"))["options"]["num_predict"])
            return Response()

        with patch.object(friday.urllib.request, "urlopen", side_effect=capture_request):
            friday.ask_ollama("hello")
            friday.ask_ollama("explain this in detail")

        self.assertEqual(budgets, [2048, 4096])

    def test_source_edit_replacement_changes_only_named_function(self):
        source = "VALUE = 1\n\ndef sample():\n    return 'old'\n\n\ndef other():\n    return 'kept'\n"
        start, end, old_block = friday.source_symbol_span(source, "sample")
        self.assertEqual(old_block, "def sample():\n    return 'old'\n")
        self.assertEqual((start, end), (3, 4))

        candidate = friday.replace_source_symbol(
            source,
            "sample",
            "def sample():\n    return 'new'\n",
        )

        self.assertIn("return 'new'", candidate)
        self.assertIn("return 'kept'", candidate)
        self.assertIn("VALUE = 1", candidate)
        with self.assertRaises(SyntaxError):
            friday.replace_source_symbol(source, "sample", "def sample(:\n    pass\n")

    def test_source_edit_is_written_only_after_exact_approval_and_backed_up(self):
        import json
        import tempfile
        from pathlib import Path

        source = "def sample():\n    return 'old'\n"
        replacement = "def sample():\n    return 'new'\n"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "friday.py"
            source_path.write_text(source, encoding="utf-8")
            assistant = object.__new__(friday.Friday)
            assistant.model_mode = "manual"
            assistant.model = "test-model"
            assistant.pending_source_edit = None
            assistant.dictation_mode = False

            with (
                patch.object(friday, "__file__", str(source_path)),
                patch.object(friday, "ask_ollama", return_value=json.dumps({"replacement": replacement})),
            ):
                proposal_text = assistant._propose_source_edit("sample", "return the new value")
                proposal_id = assistant.pending_source_edit["id"]
                self.assertIn("No file has been changed", proposal_text)
                self.assertEqual(source_path.read_text(encoding="utf-8"), source)
                self.assertIn(proposal_id, assistant.handle("review source edit"))

                wrong_id = "000000" if proposal_id != "000000" else "ffffff"
                refused = assistant.handle(f"approve source edit {wrong_id}")
                self.assertIn("There is no pending source edit", refused)
                self.assertEqual(source_path.read_text(encoding="utf-8"), source)

                result = assistant.handle(f"approve source edit {proposal_id}")
                backups = list(Path(directory).glob("friday.py.backup-*"))
                written_source = source_path.read_text(encoding="utf-8")
                backup_content = backups[0].read_text(encoding="utf-8") if backups else None

        self.assertIn("Restart FRIDAY", result)
        self.assertEqual(written_source, replacement)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backup_content, source)

    def test_source_edit_refuses_stale_proposal_and_rejection_changes_nothing(self):
        import json
        import tempfile
        from pathlib import Path

        source = "def sample():\n    return 'old'\n"
        replacement = "def sample():\n    return 'new'\n"
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "friday.py"
            source_path.write_text(source, encoding="utf-8")
            assistant = object.__new__(friday.Friday)
            assistant.model_mode = "manual"
            assistant.model = "test-model"
            assistant.pending_source_edit = None
            assistant.dictation_mode = False

            with (
                patch.object(friday, "__file__", str(source_path)),
                patch.object(friday, "ask_ollama", return_value=json.dumps({"replacement": replacement})),
            ):
                assistant._propose_source_edit("sample", "return the new value")
                source_path.write_text("def sample():\n    return 'changed elsewhere'\n", encoding="utf-8")
                stale_result = assistant.handle(
                    f"approve source edit {assistant.pending_source_edit['id']}"
                )

                assistant._propose_source_edit("sample", "return the new value")
                proposal_id = assistant.pending_source_edit["id"]
                rejected_result = assistant.handle(f"reject source edit {proposal_id}")

            current_source = source_path.read_text(encoding="utf-8")
            backups = list(Path(directory).glob("friday.py.backup-*"))

        self.assertIn("source changed", stale_result)
        self.assertIn("No file was changed", rejected_result)
        self.assertEqual(current_source, "def sample():\n    return 'changed elsewhere'\n")
        self.assertEqual(backups, [])


if __name__ == "__main__":
    unittest.main()
