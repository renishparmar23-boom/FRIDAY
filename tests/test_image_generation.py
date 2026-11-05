import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import Mock, patch

from PIL import Image

import friday


class ImageGenerationTests(unittest.TestCase):
    def test_generate_image_saves_png_and_uses_local_pipeline(self):
        with tempfile.TemporaryDirectory() as output_dir:
            class Pipeline:
                def __init__(self):
                    self.device = None
                    self.request = None

                def to(self, device):
                    self.device = device

                def __call__(self, prompt, **options):
                    self.request = (prompt, options)
                    return types.SimpleNamespace(images=[Image.new("RGB", (2, 2), "red")])

            pipeline = Pipeline()
            pipeline_factory = Mock()
            pipeline_factory.from_pretrained.return_value = pipeline
            diffusers = types.ModuleType("diffusers")
            diffusers.AutoPipelineForText2Image = pipeline_factory
            torch_stub = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                float16="float16",
                float32="float32",
            )

            with (
                patch.object(friday, "APP_DIR", output_dir),
                patch.object(friday, "torch", torch_stub),
                patch.object(friday, "_IMAGE_PIPELINE", None),
                patch.object(friday, "_IMAGE_PIPELINE_MODEL", None),
                patch.dict(sys.modules, {"diffusers": diffusers}),
            ):
                image_path = friday.generate_image("A red fox in a snowy forest")

            self.assertTrue(image_path.is_file())
            self.assertEqual(image_path.parent, Path(output_dir) / "FRIDAY Images")
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (2, 2))
            pipeline_factory.from_pretrained.assert_called_once_with(
                "stabilityai/sd-turbo", torch_dtype="float32"
            )
            self.assertEqual(pipeline.device, "cpu")
            self.assertEqual(pipeline.request[0], "A red fox in a snowy forest")
            self.assertEqual(pipeline.request[1]["num_inference_steps"], 1)

    def test_generate_image_rejects_empty_prompt(self):
        with (
            patch.dict(sys.modules, {"diffusers": None}),
        ):
            with self.assertRaisesRegex(ValueError, "Tell me what image"):
                friday.generate_image("  ")

    def test_explicit_image_request_starts_generation_worker(self):
        assistant = friday.Friday.__new__(friday.Friday)
        assistant.dictation_mode = False
        assistant.obs_mode = False
        assistant.editor_waiting_for_text = False
        assistant.response_state_lock = Lock()
        assistant.response_generation = 3
        worker = type("Worker", (), {"start": lambda self: None})()

        with patch.object(friday.threading, "Thread", return_value=worker) as thread:
            reply = assistant.handle("Create an image of a red fox in a snowy forest")

        self.assertIn("Creating your image", reply)
        self.assertEqual(thread.call_args.kwargs["args"], ("a red fox in a snowy forest", 3))
        self.assertEqual(thread.call_args.kwargs["target"], assistant._image_generation_worker)


if __name__ == "__main__":
    unittest.main()