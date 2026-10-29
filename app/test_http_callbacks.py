import unittest

from voice_demo_multicandidate import DemoHandler


class HttpCallbackBindingTest(unittest.TestCase):
    def test_personalization_callbacks_are_not_instance_bound(self) -> None:
        class TestHandler(DemoHandler):
            pass

        TestHandler.configure_personalization_callbacks(
            lambda request: {"request": request},
            lambda language: {"language": language},
        )
        handler = object.__new__(TestHandler)

        self.assertEqual(
            handler.correction_callback({"value": 1}),
            {"request": {"value": 1}},
        )
        self.assertEqual(
            handler.clear_personalization_callback("ja"),
            {"language": "ja"},
        )


if __name__ == "__main__":
    unittest.main()
