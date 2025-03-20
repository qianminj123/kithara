"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Unit tests for correctness of MaxTextModel.generate() function 

Run test on a TPU VM: python -m unittest tests/model/maxtext/test_inference.py 
"""

import unittest
import numpy as np
from transformers import AutoTokenizer
from kithara import MaxTextModel
import time
import unittest.result
import jax
from tests.test_utils import timeout
import os


@unittest.skipIf(int(os.getenv("RUN_LIGHT_TESTS_ONLY", 0)) == 1, "Heavy Test")
class TestModelGeneration(unittest.TestCase):

    def setUp(self):
        print(f"\nStarting test: {self._testMethodName}")
        self.start_time = time.time()

    def tearDown(self):
        duration = time.time() - self.start_time
        print(f"Completed test: {self._testMethodName} in {duration:.2f} seconds\n")

    @timeout(200)
    def test_input_formats(self):
        model = MaxTextModel.from_random("gemma2-2b", seq_len=100)

        with self.subTest("A single string"):
            pred = model.generate(
                "what is your name?",
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)

        with self.subTest("A list of strings"):
            pred = model.generate(
                ["what is your name?", "what is 1+1?"],
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)
            self.assertEqual(len(pred[1]), 100)

        with self.subTest("A list of integers"):
            pred = model.generate(
                [100, 101, 102],
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)

        with self.subTest("A list of list of integers"):
            pred = model.generate(
                [[100, 101, 102], [100, 101, 102]],
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)
            self.assertEqual(len(pred[1]), 100)

        with self.subTest("A numpy array"):
            pred = model.generate(
                [np.array([100, 101, 102])],
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)

        with self.subTest("A list of numpy array"):
            pred = model.generate(
                [np.array([100, 101, 102]), np.array([100, 101, 102])],
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertEqual(len(pred[0]), 100)
            self.assertEqual(len(pred[1]), 100)

    @timeout(200)
    def test_generate_stripping_prompt_with_string_input(self):
        model = MaxTextModel.from_random("gemma2-2b", seq_len=100)

        with self.subTest("Testing not stripping prompt"):
            pred = model.generate(
                "what is your name?",
                max_length=100,
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=True,
                strip_prompt=False,
            )
            self.assertIsInstance(pred[0], str)
            self.assertLess(len(pred[0].split(" ")), 100)

        with self.subTest("Testing stripping prompt"):
            pred = model.generate(
                "what is your name?",
                max_length=100,
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=True,
                strip_prompt=True,
            )
            self.assertIsInstance(pred[0], str)
            self.assertNotIn("what is your name?", pred[0])

    @timeout(200)
    def test_generate_stripping_prompt_with_token_inputs(self):
        model = MaxTextModel.from_random("gemma2-2b", seq_len=128)

        with self.subTest("Testing not stripping prompt"):
            pred = model.generate(
                "what is your name?",
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertIsInstance(pred, list)
            self.assertIsInstance(pred[0], list)
            self.assertIsInstance(pred[0][0], int)
            self.assertEqual(len(pred[0]), 100)

        with self.subTest("Testing stripping prompt"):
            pred = model.generate(
                "what is your name?",
                max_length=100,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=True,
            )
            self.assertLess(len(pred[0]), 95)

    @timeout(200)
    def test_generate_with_multiple_batch_inputs(self):
        model = MaxTextModel.from_random("gemma2-2b", seq_len=128)

        with self.subTest("Testing 10 string inputs"):
            pred = model.generate(
                ["what is your name?"] * 10,
                max_length=128,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertEqual(len(pred), 10)
            for i in range(10):
                self.assertEqual(len(pred[i]), 128)

        with self.subTest("Testing 10 integer lists"):
            pred = model.generate(
                [[3,4,5] for _ in range(10)],
                max_length=128,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertEqual(len(pred), 10)
            for i in range(10):
                self.assertEqual(len(pred[i]), 128)

        with self.subTest("Testing 10 numpy arrays"):
            pred = model.generate(
                [np.array([3,4,5]) for _ in range(10)],
                max_length=128,
                stop_token_ids=[],
                tokenizer_handle="hf://google/gemma-2-2b",
                return_decoded=False,
                strip_prompt=False,
            )
            self.assertEqual(len(pred), 10)
            for i in range(10):
                self.assertEqual(len(pred[i]), 128)

    @timeout(200)
    def test_generate_with_tokenizer_object(self):
        model = MaxTextModel.from_random("gemma2-2b", seq_len=100)
        with self.subTest("Testing with tokenizer object"):
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
            pred = model.generate(
                "what is your name?",
                tokenizer=tokenizer,
                max_length=100,
                return_decoded=True,
            )
            self.assertIsInstance(pred[0], str)
        with self.subTest("Testing without tokenizer object"):
            with self.assertRaises(AssertionError):
                model.generate("hello world")
            with self.assertRaises(AssertionError):
                model.generate([np.array([1, 2, 3])], return_decoded=True, max_length=-1)


    @timeout(200)
    def test_prompt_len_exceeds_max_prefill_len(self):
        model = MaxTextModel.from_random(
            "gemma2-2b", seq_len=100, max_prefill_predict_length=10
        )
        with self.subTest("Testing single prompt"):
            pred = model.generate(
                [np.ones(20, dtype="int")],
                tokenizer_handle="hf://google/gemma-2-2b",
                max_length=100,
                return_decoded=False,
                strip_prompt=False,
            )
            # No new tokens should be generated since prefill won't be successful
            self.assertEqual(len(pred[0]), 20)

        with self.subTest("Testing multiple prompt"):
            pred = model.generate(
                [
                    np.ones(20, dtype="int"),
                    np.ones(5, dtype="int"),
                    np.ones(5, dtype="int"),
                ],
                tokenizer_handle="hf://google/gemma-2-2b",
                max_length=100,
                return_decoded=False,
                strip_prompt=False,
                stop_token_ids=[],
            )
            self.assertEqual(len(pred[0]), 20)
            self.assertEqual(len(pred[1]), 100)
            self.assertEqual(len(pred[2]), 100)


    @timeout(200)
    def test_max_length(self):
        model = MaxTextModel.from_random(
            "gemma2-2b", seq_len=100, max_prefill_predict_length=10
        )
        pred = model.generate(
            [np.ones(5, dtype="int")],
            tokenizer_handle="hf://google/gemma-2-2b",
            max_length=120,
            stop_token_ids=[],
            return_decoded=False,
            strip_prompt=False,
        )
        self.assertEqual(len(pred[0]), 100)

if __name__ == "__main__":
    unittest.main(verbosity=2)
