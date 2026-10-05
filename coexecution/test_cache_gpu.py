"""Validate publication and reader lifetime using actual CUDA events."""

import unittest

import torch

from coexecution.cache import FeatureCache


class TestCacheGPU(unittest.TestCase):
    def test_unfinished_write_is_never_published(self):
        self.assertEqual(torch.cuda.device_count(), 1)
        stream = torch.cuda.Stream()
        cache = FeatureCache({"x": torch.zeros(4, device="cuda")})
        writer = cache.reserve_write(100)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            cache.tensors(writer)["x"].fill_(7)
            done = torch.cuda.Event()
            done.record()
        cache.finish_write(writer, done)
        self.assertFalse(cache.publish_ready())
        lease = cache.acquire()
        self.assertEqual(lease.generation, 0)
        self.assertEqual(lease.tensors["x"].sum().item(), 0)
        cache.release(lease, None)
        done.synchronize()
        self.assertTrue(cache.publish_ready())
        lease = cache.acquire()
        self.assertEqual(lease.generation, 1)
        self.assertEqual(lease.capture_ns, 100)
        torch.testing.assert_close(
            lease.tensors["x"], torch.full((4,), 7.0, device="cuda")
        )
        cache.release(lease, None)

    def test_retired_slot_is_not_reused_until_reader_completes(self):
        cache = FeatureCache({"x": torch.zeros(4, device="cuda")})
        lease = cache.acquire()
        writer = cache.reserve_write(200)
        finished = torch.cuda.Event()
        finished.record()
        finished.synchronize()
        cache.finish_write(writer, finished)
        self.assertTrue(cache.publish_ready())
        self.assertIsNone(cache.reserve_write(300))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            reader_done = torch.cuda.Event()
            reader_done.record()
        cache.release(lease, reader_done)
        self.assertIsNone(cache.reserve_write(300))
        reader_done.synchronize()
        writer = cache.reserve_write(300)
        self.assertIsNotNone(writer)
        self.assertEqual(writer.slot, lease.slot)


if __name__ == "__main__":
    unittest.main()
