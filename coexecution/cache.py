"""Two GPU-resident slots with completion-gated publication and reader leases."""

from dataclasses import dataclass, field
from threading import Lock

import torch


@dataclass(frozen=True)
class WriteTicket:
    slot: int
    generation: int
    capture_ns: int


@dataclass(frozen=True)
class ReadLease:
    slot: int
    generation: int
    capture_ns: int
    lease_id: int
    tensors: dict
    ready: object


@dataclass
class Slot:
    tensors: dict
    generation: int = -1
    capture_ns: int = 0
    ready: object = None
    writer: object = None
    readers: dict = field(default_factory=dict)


class FeatureCache:
    def __init__(self, template):
        assert all(t.is_cuda for t in template.values())
        self.slots = [
            Slot({k: v.clone() for k, v in template.items()}) for _ in range(2)
        ]
        initialized = torch.cuda.Event()
        initialized.record()
        initialized.synchronize()  # Initialization only; never a hot-path device barrier.
        self.slots[0].generation = 0
        self.slots[0].ready = initialized
        self.active = 0
        self.generation = 0
        self.next_lease = 0
        self.lock = Lock()
        self.audit = []

    def acquire(self):
        with self.lock:
            slot = self.slots[self.active]
            self.next_lease += 1
            slot.readers[self.next_lease] = (
                None  # None means submission still owns the slot.
            )
            return ReadLease(
                self.active,
                slot.generation,
                slot.capture_ns,
                self.next_lease,
                slot.tensors,
                slot.ready,
            )

    def release(self, lease, done):
        with self.lock:
            slot = self.slots[lease.slot]
            assert slot.generation == lease.generation, "Reader's cache was overwritten"
            assert lease.lease_id in slot.readers
            if done is None:
                del slot.readers[lease.lease_id]  # No GPU read was submitted.
            else:
                slot.readers[lease.lease_id] = done

    def reserve_write(self, capture_ns):
        with self.lock:
            for index, slot in enumerate(self.slots):
                if index == self.active or slot.writer is not None:
                    continue
                slot.readers = {
                    key: event
                    for key, event in slot.readers.items()
                    if event is None or not event.query()
                }
                if slot.readers:
                    continue
                self.generation += 1
                ticket = WriteTicket(index, self.generation, capture_ns)
                slot.writer = ticket
                slot.ready = None
                return ticket
            return None

    def tensors(self, ticket):
        assert self.slots[ticket.slot].writer == ticket
        return self.slots[ticket.slot].tensors

    def copy_into(self, ticket, features):
        destination = self.tensors(ticket)
        assert destination.keys() == features.keys()
        for key, tensor in features.items():
            out = destination[key]
            assert (out.shape, out.dtype, out.device) == (
                tensor.shape,
                tensor.dtype,
                tensor.device,
            )
            out.copy_(tensor)

    def finish_write(self, ticket, done):
        with self.lock:
            slot = self.slots[ticket.slot]
            assert slot.writer == ticket
            slot.ready = done

    def publish_ready(self):
        with self.lock:
            candidates = [
                (i, slot)
                for i, slot in enumerate(self.slots)
                if slot.writer is not None
                and slot.ready is not None
                and slot.ready.query()
            ]
            if not candidates:
                return False
            index, slot = max(candidates, key=lambda pair: pair[1].writer.generation)
            ticket = slot.writer
            slot.generation, slot.capture_ns = ticket.generation, ticket.capture_ns
            slot.writer = None
            self.active = index
            self.audit.append(
                {
                    "event": "publish",
                    "slot": index,
                    "generation": slot.generation,
                    "capture_ns": slot.capture_ns,
                    "gpu_event_complete": True,
                }
            )
            return True
