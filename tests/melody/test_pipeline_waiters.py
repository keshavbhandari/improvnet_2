"""A waiting stage must not wake all other waiting stages after a spurious wake."""
import queue
import threading
import torch
from improvnet.inference.melody_pipeline import NoteStream, PipelineCoordinator


class CountedCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.notifications = 0
        self.wait_entries = queue.Queue()

    def notify_all(self):
        self.notifications += 1
        return super().notify_all()

    def wait(self, timeout=None):
        self.wait_entries.put(None)
        return super().wait(timeout)


def test_idle_stage_does_not_renotify_after_an_unrelated_wakeup():
    coordinator=PipelineCoordinator(None,torch.device('cpu'),1)
    coordinator.condition=condition=CountedCondition()
    stream=NoteStream(coordinator,0)
    results=[]
    worker=threading.Thread(target=lambda:results.append(stream.read()),daemon=True)
    worker.start()
    try:
        condition.wait_entries.get(timeout=2)
        with condition:
            assert coordinator.states==['source']
            first=condition.notifications
            condition.notify_all()  # Another stage's unrelated notification.
        condition.wait_entries.get(timeout=2)
        with condition:
            assert condition.notifications==first+1
            assert coordinator.states==['source']
    finally:
        stream.put(None)
        worker.join(timeout=2)
    assert not worker.is_alive() and results==[None] and stream.done
