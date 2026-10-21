"""
resource_monitor.py
----------------------
Measures the ACTUAL Ransomware Killer service's own CPU and memory
usage against the proposal's stated non-functional requirements
(<5% CPU average, <50MB RAM). This closes a real, previously-hidden
gap: the dashboard's "Endpoint Vitals" panel was reporting the
DASHBOARD WEB SERVER's own resource usage (via psutil.Process() with
no args = current process), not the actual file-monitoring service's
usage - two genuinely different processes with genuinely different
numbers.

Writes periodic snapshots to a small JSON status file that the
dashboard reads, so the vitals shown are honest measurements of the
thing actually claiming to meet the <5%/<50MB targets, not a
convenient stand-in.
"""

import os
import json
import time
import threading
import psutil
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResourceMonitor:
    def __init__(self, status_path=None,
                 sample_interval: float = 2.0, cpu_target_percent: float = 5.0,
                 memory_target_mb: float = 50.0):
        from src.utils import paths
        status_path = status_path or paths.runtime("data/simulated_events/service_status.json")
        self.status_path = status_path
        self.sample_interval = sample_interval
        self.cpu_target_percent = cpu_target_percent
        self.memory_target_mb = memory_target_mb
        self._process = psutil.Process(os.getpid())  # the process THIS runs inside
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._samples = []  # kept in-memory for a running average, capped below
        os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)

    def start(self):
        # Prime psutil's internal CPU counter - the first call to
        # cpu_percent() always returns 0.0 by design (needs a baseline
        # to measure against on the next call).
        self._process.cpu_percent(interval=None)
        self._thread.start()
        logger.info(f"[resource_monitor] Started (targets: <{self.cpu_target_percent}% CPU, "
                    f"<{self.memory_target_mb}MB RAM)")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop_event.is_set():
            # Wait BEFORE sampling, including on the first iteration.
            # Sampling immediately after cpu_percent(interval=None) was
            # primed in start() gives psutil a near-zero elapsed window
            # to measure against, which produces wildly inflated/noisy
            # readings (confirmed via testing: a real run showed a
            # nonsensical 355% CPU on the first sample). Waiting a full
            # interval first ensures every sample has a meaningful
            # elapsed window behind it.
            self._stop_event.wait(self.sample_interval)
            if self._stop_event.is_set():
                break
            self._sample_and_write()

    def _sample_and_write(self):
        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            memory_mb = self._process.memory_info().rss / (1024 * 1024)
        except psutil.NoSuchProcess:
            return

        self._samples.append((cpu_percent, memory_mb))
        self._samples = self._samples[-30:]  # keep a bounded rolling window

        avg_cpu = sum(s[0] for s in self._samples) / len(self._samples)
        avg_mem = sum(s[1] for s in self._samples) / len(self._samples)

        status = {
            "pid": os.getpid(),
            "timestamp": time.time(),
            "cpu_percent_current": round(cpu_percent, 2),
            "cpu_percent_avg": round(avg_cpu, 2),
            "memory_mb_current": round(memory_mb, 2),
            "memory_mb_avg": round(avg_mem, 2),
            "cpu_target_percent": self.cpu_target_percent,
            "memory_target_mb": self.memory_target_mb,
            "cpu_within_target": avg_cpu <= self.cpu_target_percent,
            "memory_within_target": avg_mem <= self.memory_target_mb,
        }

        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

        if not status["cpu_within_target"] or not status["memory_within_target"]:
            logger.warning(f"[resource_monitor] Over target - CPU avg={avg_cpu:.1f}% "
                           f"(target <{self.cpu_target_percent}%), "
                           f"RAM avg={avg_mem:.1f}MB (target <{self.memory_target_mb}MB)")


if __name__ == "__main__":
    # Self-contained smoke test: run for a few seconds under light
    # load, print real measured numbers, and honestly report whether
    # this process meets the proposal's own stated targets.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        status_path = os.path.join(tmp, "status.json")
        monitor = ResourceMonitor(status_path=status_path, sample_interval=0.5)
        monitor.start()

        # Light simulated workload
        time.sleep(3)

        monitor.stop()

        with open(status_path) as f:
            final_status = json.load(f)

        print("Final measured status:")
        for k, v in final_status.items():
            print(f"  {k}: {v}")

        print(f"\nCPU within <{monitor.cpu_target_percent}% target: {final_status['cpu_within_target']}")
        print(f"Memory within <{monitor.memory_target_mb}MB target: {final_status['memory_within_target']}")
