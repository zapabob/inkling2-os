"""Boot the candidate ISO in BIOS and UEFI without attaching a writable disk."""

import argparse
from pathlib import Path
import os
import selectors
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time


def boot(iso: Path, firmware: str, deadline_seconds: int) -> None:
    args = [
        "qemu-system-x86_64", "-machine", "q35,accel=tcg", "-smp", "2",
        "-m", "1024", "-display", "none", "-monitor", "none", "-serial", "stdio",
        "-no-reboot", "-nic", "none", "-boot", "order=d", "-cdrom", str(iso),
    ]
    with TemporaryDirectory(prefix="inkling2-qemu-") as temporary:
        if firmware == "uefi":
            variables = Path(temporary) / "vars.fd"
            shutil.copyfile("/usr/share/OVMF/OVMF_VARS_4M.fd", variables)
            args += [
                "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd",
                "-drive", f"if=pflash,format=raw,file={variables}",
            ]
        run_vm(args, firmware, deadline_seconds)


def run_vm(args: list[str], firmware: str, deadline_seconds: int) -> None:
    required = (
        b"Root filesystem active: overlay; PID=1",
        b"GPU unavailable; inference is in diagnostic mode",
        b"Bridge healthy on loopback; attempt 1",
        b"Smoke: restart and health check passed",
        b"Shutdown complete",
        b"Power down",
    )
    deadline = time.monotonic() + deadline_seconds
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as vm:
        assert vm.stdin is not None and vm.stdout is not None
        selected = False
        log = bytearray()
        try:
            with selectors.DefaultSelector() as watch:
                watch.register(vm.stdout, selectors.EVENT_READ)
                while time.monotonic() < deadline:
                    if not watch.select(min(1, max(0, deadline - time.monotonic()))):
                        continue
                    chunk = os.read(vm.stdout.fileno(), 8192)
                    if not chunk:
                        break
                    log.extend(chunk)
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.flush()
                    if not selected and b"boot validation" in log:
                        vm.stdin.write(b"\x1b[B\r")
                        vm.stdin.flush()
                        selected = True
                    if len(log) > 4 * 1024 * 1024:
                        raise RuntimeError("VM serial log exceeded 4 MiB")
            remaining = max(0.1, deadline - time.monotonic())
            status = vm.wait(timeout=remaining)
            missing = [marker.decode() for marker in required if marker not in log]
            if status != 0 or missing:
                raise RuntimeError(f"{firmware} exit={status}, missing markers={missing}")
        finally:
            if vm.poll() is None:
                vm.kill()
    print(f"VERIFIED {firmware} root handoff, PID 1, diagnosis, service restart, shutdown", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", type=Path, default=Path("/out/inkling2-os-v0.1.1-x86_64-sm120.iso"))
    parser.add_argument("--firmware", choices=("bios", "uefi"), action="append")
    parser.add_argument("--timeout", type=int, default=180)
    opts = parser.parse_args()
    for target in opts.firmware or ("bios", "uefi"):
        print(f"BOOT {target}", flush=True)
        boot(opts.iso, target, opts.timeout)
