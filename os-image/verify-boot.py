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


def boot(
    iso: Path, firmware: str, deadline_seconds: int,
    disk: Path | None = None, model: bool = False, desktop: bool = False, tor: bool = False,
) -> None:
    args = [
        "qemu-system-x86_64", "-machine", "q35,accel=tcg", "-cpu", "max", "-smp", "4",
        "-m", "3072", "-display", "none", "-monitor", "none", "-serial", "stdio",
        "-no-reboot", "-nic", "none",
    ]
    if disk is None:
        args += ["-boot", "order=d", "-cdrom", str(iso)]
    else:
        args += [
            "-snapshot", "-device", "qemu-xhci", "-drive", f"if=none,id=installed,file={disk},format=raw",
            "-device", "usb-storage,drive=installed,bootindex=1",
        ]
    with TemporaryDirectory(prefix="inkling2-qemu-") as temporary:
        if firmware == "uefi":
            variables = Path(temporary) / "vars.fd"
            shutil.copyfile("/usr/share/OVMF/OVMF_VARS_4M.fd", variables)
            args += [
                "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd",
                "-drive", f"if=pflash,format=raw,file={variables}",
            ]
        run_vm(args, firmware, deadline_seconds, "ext4" if disk else "overlay", model, desktop, tor, temporary)


def run_vm(
    args: list[str], firmware: str, deadline_seconds: int, root: str,
    model: bool, desktop: bool, tor: bool, temporary: str,
) -> None:
    required = (
        f"Root filesystem active: {root}; PID=1".encode(),
        b"GPU unavailable; inference is in diagnostic mode",
        b"Bridge healthy on loopback; attempt 1",
        b"Loaded default-deny ingress and egress nftables firewall",
        b"Smoke: console UID and private file directory verified",
        b"Smoke: offline Wine executed Windows cmd.exe",
        b"Smoke: restart and health check passed",
        b"Shutdown complete",
        b"Power down",
    )
    kernel_faults = (b"BUG: unable to handle page fault", b"Kernel panic")
    if model:
        required += (b"Bundled model volume mounted read-only",)
    if desktop:
        required += (
            b"Smoke: local graphical console loaded",
            b'profile="full"',
            b"listening on http://127.0.0.1:29483",
        )
    if tor:
        required += (b"Smoke: Tor Browser opened under its dedicated account",)
    deadline = time.monotonic() + deadline_seconds
    # Keep QEMU's writable snapshot beside the temporary firmware, inside the verification tmpfs.
    with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env={**os.environ, "TMPDIR": temporary}) as vm:
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
            faults = [marker.decode() for marker in kernel_faults if marker in log]
            if status != 0 or missing or faults:
                raise RuntimeError(f"{firmware} exit={status}, missing markers={missing}, kernel faults={faults}")
        finally:
            if vm.poll() is None:
                vm.kill()
    print(f"VERIFIED {firmware} root handoff, PID 1, diagnosis, service restart, shutdown", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iso", type=Path, default=Path("/out/inkling2-os-v1.3.0-x86_64-sm120.iso"))
    parser.add_argument("--firmware", choices=("bios", "uefi"), action="append")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--installed-bundle", type=Path)
    parser.add_argument("--require-desktop", action="store_true")
    parser.add_argument("--require-tor", action="store_true")
    opts = parser.parse_args()
    if opts.require_desktop and not opts.installed_bundle:
        parser.error("--require-desktop requires --installed-bundle")
    if opts.require_tor and not opts.require_desktop:
        parser.error("--require-tor requires --require-desktop")
    if opts.installed_bundle:
        from installed_disk import reconstruct

        with TemporaryDirectory(prefix="inkling2-installed-") as temporary:
            disk = Path(temporary) / "usb.raw"
            model = reconstruct(opts.installed_bundle, disk)
            print("BOOT installed USB (UEFI)", flush=True)
            boot(opts.iso, "uefi", opts.timeout or 600, disk, model, desktop=opts.require_desktop, tor=opts.require_tor)
    else:
        for target in opts.firmware or ("bios", "uefi"):
            print(f"BOOT {target}", flush=True)
            boot(opts.iso, target, opts.timeout or 360)
