"""Build and reconstruct a UEFI disk bundle without opening a host disk device."""

import argparse
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4


SECTOR = 512
MIB = 1024 * 1024
ESP_START = 2048
ESP_SECTORS = 128 * MIB // SECTOR
ROOT_START = ESP_START + ESP_SECTORS
MODEL_BYTES = 48 * 1024 * MIB


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def allocate(path: Path, length: int) -> None:
    with path.open("xb") as file:
        file.truncate(length)


def data_ranges(file):
    """Coalesce nearby allocated extents, copying the original bytes in gaps."""
    eof = os.fstat(file.fileno()).st_size
    position = 0
    previous = None
    while position < eof:
        try:
            start = os.lseek(file.fileno(), position, os.SEEK_DATA)
        except OSError as exc:
            if exc.errno != errno.ENXIO:
                raise
            break
        end = os.lseek(file.fileno(), start, os.SEEK_HOLE)
        if previous and start - previous[1] <= MIB:
            previous = previous[0], end
        else:
            if previous:
                yield previous
            previous = start, end
        position = end
    if previous:
        yield previous


def copy_region(source, sink, count: int) -> str:
    digest = hashlib.sha256()
    while count:
        chunk = source.read(min(count, MIB))
        if not chunk:
            raise OSError("Disk bundle contained a truncated region")
        sink.write(chunk)
        digest.update(chunk)
        count -= len(chunk)
    return digest.hexdigest()


def grub_config(root_uuid: str, private_install: bool = False) -> str:
    boot = f"root=UUID={root_uuid} rootfstype=ext4 rw rootwait init=/sbin/inkling2-init noresume console=tty1 console=ttyS0,115200n8"
    if private_install:
        boot += " amd_pstate=active"
    entries = (
        ("Inkling2-OS", ""),
        ("Inkling2-OS installed disk boot validation", " inkling2.smoke=1"),
    )
    cfg = (
        "set default=0\nset timeout=8\nserial --unit=0 --speed=115200\n"
        "terminal_input console serial\nterminal_output console serial\n"
        f"search --no-floppy --fs-uuid --set=root {root_uuid}\n"
    )
    for title, extra in entries:
        cfg += f'menuentry "{title}" {{\n    linux /boot/vmlinuz {boot}{extra}\n    initrd /boot/initrd.img\n}}\n'
    return cfg


def build(disk_bytes: int, rootfs: Path, iso_boot: Path, output: Path, with_model_volume: bool = False) -> None:
    if disk_bytes < 8 * 1024 * MIB or disk_bytes % SECTOR:
        raise ValueError("Expected a sector-aligned target disk of at least 8 GiB")
    output.mkdir(parents=True, exist_ok=False)
    root_uuid = str(uuid4())
    last_lba = disk_bytes // SECTOR - 34
    model_start = ((last_lba + 1 - MODEL_BYTES // SECTOR) // 2048) * 2048
    root_end = model_start - 1 if with_model_volume else last_lba
    if root_end - ROOT_START < 8 * 1024 * MIB // SECTOR:
        raise ValueError("Disk is too small for the installed system and requested model partition")
    model_uuid = str(uuid4()) if with_model_volume else None
    with TemporaryDirectory(prefix="inkling-installed-") as tmp:
        work = Path(tmp)
        gpt, esp, root = (work / name for name in ("gpt.raw", "esp.fat", "root.ext4"))
        allocate(gpt, disk_bytes)
        allocate(esp, ESP_SECTORS * SECTOR)
        root_bytes = (root_end - ROOT_START + 1) * SECTOR
        allocate(root, root_bytes // 4096 * 4096)
        model_partition = () if model_uuid is None else (
            f"--new=3:{model_start}:{model_start + MODEL_BYTES // SECTOR - 1}",
            "--typecode=3:0700", "--change-name=3:Inkling2 models", f"--partition-guid=3:{model_uuid}",
        )
        run("sgdisk", "--clear", f"--new=1:{ESP_START}:{ROOT_START - 1}", "--typecode=1:ef00",
            "--change-name=1:Inkling2 EFI", f"--new=2:{ROOT_START}:{root_end}", "--typecode=2:8300",
            "--change-name=2:Inkling2 Linux", *model_partition, str(gpt))
        run("sgdisk", "--verify", str(gpt))

        boot = rootfs / "boot"
        boot.mkdir(exist_ok=True)
        shutil.copyfile(iso_boot / "vmlinuz", boot / "vmlinuz")
        shutil.copyfile(iso_boot / "initramfs.img", boot / "initrd.img")
        fstab = f"UUID={root_uuid} / ext4 defaults 0 1\n"
        if model_uuid:
            fstab += (f"PARTUUID={model_uuid} /models ntfs3 "
                      "ro,nodev,nosuid,noexec,uid=0,gid=61000,fmask=0337,dmask=0027 0 0\n")
            (rootfs / "etc/inkling2/model-volume").write_text(model_uuid + "\n", encoding="ascii")
        (rootfs / "etc/fstab").write_text(fstab, encoding="ascii")
        cfg = work / "grub.cfg"
        cfg.write_text(grub_config(root_uuid, with_model_volume), encoding="ascii")
        efi = work / "BOOTX64.EFI"
        run("grub-mkstandalone", "-O", "x86_64-efi", "--locales=", "--fonts=",
            "--modules=part_gpt fat ext2 search_fs_uuid serial normal linux", "-o", str(efi),
            f"boot/grub/grub.cfg={cfg}")
        run("mkfs.fat", "-F", "32", "-n", "INKLINGEFI", str(esp))
        run("mmd", "-i", str(esp), "::/EFI", "::/EFI/BOOT")
        run("mcopy", "-i", str(esp), str(efi), "::/EFI/BOOT/BOOTX64.EFI")
        run("mke2fs", "-q", "-F", "-t", "ext4", "-L", "INKLING2_ROOT", "-U", root_uuid,
            "-m", "1", "-i", "262144", "-d", str(rootfs), str(root))
        run("e2fsck", "-f", "-n", str(root))

        regions = []
        with gzip.open(output / "disk-data.gz", "wb", compresslevel=1) as payload:
            for source, base in ((root, ROOT_START * SECTOR), (esp, ESP_START * SECTOR)):
                with source.open("rb", buffering=0) as image:
                    for start, end in data_ranges(image):
                        image.seek(start)
                        digest = copy_region(image, payload, end - start)
                        regions.append({"offset": base + start, "length": end - start, "sha256": digest})
            with gpt.open("rb") as image:
                for start, length in ((disk_bytes - 33 * SECTOR, 33 * SECTOR), (0, 34 * SECTOR)):
                    image.seek(start)
                    regions.append({"offset": start, "length": length,
                                    "sha256": copy_region(image, payload, length)})
    manifest = {"version": 3 if model_uuid else 1, "disk_bytes": disk_bytes,
                "sector_bytes": SECTOR, "regions": regions}
    if model_uuid:
        manifest["model_partition"] = {"number": 3, "guid": model_uuid,
                                       "offset": model_start * SECTOR, "length": MODEL_BYTES}
    (output / "layout.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")


def reconstruct(bundle: Path, image: Path) -> bool:
    manifest = json.loads((bundle / "layout.json").read_text(encoding="ascii"))
    allocate(image, manifest["disk_bytes"])
    with gzip.open(bundle / "disk-data.gz", "rb") as payload, image.open("r+b") as raw:
        for region in manifest["regions"]:
            raw.seek(region["offset"])
            digest = copy_region(payload, raw, region["length"])
            if digest != region["sha256"]:
                raise ValueError("Disk bundle digest mismatch")
        if payload.read(1):
            raise ValueError("Unexpected trailing data in disk bundle")
        model = manifest.get("model_partition")
        if model:
            # A small synthetic NTFS volume exercises the installed mount policy
            # without copying the private model or writing to a host disk.
            with TemporaryDirectory(prefix="inkling2-test-model-") as tmp:
                volume = Path(tmp) / "models.ntfs"
                allocate(volume, model["length"])
                run("mkntfs", "--force", "--fast", "--sector-size", str(SECTOR), str(volume))
                with volume.open("rb", buffering=0) as ntfs:
                    for start, end in data_ranges(ntfs):
                        raw.seek(model["offset"] + start)
                        ntfs.seek(start)
                        copy_region(ntfs, raw, end - start)
    return bool(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-bytes", type=int, required=True)
    parser.add_argument("--rootfs", type=Path, default=Path("/rootfs"))
    parser.add_argument("--iso-boot", type=Path, default=Path("/iso/boot"))
    parser.add_argument("--output", type=Path, default=Path("/out/install"))
    parser.add_argument("--with-model-volume", action="store_true")
    args = parser.parse_args()
    build(args.disk_bytes, args.rootfs, args.iso_boot, args.output, args.with_model_volume)
