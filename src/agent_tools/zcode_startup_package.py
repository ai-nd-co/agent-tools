"""Prepare one immutable Windows Python runtime for the ZCode login wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

SCHEMA = "ai-nd-co.zcode-startup-package/v1"


def _fail() -> None:
    raise ValueError("startup_package_invalid")


def _file(path: Path) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode) or path.is_symlink() or before.st_nlink != 1
        or getattr(before, "st_file_attributes", 0) & 0x400
    ):
        _fail()
    value = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        _fail()
    return value


def _files(root: Path, skip_dirs: frozenset[str] = frozenset()) -> list[Path]:
    result = []
    for item in root.iterdir():
        if item.name.lower() in skip_dirs and item.is_dir():
            continue
        if item.is_symlink() or getattr(item.lstat(), "st_file_attributes", 0) & 0x400:
            _fail()
        if item.is_dir():
            result.extend(_files(item, skip_dirs))
        elif item.is_file():
            result.append(item)
        else:
            _fail()
    return sorted(result)


def protection(root: Path, owner_sid: str, operation: str = "Verify") -> None:
    from agent_tools.tts_startup import _windows_system_directory

    helper = Path(__file__).with_name("zcode_startup_protection.ps1")
    result = subprocess.run([
        str(_windows_system_directory() / "WindowsPowerShell/v1.0/powershell.exe"),
        "-NoProfile", "-NonInteractive", "-File", str(helper), "-Operation", operation,
        "-Root", str(root), "-OwnerSid", owner_sid,
    ], capture_output=True, timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or result.stdout != b'{"ok":true}':
        _fail()


def verify_package(
    root: Path, digest: str, owner_sid: str,
    *, protect: Callable[[Path, str, str], None] = protection,
) -> dict:
    protect(root, owner_sid, "Verify")
    raw = _file(root / "manifest.json")
    if len(raw) > 8 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
        _fail()
    manifest: dict = json.loads(raw)
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("files"), list):
        _fail()
    seen = set()
    for item in manifest["files"]:
        name = item.get("path", "")
        if (
            not name or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or name.lower() in seen
        ):
            _fail()
        seen.add(name.lower())
        content = _file(root / name)
        if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            _fail()
    required = {
        "runtime/python.exe", "launch.ps1", "bootstrap.py", "startup.json",
        "protect.ps1", "app/agent_tools/zcode_startup.py", "app/agent_tools/zcode_bridge.py",
    }
    actual = {p.relative_to(root).as_posix().lower() for p in _files(root)} - {"manifest.json"}
    if actual != seen or not required.issubset(seen):
        _fail()
    return manifest


def prepare_package(
    *, runtime_home: Path, site_packages: Path, source: Path, destination: Path,
    settings: dict, protect: Callable[[Path, str, str], None] = protection,
) -> dict:
    if not destination.is_absolute() or destination.exists():
        _fail()
    if set(settings) != {
        "ownerSid", "bridgeRoot", "bridgeConfigSha256", "producer", "producerSha256"
    }:
        _fail()
    if (
        not re.fullmatch(r"S-1-5-21-\d+-\d+-\d+-\d+", settings["ownerSid"])
        or any(not re.fullmatch(r"[a-f0-9]{64}", settings[key])
               for key in ("bridgeConfigSha256", "producerSha256"))
        or any(not Path(settings[key]).is_absolute() for key in ("bridgeRoot", "producer"))
        or any(destination.resolve().is_relative_to(path.resolve())
               for path in (runtime_home, site_packages, source))
    ):
        _fail()
    if (runtime_home / "pyvenv.cfg").exists():
        _fail()
    # Editable environment hooks must never redirect a Highest import to a checkout.
    for path in _files(site_packages):
        if path.suffix.lower() == ".pth":
            lines = {line.strip().replace("\\", "/").lower()
                     for line in _file(path).decode("utf-8").splitlines()
                     if line.strip() and not line.strip().startswith("#")}
            if path.name != "pywin32.pth" or not lines.issubset({
                "win32", "win32/lib", "pythonwin", "import pywin32_bootstrap",
            }):
                _fail()
    destination.mkdir()
    protect(destination, settings["ownerSid"], "Protect")

    def copy_tree(start: Path, end: Path, *, omit_sites: bool = False) -> None:
        excluded = frozenset({"__pycache__", "site-packages"} if omit_sites else {"__pycache__"})
        for file in _files(start, excluded):
            relative = file.relative_to(start)
            if "__pycache__" in relative.parts or file.suffix == ".pyc":
                continue
            if omit_sites and relative.parts[0].lower() == "site-packages":
                continue
            target = end / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(_file(file))

    runtime = destination / "runtime"
    runtime.mkdir()
    for file in runtime_home.iterdir():
        if file.is_file() and (
            file.name in {"python.exe", "pythonw.exe"}
            or file.name.startswith(("python", "vcruntime")) and file.suffix == ".dll"
        ):
            (runtime / file.name).write_bytes(_file(file))
    for name in ("Lib", "DLLs", "tcl"):
        if (runtime_home / name).exists():
            copy_tree(runtime_home / name, runtime / name, omit_sites=name == "Lib")
    copy_tree(site_packages, runtime / "Lib/site-packages")
    copy_tree(source / "agent_tools", destination / "app/agent_tools")
    for original, target in (
        ("zcode_startup_launch.ps1", "launch.ps1"),
        ("zcode_startup_bootstrap.pyw", "bootstrap.py"),
        ("zcode_startup_protection.ps1", "protect.ps1"),
    ):
        (destination / target).write_bytes(_file(source / "agent_tools" / original))
    (destination / "startup.json").write_text(json.dumps(settings), encoding="utf-8")
    files: list[dict] = []
    manifest = {"schema": SCHEMA, "files": files}
    for path in _files(destination):
        content = _file(path)
        files.append({
            "path": path.relative_to(destination).as_posix(), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    raw = json.dumps(manifest, separators=(",", ":")).encode()
    (destination / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    protect(destination, settings["ownerSid"], "Protect")
    verify_package(destination, digest, settings["ownerSid"], protect=protect)
    return {"root": str(destination), "manifestSha256": digest, "files": len(files)}


def prepare_task_receipt(
    package: Path, manifest_sha256: str, owner_sid: str, destination: Path,
    *, task_name: str = "ai-nd-co ZCode Owner Logon",
    probe_output: Path | None = None,
) -> dict:
    from agent_tools.tts_startup import _windows_system_directory
    from agent_tools.zcode_startup import render_task_xml

    if not re.fullmatch(r"ai-nd-co ZCode (Owner Logon|Fixture [a-f0-9]{32})", task_name):
        _fail()
    if destination.exists() or not destination.is_absolute():
        _fail()
    verify_package(package, manifest_sha256, owner_sid)
    command = _windows_system_directory() / "WindowsPowerShell/v1.0/powershell.exe"
    arguments = [
        "-NoProfile", "-NonInteractive", "-File", str(package / "launch.ps1"),
        "-ManifestSha256", manifest_sha256, "-OwnerSid", owner_sid,
    ]
    result = subprocess.run(
        [str(command), *arguments, "-Operation", "Probe"],
        capture_output=True, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if result.returncode or json.loads(result.stdout).get("code") != "package_ready":
        _fail()
    if probe_output is not None:
        if not task_name.startswith("ai-nd-co ZCode Fixture ") or not probe_output.is_absolute():
            _fail()
        arguments += ["-Operation", "Probe", "-ProbeOutput", str(probe_output)]
    destination.mkdir()
    protection(destination, owner_sid, "Protect")
    receipt = {
        "schema": "ai-nd-co.zcode-startup-task/v1", "taskName": task_name, "ownerSid": owner_sid,
        "xml": render_task_xml(owner_sid, command, arguments, package),
    }
    raw = json.dumps(receipt).encode()
    with (destination / "receipt.json").open("xb") as output:
        output.write(raw)
    protection(destination, owner_sid, "Protect")
    return {"root": str(destination), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-home", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare_package(
            runtime_home=args.runtime_home, site_packages=args.site_packages, source=args.source,
            destination=args.destination, settings=json.loads(args.settings.read_text("utf-8")),
        )
        print(json.dumps(result))
        return 0
    except Exception:
        print('{"code":"startup_package_invalid"}')
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
