import hashlib
import json

import pytest

from agent_tools.zcode_startup_package import prepare_package, verify_package


@pytest.fixture
def inputs(tmp_path):
    runtime = tmp_path / "input-python"
    sites = tmp_path / "input-sites"
    source = tmp_path / "input-source"
    for path in (runtime / "Lib/site-packages", sites, source / "agent_tools"):
        path.mkdir(parents=True)
    (runtime / "python.exe").write_bytes(b"interpreter fixture")
    (runtime / "python311.dll").write_bytes(b"interpreter DLL fixture")
    (runtime / "Lib/os.py").write_text("stdlib fixture")
    (runtime / "Lib/site-packages/unrelated.py").write_text("must not copy ambient packages")
    (sites / "dependency.py").write_text("pinned dependency fixture")
    for name in (
        "zcode_startup.py", "zcode_bridge.py", "zcode_startup_launch.ps1",
        "zcode_startup_bootstrap.pyw", "zcode_startup_protection.ps1",
    ):
        (source / "agent_tools" / name).write_text(name)
    return dict(
        runtime_home=runtime, site_packages=sites, source=source,
        destination=tmp_path / "package",
        settings={
            "ownerSid": "S-1-5-21-1-2-3-1001", "bridgeRoot": str(tmp_path / "bridge"),
            "producer": str(tmp_path / "ZCode.exe"), "producerSha256": "a" * 64,
            "bridgeConfigSha256": "b" * 64,
        },
        protect=lambda *args: None,
    )


def verify(inputs, receipt):
    return verify_package(
        inputs["destination"], receipt["manifestSha256"], inputs["settings"]["ownerSid"],
        protect=lambda *args: None,
    )


def test_complete_manifest_excludes_ambient_site_packages(inputs):
    receipt = prepare_package(**inputs)
    manifest = verify(inputs, receipt)
    paths = {entry["path"] for entry in manifest["files"]}
    assert "runtime/Lib/site-packages/dependency.py" in paths
    assert "runtime/Lib/site-packages/unrelated.py" not in paths
    assert "runtime/python311.dll" in paths
    assert "startup.json" in paths


@pytest.mark.parametrize("kind", ["edit", "extra", "missing"])
def test_changed_or_unlisted_bootstrap_bytes_refuse(inputs, kind):
    receipt = prepare_package(**inputs)
    file = inputs["destination"] / "runtime/Lib/site-packages/dependency.py"
    if kind == "edit":
        file.write_text("changed dependency")
    elif kind == "missing":
        file.unlink()
    else:
        (file.parent / "injected.py").write_text("extra import hook")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(inputs, receipt)


@pytest.mark.parametrize("hook", ["editable.pth", "distutils-precedence.pth", "pywin32.pth"])
def test_import_path_hooks_cannot_escape_into_mutable_source(inputs, hook):
    (inputs["site_packages"] / hook).write_text("import sys; sys.path.insert(0, 'C:/mutable')")
    with pytest.raises(ValueError, match="startup_package_invalid"):
        prepare_package(**inputs)
    assert not inputs["destination"].exists()


def test_known_pywin32_loader_is_preserved(inputs):
    (inputs["site_packages"] / "pywin32.pth").write_text(
        "win32\nwin32/lib\nPythonwin\nimport pywin32_bootstrap\n"
    )
    verify(inputs, prepare_package(**inputs))


def test_manifest_path_escape_is_rejected_even_with_matching_manifest_digest(inputs):
    receipt = prepare_package(**inputs)
    path = inputs["destination"] / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"][0]["path"] = "../outside"
    raw = json.dumps(manifest).encode()
    path.write_bytes(raw)
    receipt["manifestSha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="startup_package_invalid"):
        verify(inputs, receipt)


def test_destination_inside_input_cannot_recurse_or_mutate_source(inputs):
    inputs["destination"] = inputs["source"] / "nested"
    with pytest.raises(ValueError, match="startup_package_invalid"):
        prepare_package(**inputs)
