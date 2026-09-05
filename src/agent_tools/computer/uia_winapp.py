from __future__ import annotations

import base64
import ctypes
import gzip
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agent_tools.computer.capture_records import (
    _atomic_private_write,
    _capture_store_lock,
    _prepare_capture_store_root,
    _set_owner_only_acl,
)
from agent_tools.computer.element_refs import ElementReferenceStore, attach_element_references
from agent_tools.computer.models import (
    MAX_CLASS_NAME_CHARS,
    MAX_TITLE_CHARS,
    Bounds,
    ComputerError,
    WindowIdentity,
    bounded_text,
)
from agent_tools.computer.process_io import (
    MIN_PROCESS_LIFECYCLE_SECONDS,
    ProcessInputStagingError,
    close_preloaded_process_stdin,
    close_stream_os_enforced,
    preloaded_process_stdin,
)
from agent_tools.computer.win32_backend import Win32Backend

WINAPP_VERSION = "0.6.0"
WINAPP_RELEASE_ASSET = "winappcli-x64.zip"
WINAPP_RELEASE_URL = (
    "https://github.com/microsoft/WinAppCli/releases/download/"
    f"v{WINAPP_VERSION}/{WINAPP_RELEASE_ASSET}"
)
WINAPP_RELEASE_SHA256 = "f6dc42e3b4e4709c8f617003008e2cfdd9a51735e04e7170d60edda258db78a8"
WINAPP_EXE_SHA256 = "0aa2a43086ad07c7fdc35c94ecca0a586e815b1ab46e9b3109d73123e9190c90"
WINAPP_COMPANION = "libSkiaSharp.dll"
WINAPP_COMPANION_SHA256 = "9a0d95e8caaa852c70d085af6a40a744242172ad9ea3fd6bc7599875a8a1dbcd"
WINAPP_ENV = "AGENT_TOOLS_WINAPP_PATH"
LOCAL_WINAPP = (
    Path(__file__).resolve().parents[3]
    / ".tools"
    / "winapp"
    / f"v{WINAPP_VERSION}"
    / "extracted"
    / "winapp.exe"
)
INSTALL_HINT = (
    f"AgentTools installs the pinned official WinAppCLI v{WINAPP_VERSION} backend "
    "automatically on first semantic use; retry while the network is available."
)

MAX_WINAPP_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_WINAPP_EXTRACTED_BYTES = 48 * 1024 * 1024
WINAPP_DOWNLOAD_TIMEOUT_SECONDS = 90.0
WINAPP_CACHE_MANIFEST = "install.json"

MAX_BACKEND_OUTPUT_BYTES = 1024 * 1024
MAX_PROPERTY_OUTPUT_BYTES = 512 * 1024
MAX_INSPECT_ELEMENTS = 200
MAX_SCROLL_AREAS = 50
MAX_READ_CHARS = 50_000
MAX_SELECTOR_CHARS = 240
MAX_ELEMENT_ID_CHARS = 4_096
DEFAULT_TIMEOUT_SECONDS = 8.0
VERSION_TIMEOUT_SECONDS = 1.5
CLEANUP_TIMEOUT_SECONDS = 2.0
WINAPP_PROBE_MAX_SECONDS = VERSION_TIMEOUT_SECONDS + CLEANUP_TIMEOUT_SECONDS + 1.0
NATIVE_UIA_TIMEOUT_SECONDS = 4.0
WORKER_MUTEX_NAME = r"Local\ai-nd-co.agent-tools.computer-worker.v1"
_INVARIANT_DOUBLE_BITS_RE = re.compile(r"[0-9A-F]{16}\Z")
_ANCESTOR_SIGNATURE_RE = re.compile(r"sha256:[0-9a-f]{24}\Z")
_INVARIANT_DOUBLE_RE = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_MAX_INVARIANT_DOUBLE_CHARS = 24
_MAX_VALUE_CODEPOINTS = 16_000
_MAX_VALUE_UTF8_BYTES = _MAX_VALUE_CODEPOINTS * 4
_MAX_VALUE_UTF16_CHARS = _MAX_VALUE_CODEPOINTS * 2
_PATTERN_QUERY_STATUSES = frozenset({"available", "absent", "failed"})
_PRIVATE_ACTION_PROPERTY_KEYS = frozenset(
    {
        "element_reference",
        "require_title_match",
        "range_value_bits",
        "value_pattern_status",
        "range_value_pattern_status",
    }
)
_ACTION_PREFLIGHT_CAPABILITY_ERRORS = frozenset(
    {
        "backend_version_mismatch",
        "capability_missing",
        "native_uia_unavailable",
        "uia_identity_unavailable",
        "uia_provider_unavailable",
    }
)

_NATIVE_SAFETY_CSHARP = r"""
using System;
using System.Runtime.InteropServices;
using System.Text;

[ComImport]
[Guid("ff48dba4-60ef-4201-aa87-54103eef594e")]
[ClassInterface(ClassInterfaceType.None)]
internal class AgentToolsCuiAutomation {}

[ComImport]
[Guid("352ffba8-0973-437c-a61f-f64cafd81df9")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IAgentToolsUiaCondition {}

[ComImport]
[Guid("30cbe57d-d9d0-452a-ab13-7ac5ac4825ee")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IAgentToolsUiaAutomation {
    void CompareElements();
    void CompareRuntimeIds();
    IAgentToolsUiaElement GetRootElement();
    IAgentToolsUiaElement ElementFromHandle(IntPtr hwnd);
    void ElementFromPoint();
    void GetFocusedElement();
    void GetRootElementBuildCache();
    void ElementFromHandleBuildCache();
    void ElementFromPointBuildCache();
    void GetFocusedElementBuildCache();
    void CreateTreeWalker();
    void get_ControlViewWalker();
    void get_ContentViewWalker();
    void get_RawViewWalker();
    void get_RawViewCondition();
    void get_ControlViewCondition();
    void get_ContentViewCondition();
    void CreateCacheRequest();
    void CreateTrueCondition();
    void CreateFalseCondition();
    IAgentToolsUiaCondition CreatePropertyCondition(
        int propertyId,
        [MarshalAs(UnmanagedType.Struct)] object value);
}

[ComImport]
[Guid("d22108aa-8ac5-49a5-837b-37bbb3d7591e")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IAgentToolsUiaElement {
    void SetFocus();
    [return: MarshalAs(UnmanagedType.SafeArray, SafeArraySubType = VarEnum.VT_I4)]
    int[] GetRuntimeId();
    IAgentToolsUiaElement FindFirst(int scope, IAgentToolsUiaCondition condition);
    IAgentToolsUiaElementArray FindAll(int scope, IAgentToolsUiaCondition condition);
    void FindFirstBuildCache();
    void FindAllBuildCache();
    void BuildUpdatedCache();
    [return: MarshalAs(UnmanagedType.Struct)]
    object GetCurrentPropertyValue(int propertyId);
    [return: MarshalAs(UnmanagedType.Struct)]
    object GetCurrentPropertyValueEx(
        int propertyId,
        [MarshalAs(UnmanagedType.Bool)] bool ignoreDefaultValue);
    void GetCachedPropertyValue();
    void GetCachedPropertyValueEx();
    IAgentToolsLegacyPattern GetCurrentPatternAs(int patternId, ref Guid riid);
}

[ComImport]
[Guid("14314595-b4bc-4055-95f2-58f2e42c9855")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IAgentToolsUiaElementArray {
    int get_Length();
    IAgentToolsUiaElement GetElement(int index);
}

[ComImport]
[Guid("828055ad-355b-4435-86d5-3b51c14a9b1b")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IAgentToolsLegacyPattern {
    void Select(int flags);
    void DoDefaultAction();
    void SetValue([MarshalAs(UnmanagedType.LPWStr)] string value);
    int get_CurrentChildId();
    [return: MarshalAs(UnmanagedType.BStr)] string get_CurrentName();
    [return: MarshalAs(UnmanagedType.BStr)] string get_CurrentValue();
    [return: MarshalAs(UnmanagedType.BStr)] string get_CurrentDescription();
    uint get_CurrentRole();
    uint get_CurrentState();
}

public sealed class AgentToolsLegacySafetyException : Exception {
    public string Code { get; private set; }
    public AgentToolsLegacySafetyException(string code) : base(code) {
        Code = code;
    }
}

public static class AgentToolsLegacyUia {
    const int AutomationIdProperty = 30011;
    const int LegacyPattern = 10018;
    const int Descendants = 4;
    const int UiaNotSupported = unchecked((int)0x80040204);
    static readonly Guid LegacyInterface =
        new Guid("828055ad-355b-4435-86d5-3b51c14a9b1b");

    static IAgentToolsUiaElement Resolve(
        IAgentToolsUiaAutomation automation,
        IntPtr hwnd,
        string automationId,
        string expectedRuntimeId) {
        var root = automation.ElementFromHandle(hwnd);
        if (root == null) throw new AgentToolsLegacySafetyException("stale_window");
        var condition = automation.CreatePropertyCondition(
            AutomationIdProperty,
            automationId);
        if (condition == null)
            throw new AgentToolsLegacySafetyException("target_safety_unavailable");
        var matches = root.FindAll(Descendants, condition);
        if (matches == null)
            throw new AgentToolsLegacySafetyException("target_safety_unavailable");
        IAgentToolsUiaElement element = null;
        var count = matches.get_Length();
        for (var index = 0; index < count; index++) {
            var candidate = matches.GetElement(index);
            if (candidate == null) continue;
            var runtimeId = string.Join(",", candidate.GetRuntimeId());
            if (!String.Equals(runtimeId, expectedRuntimeId, StringComparison.Ordinal)) continue;
            if (element != null)
                throw new AgentToolsLegacySafetyException("ambiguous_element");
            element = candidate;
        }
        if (element == null)
            throw new AgentToolsLegacySafetyException("stale_element");
        return element;
    }

    static IAgentToolsLegacyPattern Pattern(IAgentToolsUiaElement element) {
        var iid = LegacyInterface;
        return element.GetCurrentPatternAs(LegacyPattern, ref iid);
    }

    public static long State(
        IntPtr hwnd,
        string automationId,
        string expectedRuntimeId) {
        var automation = (IAgentToolsUiaAutomation)new AgentToolsCuiAutomation();
        var element = Resolve(automation, hwnd, automationId, expectedRuntimeId);
        IAgentToolsLegacyPattern pattern;
        try {
            pattern = Pattern(element);
        } catch (COMException error) {
            if (error.HResult == UiaNotSupported) return -1;
            throw;
        }
        if (pattern == null) return -1;
        return pattern.get_CurrentState();
    }

    public static string Value(
        IntPtr hwnd,
        string automationId,
        string expectedRuntimeId) {
        var automation = (IAgentToolsUiaAutomation)new AgentToolsCuiAutomation();
        var element = Resolve(automation, hwnd, automationId, expectedRuntimeId);
        var pattern = Pattern(element);
        if (pattern == null) throw new InvalidOperationException("legacy pattern unavailable");
        return pattern.get_CurrentValue();
    }

}

public static class AgentToolsNativeSafety {
    const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
    const uint TOKEN_QUERY = 0x0008;
    const int TOKEN_INTEGRITY_LEVEL = 25;
    const uint DESKTOP_READOBJECTS = 0x0001;
    const int UOI_NAME = 2;

    [DllImport("user32.dll")]
    public static extern bool IsWindow(IntPtr hwnd);
    [DllImport("user32.dll")]
    public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int GetWindowTextW(IntPtr hwnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int GetClassNameW(IntPtr hwnd, StringBuilder text, int maxCount);
    [DllImport("user32.dll", SetLastError = true)]
    static extern IntPtr OpenInputDesktop(uint flags, bool inherit, uint access);
    [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool GetUserObjectInformationW(
        IntPtr handle, int index, StringBuilder value, uint length, out uint needed);
    [DllImport("user32.dll")]
    static extern bool CloseDesktop(IntPtr desktop);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll")]
    static extern IntPtr GetCurrentProcess();
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr handle);
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool GetTokenInformation(
        IntPtr token, int informationClass, IntPtr information,
        int informationLength, out int returnLength);
    [DllImport("advapi32.dll")]
    static extern IntPtr GetSidSubAuthorityCount(IntPtr sid);
    [DllImport("advapi32.dll")]
    static extern IntPtr GetSidSubAuthority(IntPtr sid, uint index);

    public static string WindowText(IntPtr hwnd) {
        var value = new StringBuilder(2048);
        return GetWindowTextW(hwnd, value, value.Capacity) > 0 ? value.ToString() : "";
    }

    public static string WindowClass(IntPtr hwnd) {
        var value = new StringBuilder(512);
        return GetClassNameW(hwnd, value, value.Capacity) > 0 ? value.ToString() : "";
    }

    public static string InputDesktopName() {
        IntPtr desktop = OpenInputDesktop(0, false, DESKTOP_READOBJECTS);
        if (desktop == IntPtr.Zero) return "";
        try {
            uint needed;
            GetUserObjectInformationW(desktop, UOI_NAME, null, 0, out needed);
            if (needed == 0) return "";
            var value = new StringBuilder((int)(needed / 2) + 1);
            return GetUserObjectInformationW(
                desktop, UOI_NAME, value, (uint)(value.Capacity * 2), out needed)
                ? value.ToString() : "";
        } finally {
            CloseDesktop(desktop);
        }
    }

    public static int IntegrityRid(uint pid) {
        bool closeProcess = pid != 0;
        IntPtr process = closeProcess
            ? OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid)
            : GetCurrentProcess();
        if (process == IntPtr.Zero) return -1;
        IntPtr token;
        try {
            if (!OpenProcessToken(process, TOKEN_QUERY, out token)) return -1;
        } finally {
            if (closeProcess) CloseHandle(process);
        }
        try {
            int needed;
            GetTokenInformation(token, TOKEN_INTEGRITY_LEVEL, IntPtr.Zero, 0, out needed);
            if (needed <= 0) return -1;
            IntPtr buffer = Marshal.AllocHGlobal(needed);
            try {
                if (!GetTokenInformation(
                    token, TOKEN_INTEGRITY_LEVEL, buffer, needed, out needed)) return -1;
                IntPtr sid = Marshal.ReadIntPtr(buffer);
                IntPtr countPointer = GetSidSubAuthorityCount(sid);
                if (countPointer == IntPtr.Zero) return -1;
                byte count = Marshal.ReadByte(countPointer);
                if (count == 0) return -1;
                IntPtr ridPointer = GetSidSubAuthority(sid, (uint)(count - 1));
                return ridPointer == IntPtr.Zero ? -1 : Marshal.ReadInt32(ridPointer);
            } finally {
                Marshal.FreeHGlobal(buffer);
            }
        } finally {
            CloseHandle(token);
        }
    }
}
"""

_VERSION_RE = re.compile(r"(?m)^([0-9]+\.[0-9]+\.[0-9]+)\s*$")
_CREDENTIAL_KEYS = r"password|passwd|pwd|token|secret|api[_-]?key|authorization"
_QUOTED_CREDENTIAL_PREFIX_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?(?:{_CREDENTIAL_KEYS})[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])"
)
_AUTHORIZATION_SCHEME_RE = re.compile(
    r"(?i)(?P<prefix>[\"']?authorization[\"']?\s*[:=]\s*)"
    r"(?:Basic|Bearer)\s+[A-Za-z0-9._~+/=-]{4,}"
)
_UNQUOTED_CREDENTIAL_RE = re.compile(
    rf"(?i)(?P<prefix>[\"']?(?:{_CREDENTIAL_KEYS})[\"']?\s*[:=]\s*)"
    r"(?P<value>(?![\"'])[^\s,;}}\]]+)"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_ACTIONABLE_TYPES = {
    "Button",
    "CheckBox",
    "ComboBox",
    "Document",
    "Edit",
    "Hyperlink",
    "ListItem",
    "MenuItem",
    "RadioButton",
    "Slider",
    "TabItem",
    "TreeItem",
}
_DEGRADABLE_SCROLL_CANDIDATE_ERRORS = {
    "element_not_found",
    "stale_element",
}


@dataclass(frozen=True)
class WinAppBinding:
    path: Path
    source: str
    version: str
    sha256: str

    def public(self) -> dict[str, Any]:
        return {
            "name": "winapp",
            "version": self.version,
            "source": self.source,
            "sha256": self.sha256,
            "actions_exposed": True,
        }


@dataclass(frozen=True)
class NativeNavigationBinding:
    provider: str = "native_uia_raw"

    def public(self) -> dict[str, Any]:
        return {
            "name": self.provider,
            "version": "windows-uia-managed-v1",
            "source": "windows_system",
            "actions_exposed": True,
        }


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


class WinAppAdapter:
    """Replaceable AgentTools adapter around pinned Microsoft WinApp CLI."""

    def __init__(self, backend: Win32Backend | None = None) -> None:
        self.backend = backend or Win32Backend()
        self._verified_binding: WinAppBinding | None = None

    def probe(self) -> dict[str, Any]:
        try:
            binding = self.require(auto_install=False)
        except ComputerError as exc:
            if exc.code == "backend_version_mismatch":
                status = "version_mismatch"
            elif exc.code == "capability_missing":
                status = "missing"
            else:
                status = "error"
            return {
                "available": False,
                "status": status,
                "error_code": exc.code,
                "required_version": WINAPP_VERSION,
                "capabilities": [],
                "install_hint": INSTALL_HINT,
            }
        return {
            "available": True,
            "status": "ready",
            **binding.public(),
            "capabilities": [
                "inspect",
                "read",
                "scroll-areas",
                "invoke",
                "set-value",
                "scroll",
            ],
            "actions_exposed": True,
        }

    def require(
        self,
        *,
        deadline: float | None = None,
        auto_install: bool = True,
    ) -> WinAppBinding:
        _require_deadline(deadline)
        if sys.platform != "win32":
            raise ComputerError(
                "capability_missing",
                "WinApp UI Automation is available only on Windows. " + INSTALL_HINT,
                exit_code=2,
            )
        candidate, source = _discover_candidate()
        _require_deadline(deadline)
        if candidate is None and source == "missing" and auto_install:
            candidate = _install_pinned_winapp()
            source = "managed_cache"
        if candidate is None:
            raise ComputerError(
                "capability_missing",
                "The pinned WinApp UI Automation backend is unavailable. " + INSTALL_HINT,
                exit_code=2,
            )
        digest = _sha256(candidate) if deadline is None else _sha256(candidate, deadline=deadline)
        if digest != WINAPP_EXE_SHA256 and source == "managed_cache" and auto_install:
            candidate = _install_pinned_winapp()
            digest = _sha256(candidate)
        if digest != WINAPP_EXE_SHA256:
            raise ComputerError(
                "backend_version_mismatch",
                "The discovered WinApp binary does not match the pinned v0.6.0 release. "
                + INSTALL_HINT,
                exit_code=2,
            )
        try:
            _verify_winapp_companion(candidate, deadline=deadline)
        except ComputerError:
            if source != "managed_cache" or not auto_install:
                raise
            candidate = _install_pinned_winapp()
            digest = _sha256(candidate)
            _verify_winapp_companion(candidate, deadline=deadline)
        cached = self._verified_binding
        if cached is not None and cached.path == candidate and cached.sha256 == digest:
            _require_deadline(deadline)
            return cached
        result = _run_process(
            candidate,
            ["--version"],
            timeout_seconds=_deadline_timeout(VERSION_TIMEOUT_SECONDS, deadline),
            max_output_bytes=64 * 1024,
            deadline=deadline,
        )
        _require_deadline(deadline)
        if result.returncode != 0:
            raise ComputerError(
                "uia_version_probe_failed",
                "The WinApp backend version probe failed.",
                exit_code=2,
            )
        version = _parse_version(result)
        if version != WINAPP_VERSION:
            raise ComputerError(
                "backend_version_mismatch",
                f"WinApp {WINAPP_VERSION} is required, but {version or 'an unknown version'} "
                "was discovered. " + INSTALL_HINT,
                exit_code=2,
            )
        binding = WinAppBinding(candidate, source, version, digest)
        self._verified_binding = binding
        _require_deadline(deadline)
        return binding

    def screenshot(
        self,
        identity: WindowIdentity,
        *,
        output: Path,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        deadline: float | None = None,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        """Capture an exact window through WinApp without requesting focus."""
        binding, payload = self._target_json(
            identity,
            ["ui", "screenshot", "--output", str(output)],
            max_output_bytes=128 * 1024,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
            screenshot_target=True,
        )
        width = payload.get("width")
        height = payload.get("height")
        hwnd = payload.get("hwnd")
        pid = payload.get("processId")
        file_path = payload.get("filePath")
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or type(hwnd) is not int
            or hwnd != identity.hwnd
            or type(pid) is not int
            or pid != identity.pid
            or not isinstance(file_path, str)
            or not file_path
        ):
            raise _invalid_response()
        try:
            reported_output = Path(file_path).resolve(strict=False)
            requested_output = output.resolve(strict=False)
        except OSError as exc:
            raise _invalid_response() from exc
        if os.path.normcase(str(reported_output)) != os.path.normcase(str(requested_output)):
            raise _invalid_response()
        return binding, {
            "width": width,
            "height": height,
            "output": requested_output,
            "capture_api": "windows_graphics_capture_preferred",
            "effective_capture_api": "unreported_by_winapp_0.6.0",
            "upstream_fallback_possible": "PrintWindow",
        }

    def inspect(
        self,
        identity: WindowIdentity,
        *,
        depth: int,
        interactive: bool,
        max_elements: int,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        arguments = ["ui", "inspect", "--depth", str(depth), "--hide-offscreen"]
        if interactive:
            arguments.append("--interactive")
        binding, payload = self._target_json(identity, arguments)
        window = _single_window(payload, identity)
        roots = window.get("elements")
        if not isinstance(roots, list):
            raise _invalid_response()
        flattened, traversal_truncated = _flatten_elements(
            roots,
            max_elements=max_elements,
            provider_filtered_hierarchy=interactive,
        )
        flattened.sort(
            key=lambda item: (
                bool(item.get("offscreen")),
                not bool(item.get("actionable")),
                int(item.get("depth") or 0),
                int(item.pop("_order")),
            )
        )
        returned = flattened[:max_elements]
        truncated = traversal_truncated or len(flattened) > max_elements
        return binding, {
            "depth": depth,
            "interactive": interactive,
            "count": len(returned),
            "truncated": truncated,
            "provider_filter_applied": interactive,
            "provider_bound": {
                "max_elements": max_elements,
                "interactive_filter_forwarded": interactive,
                "upstream_native_max_supported": False,
                "enforced_before_agenttools_response_serialization": True,
                "external_output_guard": "byte_cap_and_process_termination",
            },
            "values_omitted": True,
            "elements": returned,
        }

    def read(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        max_chars: int,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        resolved_selector = _validate_selector(selector)
        binding, password_payload = self._target_json(
            identity,
            ["ui", "get-property", resolved_selector, "--property", "IsPassword"],
            max_output_bytes=64 * 1024,
        )
        element_id, password_before = _password_state(password_payload)
        exact_before: dict[str, Any] | None = None
        final_exact: dict[str, Any] | None = None
        try:
            exact_binding, exact_element_id, candidate = self._element_properties(
                identity, resolved_selector
            )
        except ComputerError as exc:
            if password_before:
                return binding, _password_redacted(resolved_selector, element_id)
            if exc.code == "native_uia_unavailable":
                raise ComputerError(
                    "uia_sensitive_state_unavailable",
                    "The element's password state could not be verified by trusted UI "
                    "Automation; no text was read.",
                ) from exc
            raise
        else:
            if exact_element_id != element_id:
                raise ComputerError(
                    "stale_element",
                    "The UI Automation element changed; inspect again.",
                )
            if candidate.get("password") is not False:
                return exact_binding, _password_redacted(resolved_selector, exact_element_id)
            binding = exact_binding
            exact_before = candidate
        exact_read_method: str | None = None
        if exact_before is not None:
            exact_read_method = _effective_value_read_method(exact_before)
        range_read = exact_read_method == "uia.RangeValuePattern"
        if range_read:
            assert exact_before is not None
            binding, after_element_id, range_after = self._element_properties(
                identity, resolved_selector
            )
            if after_element_id != element_id or range_after.get("runtime_id") != exact_before.get(
                "runtime_id"
            ):
                raise ComputerError(
                    "stale_element",
                    "The UI Automation element changed; inspect again.",
                )
            if range_after.get("password") is not False:
                return binding, _password_redacted(resolved_selector, after_element_id)
            if _effective_value_read_method(range_after) != exact_read_method:
                raise ComputerError(
                    "uia_pattern_changed",
                    "The element's effective value pattern changed during the read.",
                )
            text = _range_value_text(range_after.get("range_value_bits"))
            value_element_id = after_element_id
            final_exact = range_after
        else:
            binding, value_payload = self._target_json(
                identity,
                ["ui", "get-value", resolved_selector],
                max_output_bytes=MAX_PROPERTY_OUTPUT_BYTES,
            )
            value_element_id = _exact_element_id(value_payload.get("elementId"))
            if value_element_id != element_id:
                raise ComputerError(
                    "stale_element",
                    "The UI Automation element changed; inspect again.",
                )
            text = _value_payload_text(value_payload)
            if exact_before is not None:
                binding, after_element_id, exact_after = self._element_properties(
                    identity, resolved_selector
                )
                if after_element_id != element_id or exact_after.get(
                    "runtime_id"
                ) != exact_before.get("runtime_id"):
                    raise ComputerError(
                        "stale_element",
                        "The UI Automation element changed; inspect again.",
                    )
                if exact_after.get("password") is not False:
                    return binding, _password_redacted(resolved_selector, after_element_id)
                if _effective_value_read_method(exact_after) != exact_read_method:
                    raise ComputerError(
                        "uia_pattern_changed",
                        "The element's effective value pattern changed during the read.",
                    )
                final_exact = exact_after
            else:
                binding, password_after_payload = self._target_json(
                    identity,
                    [
                        "ui",
                        "get-property",
                        resolved_selector,
                        "--property",
                        "IsPassword",
                    ],
                    max_output_bytes=64 * 1024,
                )
                after_element_id, password_after = _password_state(password_after_payload)
                if after_element_id != element_id:
                    raise ComputerError(
                        "stale_element",
                        "The UI Automation element changed; inspect again.",
                    )
                if password_after:
                    return binding, _password_redacted(resolved_selector, after_element_id)
        text, redacted = redact_credential_like_text(text)
        result = {
            "element": resolved_selector,
            "resolved_element": value_element_id,
            "text": text[:max_chars],
            "char_count": len(text),
            "truncated": len(text) > max_chars,
            "redacted": redacted,
            "redaction_reason": "credential_like_text" if redacted else None,
        }
        if final_exact is not None:
            result["_element_reference_identity"] = _read_reference_identity(final_exact)
        return binding, result

    def scroll_areas(
        self,
        identity: WindowIdentity,
        *,
        max_elements: int,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        search_limit = min(100, max(max_elements * 3, max_elements))
        binding, payload = self._target_json(
            identity,
            ["ui", "search", "scroll", "--max", str(search_limit)],
        )
        matches = payload.get("matches")
        if not isinstance(matches, list):
            raise _invalid_response()
        has_more = payload.get("hasMore")
        match_count = payload.get("matchCount")
        if type(has_more) is not bool or type(match_count) is not int or match_count < len(matches):
            raise _invalid_response()
        candidates = _deduplicate_candidates(matches)
        warnings: list[str] = []
        areas: list[dict[str, Any]] = []
        for candidate in candidates:
            if len(areas) >= max_elements:
                break
            selector, _selector_overflow = _exact_optional_text(
                candidate.get("selector"), MAX_SELECTOR_CHARS
            )
            if not selector:
                continue
            candidate_bounds = _bounds_from_element(candidate)
            if (
                _parse_bool(candidate.get("isOffscreen")) is True
                or candidate_bounds.width <= 0
                or candidate_bounds.height <= 0
            ):
                continue
            try:
                _property_binding, property_payload = self._target_json(
                    identity,
                    ["ui", "get-property", selector],
                    max_output_bytes=MAX_PROPERTY_OUTPUT_BYTES,
                )
                properties = property_payload.get("properties")
                if not isinstance(properties, dict):
                    raise _invalid_response()
                if not _has_semantic_scroll_state(properties):
                    continue
                identity_properties = {
                    **properties,
                    "AutomationId": properties.get("AutomationId") or candidate.get("automationId"),
                    "ControlType": properties.get("ControlType") or candidate.get("type"),
                    "ClassName": properties.get("ClassName") or candidate.get("className"),
                }
                native = self._native_element_state(
                    identity,
                    properties=identity_properties,
                    provider_identity=_scroll_provider_identity(
                        candidate,
                        identity_properties,
                    ),
                )
                if native is None:
                    raise ComputerError(
                        "native_uia_unavailable",
                        "The trusted Windows UI Automation bridge is unavailable.",
                    )
                area = _normalize_scroll_area(
                    candidate,
                    properties,
                    native,
                    identity,
                    binding,
                )
            except ComputerError as exc:
                if exc.code not in _DEGRADABLE_SCROLL_CANDIDATE_ERRORS:
                    raise
                warnings.append(exc.code)
                continue
            if area is not None:
                areas.append(area)
        return binding, {
            "count": len(areas),
            "truncated": bool(
                has_more
                or match_count > len(matches)
                or (len(candidates) > len(areas) and len(areas) >= max_elements)
            ),
            "searched_candidates": match_count,
            "degradation_warnings": sorted(set(warnings)),
            "areas": areas,
        }

    def invoke(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        mutation_guard: Callable[[], None] | None = None,
        disable_marker: Path | None = None,
        element_reference: dict[str, Any] | None = None,
        require_title_match: bool = False,
    ) -> tuple[WinAppBinding | NativeNavigationBinding, dict[str, Any]]:
        if _is_native_provider_locator(selector):
            return self._invoke_native_navigation(
                identity,
                selector=selector,
                mutation_guard=mutation_guard,
                disable_marker=disable_marker,
                element_reference=element_reference,
                require_title_match=require_title_match,
            )
        _reject_read_only_provider_locator(selector)
        resolved_selector = _validate_selector(selector)
        binding, element_id, before = self._action_properties(
            identity,
            resolved_selector,
            element_reference=element_reference,
            action_preflight=True,
        )
        before["require_title_match"] = require_title_match
        expected_pattern = _invoke_pattern(before)
        if expected_pattern is None:
            raise _semantic_action_unavailable(
                "invoke_pattern_unavailable",
                "No supported semantic invocation pattern was verified for the element.",
            )
        action_payload = self._native_uia_action(
            identity,
            properties=before,
            request={"action": "invoke"},
            method=expected_pattern,
            mutation_guard=mutation_guard,
            disable_marker=disable_marker,
        )
        self._validate_native_action_payload(
            action_payload,
            properties=before,
            method=expected_pattern,
        )
        method = expected_pattern
        try:
            _after_binding, after_element_id, after = self._action_properties(
                identity,
                resolved_selector,
                element_reference=element_reference,
            )
        except ComputerError as exc:
            if exc.code in {"element_not_found", "stale_element", "stale_window"}:
                return binding, {
                    "element": resolved_selector,
                    "resolved_element": element_id,
                    "pattern": method,
                    "before": _public_action_properties(before),
                    "after": None,
                    "postcondition_verified": False,
                    "method": method,
                    "outcome": "delivery_only",
                }
            raise _mark_delivery_error(exc, method) from exc
        if (
            element_reference is None and after_element_id != element_id
        ) or not _same_element_resolution(
            before,
            after,
            element_reference=element_reference,
        ):
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed after invocation.",
                details={
                    "outcome": "delivery_only",
                    "method": method,
                },
            )
        verified = _invoke_postcondition_verified(method, before, after)
        return binding, {
            "element": resolved_selector,
            "resolved_element": element_id,
            "pattern": method,
            "before": _public_action_properties(before),
            "after": _public_action_properties(after),
            "postcondition_verified": verified,
            "method": method,
            "outcome": "postcondition_verified" if verified else "delivery_only",
            "resolution_stage": action_payload.get("resolutionStage"),
        }

    def _invoke_native_navigation(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        mutation_guard: Callable[[], None] | None,
        disable_marker: Path | None,
        element_reference: dict[str, Any] | None,
        require_title_match: bool,
    ) -> tuple[NativeNavigationBinding, dict[str, Any]]:
        if element_reference is None:
            raise _semantic_action_unavailable(
                "semantic_provider_action_unsupported",
                "Native UI Automation actions require a fresh exact element reference.",
            )
        _require_complete_element_reference(element_reference)
        if (
            element_reference.get("provider") != "native_uia_raw"
            or element_reference.get("selector") != selector
            or element_reference.get("control_type") not in {"TreeItem", "ListItem"}
        ):
            raise _semantic_action_unavailable(
                "semantic_provider_action_unsupported",
                "The native UI Automation reference has no supported navigation action.",
            )
        before = self._native_navigation_state(
            identity,
            selector=selector,
            element_reference=element_reference,
        )
        before["require_title_match"] = require_title_match
        if before.get("selection_available") is not True:
            raise _semantic_action_unavailable(
                "invoke_pattern_unavailable",
                "The native navigation element no longer exposes SelectionItemPattern.",
            )
        method = "uia.SelectionItemPattern"
        action_payload = self._native_uia_action(
            identity,
            properties=before,
            request={"action": "invoke"},
            method=method,
            mutation_guard=mutation_guard,
            disable_marker=disable_marker,
        )
        self._validate_native_action_payload(
            action_payload,
            properties=before,
            method=method,
        )
        try:
            after = self._native_navigation_state(
                identity,
                selector=selector,
                element_reference=element_reference,
            )
        except ComputerError as exc:
            if exc.code in {"element_not_found", "stale_element", "stale_window"}:
                after = None
            else:
                raise _mark_delivery_error(exc, method) from exc
        if after is not None and not _same_element_resolution(
            before,
            after,
            element_reference=element_reference,
        ):
            raise ComputerError(
                "stale_element",
                "The native navigation element changed after invocation.",
                details={"outcome": "delivery_only", "method": method},
            )
        verified = bool(after is not None and after.get("selected") is True)
        return NativeNavigationBinding(), {
            "element": selector,
            "resolved_element": selector,
            "pattern": method,
            "before": _public_action_properties(before),
            "after": _public_action_properties(after) if after is not None else None,
            "postcondition_verified": verified,
            "method": method,
            "outcome": "postcondition_verified" if verified else "delivery_only",
            "resolution_stage": action_payload.get("resolutionStage"),
        }

    def _native_navigation_state(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        element_reference: dict[str, Any],
    ) -> dict[str, Any]:
        native = self._native_element_state(
            identity,
            properties={"AutomationId": element_reference.get("automation_id")},
            element_reference=element_reference,
        )
        if native is None:
            raise ComputerError(
                "native_uia_unavailable",
                "The trusted Windows UI Automation bridge is unavailable.",
            )
        safe = {
            **native,
            "password": native["is_password"],
            "invoke_available": "uia.InvokePattern" in native["patterns"],
            "toggle_available": "uia.TogglePattern" in native["patterns"],
            "selection_available": "uia.SelectionItemPattern" in native["patterns"],
            "expand_collapse_available": ("uia.ExpandCollapsePattern" in native["patterns"]),
            "element_reference": element_reference,
        }
        _validate_element_reference_properties(selector, safe, element_reference)
        if safe["is_password"]:
            raise ComputerError(
                "password_control",
                "Password controls cannot be mutated by AgentTools.",
            )
        if not safe["enabled"]:
            raise ComputerError("control_disabled", "The selected control is disabled.")
        safe.pop("is_password", None)
        return safe

    def set_value(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        value: str,
        mutation_guard: Callable[[], None] | None = None,
        disable_marker: Path | None = None,
        element_reference: dict[str, Any] | None = None,
        require_title_match: bool = False,
        require_keyboard_focus: bool = False,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        _reject_read_only_provider_locator(selector)
        resolved_selector = _validate_selector(selector)
        binding, element_id, before = self._action_properties(
            identity,
            resolved_selector,
            element_reference=element_reference,
            action_preflight=True,
        )
        before["require_title_match"] = require_title_match
        method = _set_value_pattern(before)
        if method is None:
            raise _semantic_action_unavailable(
                "set_value_pattern_unavailable",
                "No supported semantic value pattern was verified for the element.",
            )
        action_request: dict[str, Any] = {"action": "set-value", "value": value}
        if require_keyboard_focus:
            action_request["requireKeyboardFocus"] = True
        if method == "uia.RangeValuePattern":
            number = _parse_invariant_range_value(value)
            if number is None:
                raise ComputerError(
                    "invalid_range_value",
                    "The range value must be a finite invariant number.",
                )
            action_request = {
                "action": "set-value",
                "rangeBits": _range_value_to_bits(number),
            }
            if require_keyboard_focus:
                action_request["requireKeyboardFocus"] = True
        expected_value_fingerprint = (
            None if method == "uia.RangeValuePattern" else _value_fingerprint(value)
        )
        action_payload = self._native_uia_action(
            identity,
            properties=before,
            request=action_request,
            method=method,
            mutation_guard=mutation_guard,
            disable_marker=disable_marker,
        )
        self._validate_native_action_payload(
            action_payload,
            properties=before,
            method=method,
            require_keyboard_focus=require_keyboard_focus,
        )
        try:
            (
                after_element_id,
                after_runtime_id,
                readback_char_count,
                verified,
                verification_resolution_stage,
                keyboard_focus_after,
            ) = self._read_value_for_verification(
                identity,
                resolved_selector,
                method=method,
                requested_value=value,
                expected_value_fingerprint=expected_value_fingerprint,
                element_reference=element_reference,
            )
        except ComputerError as exc:
            raise _mark_delivery_error(exc, method) from exc
        if (element_reference is None and after_element_id != element_id) or (
            after_runtime_id != before.get("runtime_id") and element_reference is None
        ):
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed after setting its value.",
                details={"outcome": "delivery_only", "method": method},
            )
        if not verified:
            raise ComputerError(
                "set_value_postcondition_failed",
                "The control value readback did not match the requested value.",
                details={"outcome": "failed", "method": method},
            )
        if require_keyboard_focus and keyboard_focus_after is not True:
            raise ComputerError(
                "keyboard_focus_postcondition_failed",
                "The value matched, but keyboard focus on the exact control was not verified.",
                details={
                    "outcome": "delivery_only",
                    "method": method,
                    "postcondition_verified": False,
                    "value_postcondition_verified": True,
                    "keyboard_focus_verified": False,
                },
            )
        return binding, {
            "element": resolved_selector,
            "resolved_element": after_element_id,
            "value_char_count": len(value),
            "readback_char_count": readback_char_count,
            "postcondition_verified": True,
            "method": method,
            "outcome": "postcondition_verified",
            "resolution_stage": action_payload.get("resolutionStage"),
            "verification_resolution_stage": verification_resolution_stage,
            "keyboard_focus_verified": (keyboard_focus_after if require_keyboard_focus else None),
            "keyboard_focus_status": ("verified" if require_keyboard_focus else "not_requested"),
            "submission_ready": require_keyboard_focus and keyboard_focus_after is True,
            "change_required": False,
        }

    def scroll(
        self,
        identity: WindowIdentity,
        *,
        selector: str,
        direction: str | None,
        to: str | None,
        percent: float | None,
        mutation_guard: Callable[[], None] | None = None,
        disable_marker: Path | None = None,
        element_reference: dict[str, Any] | None = None,
        require_title_match: bool = False,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        resolved_selector = _validate_selector(selector)
        choices = sum(value is not None for value in (direction, to, percent))
        if choices != 1:
            raise ComputerError(
                "invalid_scroll_request",
                "Choose exactly one scroll direction, destination, or percent.",
                exit_code=2,
            )
        if direction is not None and direction not in {"up", "down", "left", "right"}:
            raise ComputerError("invalid_scroll_direction", "The scroll direction is invalid.")
        if to is not None and to not in {"top", "bottom"}:
            raise ComputerError("invalid_scroll_destination", "The scroll destination is invalid.")
        if percent is not None and (not math.isfinite(percent) or percent < 0 or percent > 100):
            raise ComputerError("invalid_scroll_percent", "Scroll percent must be from 0 to 100.")
        _reject_read_only_provider_locator(resolved_selector)

        binding, element_id, before, action_properties = self._scroll_snapshot(
            identity,
            resolved_selector,
            element_reference=element_reference,
            action_preflight=True,
        )
        action_properties["require_title_match"] = require_title_match
        if action_properties.get("scroll_available") is not True:
            raise _semantic_action_unavailable(
                "scroll_pattern_unavailable",
                "The element has no verified UI Automation ScrollPattern.",
            )
        if direction is not None:
            axis = "vertical" if direction in {"up", "down"} else "horizontal"
            _require_scrollable_axis(before, axis)
            if _already_at_direction_limit(before[axis], direction):
                return binding, _scroll_result(
                    resolved_selector,
                    element_id,
                    before,
                    before,
                    request={"direction": direction},
                    changed=False,
                    steps=0,
                    outcome="postcondition_verified",
                )
            payload = self._native_uia_action(
                identity,
                properties=action_properties,
                request={
                    "action": "scroll",
                    "mode": "direction",
                    "direction": direction,
                },
                method="uia.ScrollPattern",
                mutation_guard=mutation_guard,
                disable_marker=disable_marker,
            )
            self._validate_native_action_payload(
                payload,
                properties=action_properties,
                method="uia.ScrollPattern",
            )
            after = self._verified_scroll_snapshot(
                identity,
                resolved_selector,
                element_id,
                expected_runtime_id=str(action_properties["runtime_id"]),
                method="uia.ScrollPattern",
                element_reference=element_reference,
            )
            if not _direction_progressed(before[axis], after[axis], direction):
                if before[axis]["percent"] is None or after[axis]["percent"] is None:
                    outcome = "delivery_only"
                else:
                    raise ComputerError(
                        "scroll_postcondition_failed",
                        "The semantic scroll did not move in the requested direction.",
                        details={"outcome": "failed", "method": "uia.ScrollPattern"},
                    )
            else:
                outcome = "postcondition_verified"
            return binding, _scroll_result(
                resolved_selector,
                element_id,
                before,
                after,
                request={"direction": direction},
                changed=before != after,
                steps=1,
                outcome=outcome,
                resolution_stage=payload.get("resolutionStage"),
            )

        if to is not None:
            axis = "vertical"
            if not before[axis]["scrollable"]:
                axis = "horizontal"
            _require_scrollable_axis(before, axis)
            method = "uia.ScrollPattern.SetScrollPercent"
            target = 0.0 if to == "top" else 100.0
            payload = self._native_uia_action(
                identity,
                properties=action_properties,
                request={
                    "action": "scroll",
                    "mode": "percent",
                    "axis": axis,
                    "percent": target,
                },
                method=method,
                mutation_guard=mutation_guard,
                disable_marker=disable_marker,
            )
            self._validate_native_action_payload(
                payload,
                properties=action_properties,
                method=method,
            )
            after = self._verified_scroll_snapshot(
                identity,
                resolved_selector,
                element_id,
                expected_runtime_id=str(action_properties["runtime_id"]),
                method=method,
                element_reference=element_reference,
            )
            if not _percent_within(after[axis]["percent"], target, tolerance=0.05):
                raise ComputerError(
                    "scroll_postcondition_failed",
                    "The semantic scroll destination was not verified.",
                    details={"outcome": "failed", "method": method},
                )
            return binding, _scroll_result(
                resolved_selector,
                element_id,
                before,
                after,
                request={"to": to},
                changed=before != after,
                steps=1,
                outcome="postcondition_verified",
                method=method,
                resolution_stage=payload.get("resolutionStage"),
            )

        assert percent is not None
        axis = "vertical" if before["vertical"]["scrollable"] else "horizontal"
        _require_scrollable_axis(before, axis)
        if before[axis]["percent"] is None:
            raise _semantic_action_unavailable(
                "scroll_percent_unavailable",
                "The selected scroll area's percent is unavailable.",
            )
        if _percent_within(before[axis]["percent"], percent, tolerance=1.0):
            return binding, _scroll_result(
                resolved_selector,
                element_id,
                before,
                before,
                request={"percent": percent, "axis": axis},
                changed=False,
                steps=0,
                outcome="postcondition_verified",
            )
        method = "uia.ScrollPattern.SetScrollPercent"
        payload = self._native_uia_action(
            identity,
            properties=action_properties,
            request={
                "action": "scroll",
                "mode": "percent",
                "axis": axis,
                "percent": percent,
            },
            method=method,
            mutation_guard=mutation_guard,
            disable_marker=disable_marker,
        )
        self._validate_native_action_payload(
            payload,
            properties=action_properties,
            method=method,
        )
        after = self._verified_scroll_snapshot(
            identity,
            resolved_selector,
            element_id,
            expected_runtime_id=str(action_properties["runtime_id"]),
            method=method,
            element_reference=element_reference,
        )
        if not _percent_within(after[axis]["percent"], percent, tolerance=1.0):
            raise ComputerError(
                "scroll_postcondition_failed",
                "The requested semantic scroll percent was not reached within tolerance.",
                details={
                    "outcome": "failed",
                    "method": method,
                },
            )
        return binding, _scroll_result(
            resolved_selector,
            element_id,
            before,
            after,
            request={"percent": percent, "axis": axis, "tolerance": 1.0},
            changed=before != after,
            steps=1,
            outcome="postcondition_verified",
            method=method,
            resolution_stage=payload.get("resolutionStage"),
        )

    def _action_properties(
        self,
        identity: WindowIdentity,
        selector: str,
        *,
        element_reference: dict[str, Any] | None = None,
        action_preflight: bool = False,
    ) -> tuple[WinAppBinding, str, dict[str, Any]]:
        try:
            binding, element_id, safe = self._element_properties(
                identity,
                selector,
                element_reference=element_reference,
            )
        except ComputerError as exc:
            if action_preflight and exc.code in _ACTION_PREFLIGHT_CAPABILITY_ERRORS:
                raise _action_preflight_unavailable(exc) from exc
            raise
        if safe["password"]:
            raise ComputerError(
                "password_control",
                "Password controls cannot be mutated by AgentTools.",
            )
        if not safe["enabled"]:
            raise ComputerError("control_disabled", "The selected control is disabled.")
        return binding, element_id, safe

    def _element_properties(
        self,
        identity: WindowIdentity,
        selector: str,
        *,
        element_reference: dict[str, Any] | None = None,
    ) -> tuple[WinAppBinding, str, dict[str, Any]]:
        if element_reference is not None:
            _require_complete_element_reference(element_reference)
        binding, payload = self._target_json(
            identity,
            ["ui", "get-property", selector],
            max_output_bytes=MAX_PROPERTY_OUTPUT_BYTES,
        )
        properties = payload.get("properties")
        if not isinstance(properties, dict):
            raise _invalid_response()
        element_id = _exact_element_id(payload.get("elementId"))
        native_kwargs: dict[str, Any] = {"properties": properties}
        if element_reference is not None:
            native_kwargs["element_reference"] = element_reference
        native = self._native_element_state(identity, **native_kwargs)
        if native is None:
            raise ComputerError(
                "native_uia_unavailable",
                "The trusted Windows UI Automation bridge is unavailable.",
            )
        safe = _safe_action_properties(properties)
        patterns = set(native["patterns"])
        safe["password"] = bool(
            native["is_password"]
            or (
                "uia.LegacyIAccessible" in patterns
                and native.get("legacy_value_protected") is not False
            )
        )
        safe["enabled"] = native["enabled"]
        safe["runtime_id"] = native["runtime_id"]
        safe["control_type"] = native.get("control_type") or safe["control_type"]
        safe["class"] = native.get("class") or safe["class"]
        safe["ancestor_signature"] = native.get("ancestor_signature")
        safe["resolution_stage"] = native.get("resolution_stage")
        safe["invoke_available"] = "uia.InvokePattern" in patterns
        safe["toggle_available"] = "uia.TogglePattern" in patterns
        safe["selection_available"] = "uia.SelectionItemPattern" in patterns
        safe["expand_collapse_available"] = "uia.ExpandCollapsePattern" in patterns
        safe["value_available"] = "uia.ValuePattern" in patterns
        safe["range_value_available"] = "uia.RangeValuePattern" in patterns
        safe["value_pattern_status"] = native["value_pattern_status"]
        safe["range_value_pattern_status"] = native["range_value_pattern_status"]
        safe["legacy_value_available"] = "uia.LegacyIAccessible" in patterns
        safe["legacy_value_read_only"] = native["legacy_value_read_only"]
        safe["scroll_available"] = "uia.ScrollPattern" in patterns
        safe["range_value_bits"] = native.get("range_value_bits")
        safe["has_keyboard_focus"] = native.get("has_keyboard_focus")
        if element_reference is not None:
            _validate_element_reference_properties(
                selector,
                safe,
                element_reference,
            )
            safe["element_reference"] = element_reference
        return binding, element_id, safe

    def _native_element_state(
        self,
        identity: WindowIdentity,
        *,
        properties: dict[str, Any],
        element_reference: dict[str, Any] | None = None,
        provider_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        automation_id = properties.get("AutomationId")
        if not isinstance(automation_id, str) or not automation_id or len(automation_id) > 160:
            raise ComputerError(
                "uia_identity_unavailable",
                "The element's exact UI Automation identity could not be verified.",
            )
        try:
            request: dict[str, Any] = {
                "action": "inspect",
                "automationId": automation_id,
            }
            if element_reference is not None:
                request["elementRef"] = element_reference
                request["runtimeId"] = element_reference.get("runtime_id")
            if provider_identity is not None:
                request["providerIdentity"] = provider_identity
            payload = self._native_uia_command(
                identity,
                request=request,
                mutating=False,
                method="uia.inspect",
            )
        except ComputerError as exc:
            if exc.code == "native_uia_unavailable":
                return None
            raise
        patterns = payload.get("patterns")
        value_pattern_status = payload.get("valuePatternStatus")
        range_value_pattern_status = payload.get("rangeValuePatternStatus")
        control_type = payload.get("controlType")
        class_name = payload.get("className")
        ancestor_signature = payload.get("ancestorSignature")
        resolution_stage = payload.get("resolutionStage")
        selection_state = payload.get("selectionItemIsSelected")
        keyboard_focus = payload.get("hasKeyboardFocus")
        if (
            type(payload.get("count")) is not int
            or payload.get("count") != 1
            or payload.get("automationId") != automation_id
            or type(payload.get("isPassword")) is not bool
            or type(payload.get("enabled")) is not bool
            or (keyboard_focus is not None and type(keyboard_focus) is not bool)
            or not isinstance(payload.get("runtimeId"), str)
            or not payload.get("runtimeId")
            or (
                control_type is not None
                and (not isinstance(control_type, str) or len(control_type) > 64)
            )
            or (
                class_name is not None
                and (not isinstance(class_name, str) or len(class_name) > MAX_CLASS_NAME_CHARS)
            )
            or (
                ancestor_signature is not None
                and (
                    not isinstance(ancestor_signature, str)
                    or not ancestor_signature.startswith("sha256:")
                    or len(ancestor_signature) != 31
                )
            )
            or not isinstance(patterns, list)
            or not all(isinstance(item, str) for item in patterns)
            or not isinstance(value_pattern_status, str)
            or value_pattern_status not in _PATTERN_QUERY_STATUSES
            or not isinstance(range_value_pattern_status, str)
            or range_value_pattern_status not in _PATTERN_QUERY_STATUSES
            or (provider_identity is not None and resolution_stage != "provider_identity_unique")
            or (
                payload.get("legacyValueReadOnly") is not None
                and type(payload.get("legacyValueReadOnly")) is not bool
            )
            or (
                payload.get("legacyValueProtected") is not None
                and type(payload.get("legacyValueProtected")) is not bool
            )
            or (
                "uia.LegacyIAccessible" not in patterns
                and (
                    payload.get("legacyValueReadOnly") is not None
                    or payload.get("legacyValueProtected") is not None
                )
            )
            or ((value_pattern_status == "available") != ("uia.ValuePattern" in patterns))
            or (
                (range_value_pattern_status == "available") != ("uia.RangeValuePattern" in patterns)
            )
            or (
                payload.get("rangeBits") is not None
                and not _valid_range_value_bits(payload.get("rangeBits"))
            )
            or (
                payload.get("rangeBits") is not None
                and (
                    payload.get("isPassword") is not False
                    or (
                        "uia.LegacyIAccessible" in patterns
                        and payload.get("legacyValueProtected") is not False
                    )
                    or "uia.RangeValuePattern" not in patterns
                )
            )
            or (("uia.SelectionItemPattern" in patterns) != (type(selection_state) is bool))
        ):
            raise _invalid_response()
        return {
            "automation_id": automation_id,
            "runtime_id": payload["runtimeId"],
            "is_password": payload["isPassword"],
            "enabled": payload["enabled"],
            "patterns": patterns,
            "range_value_bits": payload.get("rangeBits"),
            "value_pattern_status": value_pattern_status,
            "range_value_pattern_status": range_value_pattern_status,
            "legacy_value_read_only": payload.get("legacyValueReadOnly"),
            "legacy_value_protected": payload.get("legacyValueProtected"),
            "control_type": payload.get("controlType"),
            "class": payload.get("className"),
            "ancestor_signature": payload.get("ancestorSignature"),
            "resolution_stage": payload.get("resolutionStage"),
            "selected": selection_state,
            "has_keyboard_focus": keyboard_focus,
        }

    def _native_uia_action(
        self,
        identity: WindowIdentity,
        *,
        properties: dict[str, Any],
        request: dict[str, Any],
        method: str,
        mutation_guard: Callable[[], None] | None = None,
        disable_marker: Path | None = None,
    ) -> dict[str, Any]:
        if request.get("action") == "set-value" and method == "uia.LegacyIAccessible":
            error = _semantic_action_unavailable(
                "set_value_pattern_unavailable",
                "Legacy accessibility value mutation is unavailable because it cannot "
                "provide the required final target guarantees.",
            )
            error.details.update({"outcome": "rejected", "method": method})
            raise error
        element_reference = properties.get("element_reference")
        if element_reference is not None:
            _require_complete_element_reference(element_reference)
        automation_id = properties.get("automation_id")
        runtime_id = properties.get("runtime_id")
        if not isinstance(automation_id, str) or not isinstance(runtime_id, str):
            raise ComputerError(
                "uia_identity_unavailable",
                "The element's exact UI Automation identity is unavailable.",
            )
        return self._native_uia_command(
            identity,
            request={
                **request,
                "automationId": automation_id,
                "runtimeId": runtime_id,
                "method": method,
                "elementRef": element_reference,
                "disableMarker": str(disable_marker) if disable_marker is not None else None,
                "window": {
                    "hwnd": identity.hwnd,
                    "pid": identity.pid,
                    "threadId": identity.thread_id,
                    "title": identity.title,
                    "className": identity.class_name,
                    "process": identity.process,
                    "processStarted": identity.process_started,
                    "sessionId": identity.session_id,
                    "desktop": identity.desktop_name,
                    "requireTitleMatch": bool(properties.get("require_title_match")),
                },
            },
            mutating=True,
            method=method,
            mutation_guard=mutation_guard,
        )

    def _validate_native_action_payload(
        self,
        payload: dict[str, Any],
        *,
        properties: dict[str, Any],
        method: str,
        require_keyboard_focus: bool = False,
    ) -> None:
        resolution_stage = payload.get("resolutionStage")
        if (
            type(payload.get("count")) is not int
            or payload.get("count") != 1
            or payload.get("automationId") != properties.get("automation_id")
            or not isinstance(payload.get("runtimeId"), str)
            or not payload.get("runtimeId")
            or payload.get("method") != method
            or payload.get("delivered") is not True
            or (require_keyboard_focus and payload.get("keyboardFocusVerified") is not True)
            or (
                resolution_stage is not None
                and resolution_stage
                not in {
                    "automation_id_unique",
                    "runtime_id_exact",
                    "stable_metadata_unique",
                    "stable_geometry_unique",
                }
            )
            or (
                properties.get("element_reference") is not None
                and resolution_stage
                not in {
                    "runtime_id_exact",
                    "stable_metadata_unique",
                    "stable_geometry_unique",
                }
            )
            or (
                resolution_stage not in {"stable_metadata_unique", "stable_geometry_unique"}
                and payload.get("runtimeId") != properties.get("runtime_id")
            )
        ):
            raise ComputerError(
                "uia_delivery_unverified",
                "The exact UI Automation action target could not be verified.",
                details={"outcome": "delivery_only", "method": method},
            )

    def _native_method_value_fingerprint(
        self,
        identity: WindowIdentity,
        *,
        properties: dict[str, Any],
        method: str,
    ) -> tuple[int, int, str, str, str]:
        element_reference = properties.get("element_reference")
        if element_reference is not None:
            _require_complete_element_reference(element_reference)
        automation_id = properties.get("automation_id")
        runtime_id = properties.get("runtime_id")
        if not isinstance(automation_id, str) or not isinstance(runtime_id, str):
            raise ComputerError(
                "uia_identity_unavailable",
                "The element's exact UI Automation identity is unavailable.",
            )
        payload = self._native_uia_command(
            identity,
            request={
                "action": "read-value",
                "automationId": automation_id,
                "runtimeId": runtime_id,
                "method": method,
                "elementRef": element_reference,
            },
            mutating=False,
            method=method,
        )
        resolution_stage = payload.get("resolutionStage")
        referenced = element_reference is not None
        if (
            type(payload.get("count")) is not int
            or payload.get("count") != 1
            or payload.get("automationId") != automation_id
            or not isinstance(payload.get("runtimeId"), str)
            or not payload.get("runtimeId")
            or payload.get("method") != method
            or payload.get("isPassword") is not False
            or resolution_stage
            not in {
                "automation_id_unique",
                "runtime_id_exact",
                "stable_metadata_unique",
                "stable_geometry_unique",
            }
            or (
                referenced
                and resolution_stage
                not in {
                    "runtime_id_exact",
                    "stable_metadata_unique",
                    "stable_geometry_unique",
                }
            )
            or (
                resolution_stage not in {"stable_metadata_unique", "stable_geometry_unique"}
                and payload.get("runtimeId") != runtime_id
            )
            or type(payload.get("valueUtf8Bytes")) is not int
            or payload.get("valueUtf8Bytes", -1) < 0
            or payload.get("valueUtf8Bytes", _MAX_VALUE_UTF8_BYTES + 1) > _MAX_VALUE_UTF8_BYTES
            or type(payload.get("valueUtf16Chars")) is not int
            or payload.get("valueUtf16Chars", -1) < 0
            or payload.get("valueUtf16Chars", _MAX_VALUE_UTF16_CHARS + 1) > _MAX_VALUE_UTF16_CHARS
            or not isinstance(payload.get("valueSha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", payload["valueSha256"]) is None
        ):
            raise _invalid_response()
        return (
            cast(int, payload["valueUtf8Bytes"]),
            cast(int, payload["valueUtf16Chars"]),
            cast(str, payload["valueSha256"]),
            cast(str, payload["runtimeId"]),
            cast(str, resolution_stage),
        )

    def _native_uia_command(
        self,
        identity: WindowIdentity,
        *,
        request: dict[str, Any],
        mutating: bool,
        method: str,
        mutation_guard: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        powershell = _trusted_powershell()
        if powershell is None:
            error = ComputerError(
                "native_uia_unavailable",
                "The trusted Windows UI Automation bridge is unavailable.",
            )
            if mutating:
                raise _action_preflight_unavailable(error) from error
            raise error
        script = _native_uia_script(identity.hwnd)
        encoded_script = _compressed_powershell_command(script)
        if mutating:
            self.backend.assert_action_allowed(identity)
        self._revalidate_target(identity)
        if mutating and mutation_guard is not None:
            mutation_guard()
        effective_request = dict(request)
        if mutating:
            owner_started = Win32Backend().process_started(os.getpid())
            if owner_started is None:
                raise ComputerError(
                    "computer_action_state_unavailable",
                    "The mutation worker owner identity could not be recorded.",
                    details={"outcome": "failed"},
                )
            effective_request["owner"] = {
                "pid": os.getpid(),
                "processStarted": owner_started,
            }
        try:
            result = _run_process(
                powershell,
                [
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-STA",
                    "-Command",
                    encoded_script,
                ],
                timeout_seconds=NATIVE_UIA_TIMEOUT_SECONDS,
                max_output_bytes=64 * 1024,
                input_bytes=json.dumps(effective_request, ensure_ascii=True).encode("ascii"),
            )
        except ComputerError as exc:
            if mutating:
                process_started = exc.details.get("process_started")
                definitely_not_delivered = process_started is False or (
                    process_started is not True
                    and exc.code
                    in {
                        "uia_backend_start_failed",
                        "uia_pipe_unavailable",
                        "uia_input_delivery_failed",
                    }
                )
                if definitely_not_delivered:
                    raise ComputerError(
                        exc.code,
                        exc.message,
                        exit_code=exc.exit_code,
                        details={**exc.details, "outcome": "failed"},
                    ) from exc
                raise _mark_delivery_error(exc, method) from exc
            raise
        payload: dict[str, Any] | None = None
        try:
            decoded = json.loads(result.stdout.decode("utf-8-sig"))
            if isinstance(decoded, dict):
                payload = decoded
        except (UnicodeDecodeError, RecursionError, ValueError):
            pass
        if result.returncode == 3 and payload is not None and payload.get("count") == 0:
            raise ComputerError(
                "stale_element",
                "The UI Automation element no longer exists.",
                details=_resolution_details(payload),
            )
        if result.returncode == 3:
            raise ComputerError(
                "ambiguous_element",
                "The UI Automation element identity was not unique.",
                details=_resolution_details(payload),
            )
        if result.returncode == 4:
            raise ComputerError(
                "stale_element",
                "The UI Automation element changed before the action.",
                details=_resolution_details(payload),
            )
        if (
            result.returncode == 5
            and payload is not None
            and payload.get("error") == "pattern_unavailable"
        ):
            code = (
                "scroll_pattern_unavailable"
                if method.startswith("uia.ScrollPattern")
                else "set_value_pattern_unavailable"
                if method
                in {
                    "uia.ValuePattern",
                    "uia.RangeValuePattern",
                    "uia.LegacyIAccessible",
                }
                else "invoke_pattern_unavailable"
            )
            error = _semantic_action_unavailable(
                code,
                "The semantic action pattern became unavailable before delivery.",
            )
            error.details.update(
                {
                    "outcome": "rejected",
                    "method": method,
                    **_resolution_details(payload),
                }
            )
            raise error
        if result.returncode == 7:
            raise ComputerError(
                "computer_actions_disabled",
                "Computer mutations are disabled by the emergency stop.",
                details={"outcome": "rejected", "method": method},
            )
        if result.returncode == 10:
            details = {"outcome": "rejected"} if mutating else {}
            raise ComputerError(
                "computer_action_worker_busy",
                "Another computer mutation worker is still active.",
                details=details,
            )
        if result.returncode == 11:
            raise ComputerError(
                "computer_action_owner_gone",
                "The mutation worker owner ended before delivery.",
                details={"outcome": "rejected"},
            )
        if (
            result.returncode == 12
            and payload is not None
            and payload.get("error") == "keyboard_focus_unverified"
        ):
            focus_delivered = payload.get("focusDelivered") is True
            raise ComputerError(
                "keyboard_focus_unverified",
                "The exact UI Automation control could not acquire verified keyboard focus.",
                details={
                    "outcome": "delivery_only",
                    "method": method,
                    "value_delivery": "not_attempted",
                    "focus_delivery": "delivered" if focus_delivered else "attempted",
                    "partial_mutation": "element_focus",
                    "fallback": {
                        "kind": "guarded_physical_input",
                        "status": "available",
                        "reason": "requires_fresh_capture_and_explicit_allow_physical",
                    },
                },
            )
        if result.returncode in {8, 9}:
            error_code = payload.get("error") if payload is not None else None
            if error_code not in {
                "stale_window",
                "stale_element",
                "secure_desktop_unavailable",
                "target_integrity_unavailable",
                "target_integrity_denied",
                "target_safety_unavailable",
                "password_control",
                "control_disabled",
                "control_read_only",
                "element_ref_expired",
                "target_not_foreground",
            }:
                error_code = "target_safety_unavailable"
            action_details: dict[str, Any] = (
                {"outcome": "rejected", "method": method} if mutating else {}
            )
            if payload is not None:
                action_details.update(_resolution_details(payload))
                mismatch_fields = payload.get("mismatchFields")
                if (
                    isinstance(mismatch_fields, list)
                    and len(mismatch_fields) <= 8
                    and all(isinstance(item, str) for item in mismatch_fields)
                ):
                    action_details["identity_mismatch_fields"] = mismatch_fields
                if mutating and payload.get("focusDelivered") is True:
                    action_details.update(
                        {
                            "outcome": "delivery_only",
                            "partial_mutation": "element_focus",
                            "focus_delivery": "delivered",
                            "value_delivery": "not_attempted",
                        }
                    )
            raise ComputerError(
                str(error_code),
                "The exact UI Automation target failed its final safety check.",
                details=action_details,
            )
        if result.returncode == 0 and payload is None and not mutating:
            raise _invalid_response()
        if result.returncode != 0 or payload is None:
            details = {"method": method}
            if mutating:
                delivered = payload.get("delivered") is True if payload is not None else True
                details["outcome"] = "delivery_only" if delivered else "rejected"
            raise ComputerError(
                "native_uia_action_failed" if mutating else "native_uia_query_failed",
                "The trusted UI Automation operation failed.",
                details=details,
            )
        try:
            self._revalidate_target(identity)
        except ComputerError as exc:
            if mutating:
                raise _mark_delivery_error(exc, method) from exc
            raise
        return payload

    def _read_value_for_verification(
        self,
        identity: WindowIdentity,
        selector: str,
        *,
        method: str,
        requested_value: str,
        expected_value_fingerprint: tuple[int, int, str] | None,
        element_reference: dict[str, Any] | None = None,
    ) -> tuple[str, str, int, bool, str | None, bool | None]:
        _binding, element_id, final = self._action_properties(
            identity,
            selector,
            element_reference=element_reference,
        )
        if not _value_method_available(final, method):
            raise ComputerError(
                "uia_pattern_changed",
                "The delivered value pattern became unavailable after delivery.",
            )
        runtime_id = str(final.get("runtime_id") or "")
        if not runtime_id:
            raise ComputerError(
                "uia_identity_unavailable",
                "The element's final UI Automation identity is unavailable.",
            )
        if method == "uia.RangeValuePattern":
            range_bits = final.get("range_value_bits")
            observed = _range_value_text(range_bits)
            return (
                element_id,
                runtime_id,
                len(observed),
                _values_match(
                    requested_value,
                    observed,
                    method=method,
                    observed_range_bits=cast(str, range_bits),
                ),
                None,
                _optional_bool(final.get("has_keyboard_focus")),
            )
        (
            observed_utf8_bytes,
            observed_utf16_chars,
            observed_sha256,
            observed_runtime_id,
            resolution_stage,
        ) = self._native_method_value_fingerprint(
            identity,
            properties=final,
            method=method,
        )
        if expected_value_fingerprint is None:
            raise ComputerError(
                "uia_invalid_response",
                "The expected value fingerprint is unavailable.",
            )
        return (
            element_id,
            observed_runtime_id,
            observed_utf16_chars,
            (
                observed_utf8_bytes,
                observed_utf16_chars,
                observed_sha256,
            )
            == expected_value_fingerprint,
            resolution_stage,
            _optional_bool(final.get("has_keyboard_focus")),
        )

    def _scroll_snapshot(
        self,
        identity: WindowIdentity,
        selector: str,
        *,
        element_reference: dict[str, Any] | None = None,
        action_preflight: bool = False,
    ) -> tuple[WinAppBinding, str, dict[str, Any], dict[str, Any]]:
        binding, element_id, properties = self._action_properties(
            identity,
            selector,
            element_reference=element_reference,
            action_preflight=action_preflight,
        )
        return (
            binding,
            element_id,
            {
                "horizontal": _axis_state(
                    bool(properties.get("horizontal_scrollable")),
                    properties.get("horizontal_percent"),
                    properties.get("horizontal_view_size"),
                ),
                "vertical": _axis_state(
                    bool(properties.get("vertical_scrollable")),
                    properties.get("vertical_percent"),
                    properties.get("vertical_view_size"),
                ),
            },
            properties,
        )

    def _verified_scroll_snapshot(
        self,
        identity: WindowIdentity,
        selector: str,
        expected_element_id: str,
        *,
        expected_runtime_id: str,
        method: str,
        element_reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            _binding, element_id, snapshot, properties = self._scroll_snapshot(
                identity,
                selector,
                element_reference=element_reference,
            )
        except ComputerError as exc:
            raise _mark_delivery_error(exc, method) from exc
        if (element_reference is None and element_id != expected_element_id) or (
            properties.get("runtime_id") != expected_runtime_id and element_reference is None
        ):
            raise ComputerError(
                "stale_element",
                "The UI Automation scroll element changed during the action.",
                details={"outcome": "delivery_only", "method": method},
            )
        return snapshot

    def _target_json(
        self,
        identity: WindowIdentity,
        arguments: list[str],
        *,
        max_output_bytes: int = MAX_BACKEND_OUTPUT_BYTES,
        mutating: bool = False,
        action_method: str | None = None,
        mutation_guard: Callable[[], None] | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        deadline: float | None = None,
        screenshot_target: bool = False,
    ) -> tuple[WinAppBinding, dict[str, Any]]:
        binding = self.require() if deadline is None else self.require(deadline=deadline)
        if mutating:
            self.backend.assert_action_allowed(identity)
        self._revalidate_target(
            identity,
            screenshot_target=screenshot_target,
        )
        _require_deadline(deadline)
        if mutating and mutation_guard is not None:
            mutation_guard()
        result: ProcessResult | None = None
        try:
            result = _run_process(
                binding.path,
                [*arguments, "--window", str(identity.hwnd), "--json"],
                timeout_seconds=_deadline_timeout(timeout_seconds, deadline),
                max_output_bytes=max_output_bytes,
                deadline=deadline,
            )
        except ComputerError as primary:
            try:
                self._revalidate_target(
                    identity,
                    screenshot_target=screenshot_target,
                )
            except ComputerError as secondary:
                if primary.code in {
                    "uia_cleanup_failed",
                    "uia_termination_failed",
                }:
                    primary.details.setdefault("secondary_errors", []).append(secondary.code)
                    raise primary from secondary
                raise
            raise
        try:
            self._revalidate_target(
                identity,
                screenshot_target=screenshot_target,
            )
        except ComputerError as exc:
            if mutating and result is not None and result.returncode == 0:
                raise ComputerError(
                    exc.code,
                    exc.message,
                    details={
                        "outcome": "delivery_only",
                        "method": action_method or "winapp.ui.action",
                    },
                ) from exc
            raise
        _require_deadline(deadline)
        if result.returncode != 0:
            raise _map_backend_error(result.stderr)
        try:
            payload = json.loads(result.stdout.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _invalid_response() from exc
        if not isinstance(payload, dict):
            raise _invalid_response()
        _require_deadline(deadline)
        return binding, payload

    def _revalidate_target(
        self,
        expected: WindowIdentity,
        *,
        screenshot_target: bool = False,
    ) -> WindowIdentity:
        if screenshot_target:
            return self.backend.revalidate(expected)
        return self.backend.revalidate_identity(expected)


def inspect_window(
    *,
    hwnd: int,
    depth: int,
    interactive: bool,
    max_elements: int,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    identity = resolved_backend.capture_identity(hwnd)
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)
    binding, inspection = resolved_adapter.inspect(
        identity,
        depth=depth,
        interactive=interactive,
        max_elements=max_elements,
    )
    return {"window": identity.public(), "backend": binding.public(), **inspection}


def read_element(
    *,
    hwnd: int,
    element: str,
    max_chars: int,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    identity = resolved_backend.capture_identity(hwnd)
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)
    binding, reading = resolved_adapter.read(
        identity,
        selector=element,
        max_chars=max_chars,
    )
    actual = _revalidate_wrapper_identity(
        resolved_backend,
        identity,
        require_title_match=require_title_match,
    )
    return {
        "window": actual.public(),
        "backend": binding.public(),
        **reading,
        "strong_identity_verified": True,
        "identity_mismatch_fields": [],
        "title_changed_during_operation": identity.title != actual.title,
        **_title_transaction_hashes(identity.title, actual.title),
        "observation_tier": "uia_accessibility_provider",
    }


def list_scroll_areas(
    *,
    hwnd: int,
    max_elements: int,
    backend: Win32Backend | None = None,
    adapter: WinAppAdapter | None = None,
    element_ref_store: ElementReferenceStore | None = None,
    require_title_match: bool = False,
) -> dict[str, Any]:
    resolved_backend = backend or Win32Backend()
    identity = resolved_backend.capture_identity(hwnd)
    resolved_adapter = adapter or WinAppAdapter(resolved_backend)
    binding, result = resolved_adapter.scroll_areas(
        identity,
        max_elements=max_elements,
    )
    actual = _revalidate_wrapper_identity(
        resolved_backend,
        identity,
        require_title_match=require_title_match,
    )
    result = attach_element_references(
        result,
        identity=actual,
        collection="areas",
        store=element_ref_store,
    )
    return {
        "window": actual.public(),
        "backend": binding.public(),
        **result,
        "strong_identity_verified": True,
        "identity_mismatch_fields": [],
        "title_changed_during_operation": identity.title != actual.title,
        **_title_transaction_hashes(identity.title, actual.title),
        "observation_tier": "uia_accessibility_provider",
    }


def capabilities(*, adapter: WinAppAdapter | None = None) -> dict[str, Any]:
    probe = (adapter or WinAppAdapter()).probe()
    return {
        "read_only": False,
        "actions_exposed": True,
        "commands": [
            "info",
            "windows",
            "focused",
            "screenshot",
            "ocr",
            "inspect",
            "read",
            "scroll-areas",
            "focus",
            "restore",
            "minimize",
            "maximize",
            "resize",
            "invoke",
            "set-value",
            "scroll",
            "notify",
            "capabilities",
        ],
        "uia_winapp": probe,
        "pin": {
            "version": WINAPP_VERSION,
            "asset": WINAPP_RELEASE_ASSET,
            "release_url": WINAPP_RELEASE_URL,
            "release_sha256": WINAPP_RELEASE_SHA256,
            "runtime_auto_download": sys.platform == "win32",
            "managed_cache": _managed_cache_public_path(),
        },
    }


def _safe_action_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "automation_id": _safe_optional_text(properties.get("AutomationId"), 160),
        "name": _safe_optional_text(properties.get("Name"), MAX_TITLE_CHARS),
        "control_type": _safe_optional_text(properties.get("ControlType"), 64),
        "class": _safe_optional_text(properties.get("ClassName"), MAX_CLASS_NAME_CHARS),
        "enabled": _parse_bool(properties.get("IsEnabled")),
        "offscreen": _parse_bool(properties.get("IsOffscreen")),
        "password": _parse_bool(properties.get("IsPassword")),
        "toggle_state": _safe_optional_text(properties.get("ToggleState"), 32),
        "selected": _first_bool(
            properties.get("SelectionItemIsSelected"), properties.get("IsSelected")
        ),
        "expand_collapse_state": _safe_optional_text(properties.get("ExpandCollapseState"), 32),
        "value_read_only": _first_bool(
            properties.get("ValueIsReadOnly"), properties.get("IsReadOnly")
        ),
        "range_value_read_only": _parse_bool(properties.get("RangeValueIsReadOnly")),
        "invoke_available": _parse_bool(properties.get("IsInvokePatternAvailable")),
        "toggle_available": _parse_bool(properties.get("IsTogglePatternAvailable")),
        "selection_available": _parse_bool(properties.get("IsSelectionItemPatternAvailable")),
        "expand_collapse_available": _parse_bool(
            properties.get("IsExpandCollapsePatternAvailable")
        ),
        "value_available": bool(
            _parse_bool(properties.get("IsValuePatternAvailable")) or "Value" in properties
        ),
        "range_value_available": _parse_bool(properties.get("IsRangeValuePatternAvailable")),
        "legacy_value_available": _parse_bool(
            properties.get("IsLegacyIAccessiblePatternAvailable")
        ),
        "horizontal_scrollable": bool(
            _parse_bool(properties.get("HorizontallyScrollable"))
            or _parse_percent(properties.get("ScrollHorizontalPercent")) is not None
        ),
        "vertical_scrollable": bool(
            _parse_bool(properties.get("VerticallyScrollable"))
            or _parse_percent(properties.get("ScrollVerticalPercent")) is not None
        ),
        "horizontal_percent": _parse_percent(properties.get("ScrollHorizontalPercent")),
        "vertical_percent": _parse_percent(properties.get("ScrollVerticalPercent")),
        "horizontal_view_size": _parse_percent(properties.get("HorizontalViewSize")),
        "vertical_view_size": _parse_percent(properties.get("VerticalViewSize")),
    }


def _read_reference_identity(properties: dict[str, Any]) -> dict[str, Any]:
    """Keep the final native identity private until post-read reacquisition."""
    return {
        "automation_id": properties.get("automation_id"),
        "runtime_id": properties.get("runtime_id"),
        "control_type": properties.get("control_type"),
        "class": properties.get("class"),
        "ancestor_signature": properties.get("ancestor_signature"),
        "enabled": properties.get("enabled"),
        "read_only": _first_bool(
            properties.get("value_read_only"),
            properties.get("range_value_read_only"),
        ),
        "password": properties.get("password"),
    }


def _validate_element_reference_properties(
    selector: str,
    properties: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    _require_complete_element_reference(reference)
    if reference.get("selector") != selector:
        raise ComputerError(
            "stale_element",
            "The element reference provider selector changed.",
            details={"resolution_stage": "provider_selector"},
        )
    comparisons = (
        ("automation_id", properties.get("automation_id"), reference.get("automation_id")),
        ("control_type", properties.get("control_type"), reference.get("control_type")),
        ("class", properties.get("class"), reference.get("class")),
        (
            "ancestor_signature",
            properties.get("ancestor_signature"),
            reference.get("ancestor_signature"),
        ),
    )
    mismatches = [
        name
        for name, actual, expected in comparisons
        if str(actual or "").casefold() != str(expected).casefold()
    ]
    if (
        mismatches == ["ancestor_signature"]
        and properties.get("resolution_stage") == "stable_geometry_unique"
    ):
        mismatches = []
    if mismatches:
        raise ComputerError(
            "stale_element",
            "The element reference resolved to different stable metadata.",
            details={
                "resolution_stage": "provider_metadata",
                "element_mismatch_fields": mismatches,
            },
        )
    if reference.get("password") is False and properties.get("password") is not False:
        raise ComputerError(
            "password_control",
            "The referenced control became password-protected.",
            details={"resolution_stage": "provider_safety_state"},
        )
    if reference.get("enabled") is True and properties.get("enabled") is not True:
        raise ComputerError(
            "control_disabled",
            "The referenced control is no longer enabled.",
            details={"resolution_stage": "provider_safety_state"},
        )
    if reference.get("read_only") is False:
        current_read_only = _first_bool(
            properties.get("value_read_only"),
            properties.get("range_value_read_only"),
        )
        if current_read_only is True:
            raise ComputerError(
                "control_read_only",
                "The referenced control became read-only.",
                details={"resolution_stage": "provider_safety_state"},
            )


def _require_complete_element_reference(reference: object) -> None:
    if not isinstance(reference, dict):
        raise ComputerError(
            "element_ref_invalid",
            "The element reference has invalid stable identity metadata.",
            details={"resolution_stage": "reference_metadata"},
        )
    stable = ("automation_id", "control_type", "class")
    ancestor = reference.get("ancestor_signature")
    if (
        any(
            not isinstance(reference.get(field), str) or not reference.get(field)
            for field in stable
        )
        or not isinstance(ancestor, str)
        or _ANCESTOR_SIGNATURE_RE.fullmatch(ancestor) is None
    ):
        raise ComputerError(
            "element_ref_invalid",
            "The element reference has incomplete stable UI Automation identity.",
            details={"resolution_stage": "reference_metadata"},
        )


def _same_element_resolution(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    element_reference: dict[str, Any] | None,
) -> bool:
    if before.get("runtime_id") == after.get("runtime_id"):
        return True
    if element_reference is None:
        return False
    try:
        _validate_element_reference_properties(
            str(element_reference.get("selector") or ""),
            after,
            element_reference,
        )
    except ComputerError:
        return False
    return True


def _invoke_pattern(properties: dict[str, Any]) -> str | None:
    for key, method in (
        ("invoke_available", "uia.InvokePattern"),
        ("toggle_available", "uia.TogglePattern"),
        ("selection_available", "uia.SelectionItemPattern"),
        ("expand_collapse_available", "uia.ExpandCollapsePattern"),
    ):
        if properties.get(key) is True:
            return method
    return None


def _first_bool(*values: object) -> bool | None:
    for value in values:
        parsed = _parse_bool(value)
        if parsed is not None:
            return parsed
    return None


def _set_value_pattern(properties: dict[str, Any]) -> str | None:
    value_available = properties.get("value_available") is True
    range_available = properties.get("range_value_available") is True
    if value_available and properties.get("value_read_only") is False:
        return "uia.ValuePattern"
    if range_available and properties.get("range_value_read_only") is False:
        return "uia.RangeValuePattern"
    if value_available and properties.get("value_read_only") is None:
        return "uia.ValuePattern"
    if range_available and properties.get("range_value_read_only") is None:
        return "uia.RangeValuePattern"
    if (
        properties.get("value_pattern_status") == "failed"
        or properties.get("range_value_pattern_status") == "failed"
    ):
        raise ComputerError(
            "uia_pattern_query_failed",
            "The element's value patterns could not be determined safely.",
        )
    if value_available or range_available:
        raise ComputerError("control_read_only", "The selected control is read-only.")
    if properties.get("legacy_value_available") is True:
        if properties.get("legacy_value_read_only") is True:
            raise ComputerError("control_read_only", "The selected control is read-only.")
        return None
    return None


def _effective_value_read_method(properties: dict[str, Any]) -> str:
    if properties.get("value_available") is True:
        return "uia.ValuePattern"
    if properties.get("range_value_available") is True:
        return "uia.RangeValuePattern"
    if (
        properties.get("value_pattern_status") == "failed"
        or properties.get("range_value_pattern_status") == "failed"
    ):
        raise ComputerError(
            "uia_pattern_query_failed",
            "The element's value patterns could not be determined safely.",
        )
    if properties.get("legacy_value_available") is True:
        return "uia.LegacyIAccessible"
    return "winapp.get-value"


def _value_method_available(properties: dict[str, Any], method: str) -> bool:
    key = {
        "uia.ValuePattern": "value_available",
        "uia.RangeValuePattern": "range_value_available",
        "uia.LegacyIAccessible": "legacy_value_available",
    }.get(method)
    return key is not None and properties.get(key) is True


def _invoke_postcondition_verified(
    method: str, before: dict[str, Any], after: dict[str, Any]
) -> bool:
    if method == "uia.TogglePattern":
        return (
            before.get("toggle_state") is not None
            and after.get("toggle_state") is not None
            and before.get("toggle_state") != after.get("toggle_state")
        )
    if method == "uia.SelectionItemPattern":
        return after.get("selected") is True
    if method == "uia.ExpandCollapsePattern":
        return (
            before.get("expand_collapse_state") is not None
            and after.get("expand_collapse_state") is not None
            and before.get("expand_collapse_state") != after.get("expand_collapse_state")
        )
    return False


def _values_match(
    requested: str,
    observed: str,
    *,
    method: str,
    observed_range_bits: str | None = None,
) -> bool:
    if method != "uia.RangeValuePattern":
        return requested == observed
    requested_number = _parse_invariant_range_value(requested)
    if requested_number is None:
        return False
    requested_bits = _range_value_to_bits(requested_number)
    if observed_range_bits is not None:
        return _valid_range_value_bits(observed_range_bits) and (
            requested_bits == observed_range_bits
        )
    observed_number = _parse_invariant_range_value(observed)
    return observed_number is not None and (requested_bits == _range_value_to_bits(observed_number))


def _parse_invariant_range_value(value: str) -> float | None:
    if len(value) > _MAX_INVARIANT_DOUBLE_CHARS or _INVARIANT_DOUBLE_RE.fullmatch(value) is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _normalized_value_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise _invalid_response()
    return value


def _value_payload_text(payload: dict[str, Any]) -> str:
    if "text" not in payload:
        raise _invalid_response()
    return _normalized_value_text(payload["text"])


def _value_fingerprint(value: str) -> tuple[int, int, str]:
    if len(value) > _MAX_VALUE_CODEPOINTS:
        raise ComputerError(
            "invalid_value",
            f"The value must contain at most {_MAX_VALUE_CODEPOINTS} characters.",
        )
    try:
        utf8 = value.encode("utf-8")
        utf16 = value.encode("utf-16-le")
    except UnicodeEncodeError as exc:
        raise ComputerError(
            "invalid_value",
            "The value must contain valid Unicode text.",
        ) from exc
    return len(utf8), len(utf16) // 2, hashlib.sha256(utf8).hexdigest()


def _public_action_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in properties.items() if key not in _PRIVATE_ACTION_PROPERTY_KEYS
    }


def _valid_range_value_bits(value: object) -> bool:
    if not isinstance(value, str) or _INVARIANT_DOUBLE_BITS_RE.fullmatch(value) is None:
        return False
    try:
        number = _range_value_from_bits(value)
    except (struct.error, ValueError):
        return False
    return math.isfinite(number)


def _range_value_from_bits(value: str) -> float:
    return cast(float, struct.unpack(">d", bytes.fromhex(value))[0])


def _range_value_to_bits(value: float) -> str:
    return struct.pack(">d", value).hex().upper()


def _canonical_invariant_double(number: float) -> str:
    if number == 0:
        return "0"
    text = repr(number).upper()
    return text[:-2] if text.endswith(".0") else text


def _range_value_text(value: object) -> str:
    if not _valid_range_value_bits(value):
        raise ComputerError(
            "uia_range_value_read_failed",
            "The exact UI Automation range value could not be read.",
        )
    return _canonical_invariant_double(_range_value_from_bits(cast(str, value)))


def _mark_delivery_error(error: ComputerError, method: str) -> ComputerError:
    if "outcome" in error.details:
        return error
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={**error.details, "outcome": "delivery_only", "method": method},
    )


def _require_scrollable_axis(snapshot: dict[str, Any], axis: str) -> None:
    if not bool(snapshot[axis].get("scrollable")):
        raise _semantic_action_unavailable(
            "scroll_axis_unavailable",
            f"The selected element is not {axis}ly scrollable.",
        )


def _already_at_direction_limit(axis: dict[str, Any], direction: str) -> bool:
    if direction in {"up", "left"}:
        return axis.get("moreBefore") is False
    return axis.get("moreAfter") is False


def _direction_progressed(before: dict[str, Any], after: dict[str, Any], direction: str) -> bool:
    before_percent = before.get("percent")
    after_percent = after.get("percent")
    if before_percent is None or after_percent is None:
        return False
    if direction in {"up", "left"}:
        return float(after_percent) < float(before_percent)
    return float(after_percent) > float(before_percent)


def _percent_within(value: object, target: float, *, tolerance: float) -> bool:
    return isinstance(value, (int, float)) and abs(float(value) - target) <= tolerance


def _scroll_result(
    selector: str,
    element_id: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    request: dict[str, Any],
    changed: bool,
    steps: int,
    outcome: str,
    method: str = "uia.ScrollPattern",
    resolution_stage: object = None,
) -> dict[str, Any]:
    return {
        "element": selector,
        "resolved_element": element_id,
        "request": request,
        "before": before,
        "after": after,
        "changed": changed,
        "steps": steps,
        "postcondition_verified": outcome == "postcondition_verified",
        "method": method,
        "outcome": outcome,
        "resolution_stage": resolution_stage,
    }


def redact_credential_like_text(value: str) -> tuple[str, bool]:
    output, redacted = _redact_quoted_credential_values(value)

    def replace_value(match: re.Match[str]) -> str:
        nonlocal redacted
        redacted = True
        return f"{match.group('prefix')}[REDACTED]"

    output = _AUTHORIZATION_SCHEME_RE.sub(replace_value, output)
    output = _UNQUOTED_CREDENTIAL_RE.sub(replace_value, output)
    if _BEARER_RE.search(output):
        redacted = True
        output = _BEARER_RE.sub("Bearer [REDACTED]", output)
    return output, redacted


def _redact_quoted_credential_values(value: str) -> tuple[str, bool]:
    parts: list[str] = []
    cursor = 0
    redacted = False
    while match := _QUOTED_CREDENTIAL_PREFIX_RE.search(value, cursor):
        parts.append(value[cursor : match.start()])
        quote = match.group("quote")
        parts.append(f"{match.group('prefix')}{quote}[REDACTED]")
        redacted = True
        index = match.end()
        while index < len(value):
            character = value[index]
            if character == "\\":
                index += 2 if index + 1 < len(value) else 1
                continue
            if character == quote:
                parts.append(quote)
                cursor = index + 1
                break
            index += 1
        else:
            cursor = len(value)
            break
    parts.append(value[cursor:])
    return "".join(parts), redacted


def _native_uia_script(hwnd: int) -> str:
    script = (
        "$ErrorActionPreference='Stop';\n"
        "Add-Type -AssemblyName UIAutomationClient;\n"
        "function Emit($payload,[int]$code){"
        "$payload|ConvertTo-Json -Compress -Depth 5;exit $code};\n"
        f"$workerMutex=New-Object Threading.Mutex($false,'{WORKER_MUTEX_NAME}');\n"
        "$workerAcquired=$false;\n"
        "try {\n"
        "$workerAcquired=$workerMutex.WaitOne(0);\n"
        "if(-not $workerAcquired){Emit @{error='computer_action_worker_busy'} 10};\n"
        "$requestText=[Console]::In.ReadToEnd();\n"
        "$request=$requestText|ConvertFrom-Json;\n"
        "$action=[string]$request.action;\n"
        "function Assert-OwnerAlive {\n"
        "try {\n"
        "$owner=$request.owner;\n"
        "if(-not $owner){Emit @{error='computer_action_owner_gone'} 11};\n"
        "$ownerProcess=[System.Diagnostics.Process]::GetProcessById([int]$owner.pid);\n"
        "if([int64]$ownerProcess.StartTime.ToFileTime() -ne "
        "[int64]$owner.processStarted){"
        "Emit @{error='computer_action_owner_gone'} 11};\n"
        "} catch {Emit @{error='computer_action_owner_gone'} 11}\n"
        "};\n"
        "if($action -ne 'inspect' -and $action -ne 'read-value'){"
        "Assert-OwnerAlive};\n"
        "Add-Type -TypeDefinition @'\n"
        + _NATIVE_SAFETY_CSHARP
        + "\n'@\n"
        + r"""
$root=[System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]__HWND__);
function Assert-ReferenceFresh {
    if($request.elementRef -and
        [int64]$request.elementRef.expires_unix_ms -le
        [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()){
        Emit @{error='element_ref_expired';resolutionStage='reference_freshness'} 8}
};
function Control-Type-Name($candidate) {
    $name=[string]$candidate.Current.ControlType.ProgrammaticName;
    if($name.StartsWith('ControlType.')){$name=$name.Substring(12)};
    return $name.ToLowerInvariant()
};
function Compatible-Control-Type($candidateType,$expectedType,$expectedClass) {
    if([string]$candidateType -ceq [string]$expectedType){return $true};
    if([string]$candidateType -cne 'pane'){return $false};
    $class=[string]$expectedClass;
    if([string]$expectedType -ceq 'edit' -and
        $class -clike 'windowsforms10.edit.*'){return $true};
    if([string]$expectedType -ceq 'button' -and
        $class -clike 'windowsforms10.button.*'){return $true};
    if([string]$expectedType -ceq 'text' -and
        $class -clike 'windowsforms10.static.*'){return $true};
    return $false
};
function Ancestor-Signature($candidate) {
    $segments=New-Object Collections.Generic.List[string];
    $walker=[System.Windows.Automation.TreeWalker]::RawViewWalker;
    $current=$walker.GetParent($candidate);$count=0;
    while($null -ne $current -and $count -lt 12){
        $segments.Add(([string]$current.Current.AutomationId)+'|'+
            (Control-Type-Name $current)+'|'+
            ([string]$current.Current.ClassName).ToLowerInvariant());
        if([System.Windows.Automation.Automation]::Compare($current,$root)){break};
        $current=$walker.GetParent($current);$count++
    };
    $bytes=[Text.Encoding]::UTF8.GetBytes(($segments -join "`n"));
    $sha=[Security.Cryptography.SHA256]::Create();
    try{$digest=$sha.ComputeHash($bytes)}finally{$sha.Dispose()};
    return 'sha256:'+([BitConverter]::ToString($digest)).Replace('-','').
        ToLowerInvariant().Substring(0,24)
};
function Resolve-TargetElement {
    $condition=New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::AutomationIdProperty,
        [string]$request.automationId);
    $elementMatches=$root.FindAll(
        [System.Windows.Automation.TreeScope]::Descendants,$condition);
    $geometryCandidates=$elementMatches;
    $resolvedElement=$null;$stage='automation_id_unique';
    if($request.elementRef){
        Assert-ReferenceFresh;
        $expectedAutomation=[string]$request.elementRef.automation_id;
        $expectedType=([string]$request.elementRef.control_type).ToLowerInvariant();
        $expectedClassRaw=[string]$request.elementRef.class;
        $expectedClass=$expectedClassRaw.ToLowerInvariant();
        $expectedAncestor=[string]$request.elementRef.ancestor_signature;
        $expectedBounds=$request.elementRef.bounds;
        if(-not $expectedAutomation -or -not $expectedType -or -not $expectedClass -or
            $expectedAncestor -cnotmatch '^sha256:[0-9a-f]{24}$' -or
            $null -eq $expectedBounds -or $null -eq $expectedBounds.x -or
            $null -eq $expectedBounds.y -or $null -eq $expectedBounds.width -or
            $null -eq $expectedBounds.height -or
            $expectedAutomation -cne [string]$request.automationId){
            Emit @{error='stale_element';resolutionStage='reference_metadata'} 4};
        if($geometryCandidates.Count -eq 0){
            $geometryCondition=New-Object System.Windows.Automation.PropertyCondition(
                [System.Windows.Automation.AutomationElement]::ClassNameProperty,
                $expectedClassRaw);
            $geometryCandidates=$root.FindAll(
                [System.Windows.Automation.TreeScope]::Descendants,
                $geometryCondition);
            if($geometryCandidates.Count -gt 4096){
                Emit @{error='target_safety_unavailable';
                    resolutionStage='bounded_geometry_candidates'} 8}
        };
        $runtimeExpected=[string]$request.elementRef.runtime_id;
        if(-not $runtimeExpected){$runtimeExpected=[string]$request.runtimeId};
        if($runtimeExpected){
            $runtimeMatches=@();
            for($matchIndex=0;$matchIndex -lt $elementMatches.Count;$matchIndex++){
                [System.Windows.Automation.AutomationElement]$candidate=
                    $elementMatches.Item($matchIndex);
                if(($candidate.GetRuntimeId() -join ',') -ceq $runtimeExpected){
                    $runtimeMatches+=$candidate}};
            if($runtimeMatches.Count -eq 1){
                $resolvedElement=$runtimeMatches[0];$stage='runtime_id_exact'
            } elseif($runtimeMatches.Count -gt 1){
                Emit @{error='element_not_unique';count=$runtimeMatches.Count;
                    resolutionStage='runtime_id_exact'} 3
            }
        };
        if($null -eq $resolvedElement){
            $stableMatches=@();
            $stableMismatchFields=@();
            for($matchIndex=0;$matchIndex -lt $elementMatches.Count;$matchIndex++){
                [System.Windows.Automation.AutomationElement]$candidate=
                    $elementMatches.Item($matchIndex);
                $candidateType=Control-Type-Name $candidate;
                $candidateClass=([string]$candidate.Current.ClassName).ToLowerInvariant();
                $candidateAncestor=Ancestor-Signature $candidate;
                if($candidateType -cne $expectedType){
                    $stableMismatchFields+=@('control_type')};
                if($candidateClass -cne $expectedClass){
                    $stableMismatchFields+=@('class')};
                if($candidateAncestor -cne $expectedAncestor){
                    $stableMismatchFields+=@('ancestor_signature')};
                if($candidateType -ceq $expectedType -and
                    $candidateClass -ceq $expectedClass -and
                    $candidateAncestor -ceq $expectedAncestor){
                    $stableMatches+=$candidate}};
            if($stableMatches.Count -eq 0){
                $geometryMatches=@();
                for($matchIndex=0;$matchIndex -lt $geometryCandidates.Count;$matchIndex++){
                    [System.Windows.Automation.AutomationElement]$candidate=
                        $geometryCandidates.Item($matchIndex);
                    $candidateRect=$candidate.Current.BoundingRectangle;
                    $candidateType=Control-Type-Name $candidate;
                    $candidateTypeCompatible=Compatible-Control-Type `
                        $candidateType $expectedType $expectedClass;
                    if($candidateTypeCompatible -and
                        ([string]$candidate.Current.ClassName).ToLowerInvariant() -ceq
                            $expectedClass -and
                        [int][Math]::Round($candidateRect.X) -eq
                            [int]$expectedBounds.x -and
                        [int][Math]::Round($candidateRect.Y) -eq
                            [int]$expectedBounds.y -and
                        [int][Math]::Round($candidateRect.Width) -eq
                            [int]$expectedBounds.width -and
                        [int][Math]::Round($candidateRect.Height) -eq
                            [int]$expectedBounds.height){
                        $geometryMatches+=$candidate}};
                if($geometryMatches.Count -eq 0){
                    Emit @{error='stale_element';count=0;
                        resolutionStage='stable_geometry_unique';
                        mismatchFields=@($stableMismatchFields|Select-Object -Unique)} 4};
                if($geometryMatches.Count -gt 1){
                    Emit @{error='element_not_unique';count=$geometryMatches.Count;
                        resolutionStage='stable_geometry_unique'} 3};
                $resolvedElement=$geometryMatches[0];$stage='stable_geometry_unique'
            };
            if($stableMatches.Count -gt 1){
                Emit @{error='element_not_unique';count=$stableMatches.Count;
                    resolutionStage='stable_metadata_unique'} 3};
            if($stableMatches.Count -eq 1){
                $resolvedElement=$stableMatches[0];$stage='stable_metadata_unique'}
        };
        $mismatches=New-Object Collections.Generic.List[string];
        if($stage -cne 'stable_geometry_unique' -and
            [string]$resolvedElement.Current.AutomationId -cne $expectedAutomation){
            $mismatches.Add('automation_id')};
        $resolvedType=Control-Type-Name $resolvedElement;
        $resolvedTypeCompatible=Compatible-Control-Type `
            $resolvedType $expectedType $expectedClass;
        if(-not $resolvedTypeCompatible){
            $mismatches.Add('control_type')};
        if(([string]$resolvedElement.Current.ClassName).ToLowerInvariant() -cne
            $expectedClass){$mismatches.Add('class')};
        if($stage -cne 'stable_geometry_unique' -and
            (Ancestor-Signature $resolvedElement) -cne $expectedAncestor){
            $mismatches.Add('ancestor_signature')};
        if($stage -ceq 'stable_geometry_unique'){
            $resolvedRect=$resolvedElement.Current.BoundingRectangle;
            if([int][Math]::Round($resolvedRect.X) -ne [int]$expectedBounds.x -or
                [int][Math]::Round($resolvedRect.Y) -ne [int]$expectedBounds.y -or
                [int][Math]::Round($resolvedRect.Width) -ne
                    [int]$expectedBounds.width -or
                [int][Math]::Round($resolvedRect.Height) -ne
                    [int]$expectedBounds.height){$mismatches.Add('bounds')}};
        if($mismatches.Count -ne 0){
            Emit @{error='stale_element';resolutionStage='reference_metadata';
                mismatchFields=@($mismatches)} 4}
    } elseif($request.providerIdentity) {
        $expectedType=([string]$request.providerIdentity.controlType).ToLowerInvariant();
        $expectedClassRaw=[string]$request.providerIdentity.className;
        $expectedClass=$expectedClassRaw.ToLowerInvariant();
        $expectedBounds=$request.providerIdentity.bounds;
        if(-not $expectedType -or $null -eq $expectedBounds -or
            $null -eq $expectedBounds.x -or $null -eq $expectedBounds.y -or
            $null -eq $expectedBounds.width -or $null -eq $expectedBounds.height){
            Emit @{error='stale_element';count=0;
                resolutionStage='provider_identity_unique';
                mismatchFields=@('provider_identity')} 4};
        if($geometryCandidates.Count -eq 0){
            $geometryCondition=if($expectedClassRaw){
                New-Object System.Windows.Automation.PropertyCondition(
                    [System.Windows.Automation.AutomationElement]::ClassNameProperty,
                    $expectedClassRaw)
            }else{[System.Windows.Automation.Condition]::TrueCondition};
            $geometryCandidates=$root.FindAll(
                [System.Windows.Automation.TreeScope]::Descendants,
                $geometryCondition);
            if($geometryCandidates.Count -gt 4096){
                Emit @{error='target_safety_unavailable';
                    resolutionStage='bounded_geometry_candidates'} 8}
        };
        $providerMatches=@();$providerMismatchFields=@();
        for($matchIndex=0;$matchIndex -lt $geometryCandidates.Count;$matchIndex++){
            [System.Windows.Automation.AutomationElement]$candidate=
                $geometryCandidates.Item($matchIndex);
            $candidateType=Control-Type-Name $candidate;
            $candidateClass=([string]$candidate.Current.ClassName).ToLowerInvariant();
            $candidateRect=$candidate.Current.BoundingRectangle;
            $candidateX=[int][Math]::Round($candidateRect.X);
            $candidateY=[int][Math]::Round($candidateRect.Y);
            $candidateWidth=[int][Math]::Round($candidateRect.Width);
            $candidateHeight=[int][Math]::Round($candidateRect.Height);
            $candidateTypeCompatible=Compatible-Control-Type `
                $candidateType $expectedType $expectedClass;
            if(-not $candidateTypeCompatible){
                $providerMismatchFields+=@('control_type')};
            if($expectedClass -and $candidateClass -cne $expectedClass){
                $providerMismatchFields+=@('class')};
            if($candidateX -ne [int]$expectedBounds.x -or
                $candidateY -ne [int]$expectedBounds.y -or
                $candidateWidth -ne [int]$expectedBounds.width -or
                $candidateHeight -ne [int]$expectedBounds.height){
                $providerMismatchFields+=@('bounds')};
            if($candidateTypeCompatible -and
                (-not $expectedClass -or $candidateClass -ceq $expectedClass) -and
                $candidateX -eq [int]$expectedBounds.x -and
                $candidateY -eq [int]$expectedBounds.y -and
                $candidateWidth -eq [int]$expectedBounds.width -and
                $candidateHeight -eq [int]$expectedBounds.height){
                $providerMatches+=$candidate}};
        if($providerMatches.Count -eq 0){
            Emit @{error='stale_element';count=0;
                resolutionStage='provider_identity_unique';
                mismatchFields=@($providerMismatchFields|Select-Object -Unique)} 4};
        if($providerMatches.Count -gt 1){
            Emit @{error='element_not_unique';count=$providerMatches.Count;
                resolutionStage='provider_identity_unique'} 3};
        $resolvedElement=$providerMatches[0];$stage='provider_identity_unique'
    } else {
        if($elementMatches.Count -ne 1){
            Emit @{error='element_not_unique';count=$elementMatches.Count;
                resolutionStage='automation_id_unique'} 3};
        $resolvedElement=$elementMatches.Item(0)
    };
    return @{element=$resolvedElement;resolutionStage=$stage}
};
$resolvedTarget=Resolve-TargetElement;
$element=$resolvedTarget.element;
$resolutionStage=[string]$resolvedTarget.resolutionStage;
$runtimeId=($element.GetRuntimeId() -join ',');
$nativeAutomationId=[string]$element.Current.AutomationId;
if(-not $request.elementRef -and $request.runtimeId -and
    $runtimeId -cne [string]$request.runtimeId){
    Emit @{error='stale_element';runtimeId=$runtimeId} 4};
function Is-PatternUnavailable($errorRecord) {
    $current=$errorRecord.Exception;
    while($null -ne $current){
        if($current -is [System.InvalidOperationException]){return $true};
        if($current -is [System.Runtime.InteropServices.COMException] -and
            $current.HResult -eq -2147220988){return $true};
        $current=$current.InnerException
    };
    return $false
};
function Legacy-Safety-Code($errorRecord) {
    $current=$errorRecord.Exception;
    while($null -ne $current){
        if($current -is [AgentToolsLegacySafetyException]){
            return [string]$current.Code};
        $current=$current.InnerException
    };
    return $null
};
function Emit-Legacy-Safety($errorRecord) {
    $legacySafetyCode=Legacy-Safety-Code $errorRecord;
    if(-not $legacySafetyCode){return};
    $legacyExit=if($legacySafetyCode -eq 'ambiguous_element'){3}
        elseif($legacySafetyCode -eq 'stale_element'){4}
        elseif($legacySafetyCode -eq 'computer_actions_disabled'){7}
        elseif($legacySafetyCode -eq 'computer_action_owner_gone'){11}
        else{8};
    Emit @{error=$legacySafetyCode;runtimeId=$runtimeId;method=$method;
        delivered=$false;resolutionStage=$resolutionStage} $legacyExit
};
function Is-ElementUnavailable($errorRecord) {
    $current=$errorRecord.Exception;
    while($null -ne $current){
        if($current -is [System.Windows.Automation.ElementNotAvailableException]){
            return $true};
        if($current -is [System.Runtime.InteropServices.COMException] -and
            $current.HResult -eq -2147220991){return $true};
        $current=$current.InnerException
    };
    return $false
};
function Assert-LegacyUnprotected {
    try{$legacySafetyState=[int64][AgentToolsLegacyUia]::State(
        [IntPtr]__HWND__,$nativeAutomationId,$runtimeId)
    }catch{
        Emit-Legacy-Safety $_;
        if(Is-ElementUnavailable $_){
            Emit @{error='stale_element';resolutionStage=$resolutionStage} 4};
        Emit @{error='password_control'} 9
    };
    if($legacySafetyState -eq -1){return};
    if($legacySafetyState -lt 0 -or $legacySafetyState -gt [uint32]::MaxValue){
        Emit @{error='password_control'} 9};
    if(($legacySafetyState -band 0x20000000) -ne 0){
        Emit @{error='password_control'} 9}
};
if($action -eq 'inspect'){
    $patterns=@();$rangeBits=$null;$selectionItemIsSelected=$null;
    $legacyValueReadOnly=$null;$legacyValueProtected=$null;
    $valuePatternStatus='failed';$rangeValuePatternStatus='failed';
    $runtimeBefore=$runtimeId;$passwordBefore=$element.Current.IsPassword;
    try{[void]$element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern);
        $patterns+='uia.InvokePattern'}catch{};
    try{[void]$element.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern);
        $patterns+='uia.TogglePattern'}catch{};
    try{$selectionPattern=$element.GetCurrentPattern(
        [System.Windows.Automation.SelectionItemPattern]::Pattern);
        $patterns+='uia.SelectionItemPattern';
        $selectionItemIsSelected=$selectionPattern.Current.IsSelected}catch{};
    try{[void]$element.GetCurrentPattern(
        [System.Windows.Automation.ExpandCollapsePattern]::Pattern);
        $patterns+='uia.ExpandCollapsePattern'}catch{};
    try{[void]$element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern);
        $patterns+='uia.ValuePattern';$valuePatternStatus='available'
    }catch{
        if(Is-PatternUnavailable $_){
            $valuePatternStatus='absent'
        }else{$valuePatternStatus='failed'}};
    try{[void]$element.GetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern);
        $patterns+='uia.RangeValuePattern';$rangeValuePatternStatus='available';
        if($passwordBefore -eq $false){try{
            $rangeValue=$element.GetCurrentPropertyValue(
                [System.Windows.Automation.RangeValuePattern]::ValueProperty,$false);
            $number=[double]$rangeValue;
            if(-not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)){
                $bytes=[BitConverter]::GetBytes($number);
                if([BitConverter]::IsLittleEndian){[Array]::Reverse($bytes)};
                $rangeBits=([BitConverter]::ToString($bytes)).Replace('-','')
            }
        }catch{$rangeBits=$null}}
    }catch{
        if(Is-PatternUnavailable $_){
            $rangeValuePatternStatus='absent'
        }else{$rangeValuePatternStatus='failed'}};
    try{$legacyState=[int64][AgentToolsLegacyUia]::State(
        [IntPtr]__HWND__,$nativeAutomationId,$runtimeId);
        if($legacyState -lt -1 -or $legacyState -gt [uint32]::MaxValue){
            Emit @{error='password_control'} 9};
        if($legacyState -ne -1){
            $patterns+='uia.LegacyIAccessible';
            $legacyValueReadOnly=(($legacyState -band 0x40) -ne 0);
            $legacyValueProtected=(($legacyState -band 0x20000000) -ne 0)
        }
    }catch{
        Emit-Legacy-Safety $_;
        if(Is-ElementUnavailable $_){
            Emit @{error='stale_element';resolutionStage=$resolutionStage} 4};
        Emit @{error='password_control'} 9
    };
    try{[void]$element.GetCurrentPattern([System.Windows.Automation.ScrollPattern]::Pattern);
        $patterns+='uia.ScrollPattern'}catch{};
    $runtimeAfter=($element.GetRuntimeId() -join ',');
    $passwordAfter=$element.Current.IsPassword;
    if($runtimeBefore -cne $runtimeAfter){
        Emit @{error='stale_element';runtimeId=$runtimeAfter} 4};
    if($passwordBefore -ne $false -or $passwordAfter -ne $false -or
        ($patterns -contains 'uia.LegacyIAccessible' -and
            $legacyValueProtected -ne $false)){$rangeBits=$null};
    $nativeControlType=Control-Type-Name $element;
    $reportedControlType=if($resolutionStage -ceq 'stable_geometry_unique'){
        [string]$request.elementRef.control_type}else{$nativeControlType};
    Emit @{count=1;automationId=[string]$request.automationId;
        nativeAutomationId=[string]$element.Current.AutomationId;runtimeId=$runtimeAfter;
        resolutionStage=$resolutionStage;
        isPassword=$passwordAfter;enabled=$element.Current.IsEnabled;
        hasKeyboardFocus=$element.Current.HasKeyboardFocus;
        controlType=$reportedControlType;nativeControlType=$nativeControlType;
        className=[string]$element.Current.ClassName;
        ancestorSignature=(Ancestor-Signature $element);
        patterns=$patterns;rangeBits=$rangeBits;
        selectionItemIsSelected=$selectionItemIsSelected;
        valuePatternStatus=$valuePatternStatus;
        rangeValuePatternStatus=$rangeValuePatternStatus;
        legacyValueReadOnly=$legacyValueReadOnly;
        legacyValueProtected=$legacyValueProtected} 0};
$method=[string]$request.method;
if($action -eq 'read-value'){
    $readTarget=Resolve-TargetElement;
    $element=$readTarget.element;
    $resolutionStage=[string]$readTarget.resolutionStage;
    $runtimeBefore=($element.GetRuntimeId() -join ',');
    $runtimeId=$runtimeBefore;
    if(-not $request.elementRef -and $request.runtimeId -and
        $runtimeId -cne [string]$request.runtimeId){
        Emit @{error='stale_element';runtimeId=$runtimeId;
            resolutionStage=$resolutionStage} 4};
    $passwordBefore=$element.Current.IsPassword;
    if($passwordBefore -ne $false){Emit @{error='password_control'} 9};
    Assert-LegacyUnprotected;
    try{
        if($method -eq 'uia.ValuePattern'){
            $readPattern=$element.GetCurrentPattern(
                [System.Windows.Automation.ValuePattern]::Pattern);
            $readValue=[string]$readPattern.Current.Value
        } elseif($method -eq 'uia.LegacyIAccessible'){
            $legacyStateBefore=[int64][AgentToolsLegacyUia]::State(
                [IntPtr]__HWND__,$nativeAutomationId,$runtimeBefore);
            if($legacyStateBefore -lt 0 -or
                $legacyStateBefore -gt [uint32]::MaxValue){
                Emit @{error='native_value_read_failed';method=$method} 6};
            if(($legacyStateBefore -band 0x20000000) -ne 0){
                Emit @{error='password_control'} 9};
            $readValue=[string][AgentToolsLegacyUia]::Value(
                [IntPtr]__HWND__,$nativeAutomationId,$runtimeBefore);
            $legacyStateAfter=[int64][AgentToolsLegacyUia]::State(
                [IntPtr]__HWND__,$nativeAutomationId,$runtimeBefore);
            if($legacyStateAfter -lt 0 -or
                $legacyStateAfter -gt [uint32]::MaxValue){
                $readValue=$null;Emit @{error='native_value_read_failed';method=$method} 6};
            if(($legacyStateAfter -band 0x20000000) -ne 0){
                $readValue=$null;Emit @{error='password_control'} 9}
        } else {Emit @{error='pattern_unavailable';method=$method} 6}
    }catch{
        $readValue=$null;
        Emit-Legacy-Safety $_;
        if(Is-ElementUnavailable $_){
            Emit @{error='stale_element';resolutionStage=$resolutionStage} 4};
        Emit @{error='native_value_read_failed';method=$method} 6
    };
    $runtimeAfter=($element.GetRuntimeId() -join ',');
    $passwordAfter=$element.Current.IsPassword;
    if($runtimeBefore -cne $runtimeAfter){
        Emit @{error='stale_element';runtimeId=$runtimeAfter} 4};
    if($passwordAfter -ne $false){$readValue=$null;Emit @{error='password_control'} 9};
    Assert-LegacyUnprotected;
    if($readValue.Length -gt 32000){
        $readValue=$null;Emit @{error='value_read_too_large'} 6};
    $strictUtf8=[System.Text.UTF8Encoding]::new($false,$true);
    try{$readBytes=$strictUtf8.GetBytes($readValue)
    }catch{$readValue=$null;Emit @{error='native_value_read_failed'} 6};
    if($readBytes.Length -gt 64000){
        $readValue=$null;$readBytes=$null;Emit @{error='value_read_too_large'} 6};
    $sha256=[System.Security.Cryptography.SHA256]::Create();
    try{$hashBytes=$sha256.ComputeHash($readBytes)}finally{$sha256.Dispose()};
    $readHash=([BitConverter]::ToString($hashBytes)).Replace('-','').ToLowerInvariant();
    Emit @{count=1;automationId=[string]$request.automationId;
        nativeAutomationId=[string]$element.Current.AutomationId;
        runtimeId=$runtimeAfter;method=$method;isPassword=$passwordAfter;
        valueUtf8Bytes=$readBytes.Length;valueUtf16Chars=$readValue.Length;
        valueSha256=$readHash;resolutionStage=$resolutionStage} 0};
$expected=$request.window;
function Assert-ActionsEnabled {
    $disabled=[string]$env:AGENT_TOOLS_COMPUTER_ACTIONS_DISABLED;
    if($disabled -match '^(?i:1|true|yes|on)$'){
        Emit @{error='computer_actions_disabled'} 7};
    if($request.disableMarker -and
        [System.IO.File]::Exists([string]$request.disableMarker)){
        Emit @{error='computer_actions_disabled'} 7}};
function Assert-FinalWindowIdentity {
    $handle=[IntPtr][int64]$expected.hwnd;
    if(-not [AgentToolsNativeSafety]::IsWindow($handle)){
        Emit @{error='stale_window';mismatchFields=@('window')} 8};
    [uint32]$actualPid=0;
    $actualThread=[AgentToolsNativeSafety]::GetWindowThreadProcessId(
        $handle,[ref]$actualPid);
    $mismatches=@();
    if($actualPid -ne [uint32]$expected.pid){$mismatches+='pid'};
    if($actualThread -ne [uint32]$expected.threadId){$mismatches+='thread_id'};
    if([AgentToolsNativeSafety]::WindowClass($handle) -cne
        [string]$expected.className){$mismatches+='window_class'};
    if($expected.requireTitleMatch -and
        [AgentToolsNativeSafety]::WindowText($handle) -cne [string]$expected.title){
        $mismatches+='title'};
    if($mismatches.Count -gt 0){
        Emit @{error='stale_window';mismatchFields=$mismatches} 8};
    $process=[System.Diagnostics.Process]::GetProcessById([int]$expected.pid);
    $processName=[System.IO.Path]::GetFileName($process.MainModule.FileName);
    if($processName -ine [string]$expected.process){$mismatches+='process_name'};
    if([int64]$process.StartTime.ToFileTime() -ne
        [int64]$expected.processStarted){$mismatches+='process_start_identity'};
    if([int]$process.SessionId -ne [int]$expected.sessionId){
        $mismatches+='session_id'};
    $desktop=[AgentToolsNativeSafety]::InputDesktopName();
    if($desktop -ine 'Default'){
        Emit @{error='secure_desktop_unavailable'} 8};
    if($desktop -ine [string]$expected.desktop){$mismatches+='input_desktop'};
    if($mismatches.Count -gt 0){
        Emit @{error='stale_window';mismatchFields=$mismatches} 8};
    $currentIntegrity=[AgentToolsNativeSafety]::IntegrityRid([uint32]0);
    $targetIntegrity=[AgentToolsNativeSafety]::IntegrityRid([uint32]$expected.pid);
    if($currentIntegrity -lt 0 -or $targetIntegrity -lt 0){
        Emit @{error='target_integrity_unavailable'} 8};
    if($targetIntegrity -gt $currentIntegrity){
        Emit @{error='target_integrity_denied'} 8}
};
function Assert-TargetSafe($pattern) {
    try {
        $handle=[IntPtr][int64]$expected.hwnd;
        if(-not [AgentToolsNativeSafety]::IsWindow($handle)){
            Emit @{error='stale_window';mismatchFields=@('window')} 8};
        [uint32]$actualPid=0;
        $actualThread=[AgentToolsNativeSafety]::GetWindowThreadProcessId(
            $handle,[ref]$actualPid);
        $windowMismatches=@();
        if($actualPid -ne [uint32]$expected.pid){$windowMismatches+='pid'};
        if($actualThread -ne [uint32]$expected.threadId){$windowMismatches+='thread_id'};
        if([AgentToolsNativeSafety]::WindowClass($handle) -cne
            [string]$expected.className){$windowMismatches+='window_class'};
        if($windowMismatches.Count -gt 0){
            Emit @{error='stale_window';mismatchFields=$windowMismatches} 8};
        if($expected.requireTitleMatch -and
            [AgentToolsNativeSafety]::WindowText($handle) -cne [string]$expected.title){
            Emit @{error='stale_window';mismatchFields=@('title')} 8};
        $process=[System.Diagnostics.Process]::GetProcessById([int]$expected.pid);
        $processName=[System.IO.Path]::GetFileName($process.MainModule.FileName);
        $processMismatches=@();
        if($processName -ine [string]$expected.process){
            $processMismatches+='process_name'};
        if([int64]$process.StartTime.ToFileTime() -ne
            [int64]$expected.processStarted){$processMismatches+='process_start_identity'};
        if([int]$process.SessionId -ne [int]$expected.sessionId){
            $processMismatches+='session_id'};
        if($processMismatches.Count -gt 0){
            Emit @{error='stale_window';mismatchFields=$processMismatches} 8};
        $desktop=[AgentToolsNativeSafety]::InputDesktopName();
        if($desktop -ine 'Default'){
            Emit @{error='secure_desktop_unavailable'} 8};
        if($desktop -ine [string]$expected.desktop){
            Emit @{error='stale_window';mismatchFields=@('input_desktop')} 8};
        $currentIntegrity=[AgentToolsNativeSafety]::IntegrityRid([uint32]0);
        $targetIntegrity=[AgentToolsNativeSafety]::IntegrityRid([uint32]$expected.pid);
        if($currentIntegrity -lt 0 -or $targetIntegrity -lt 0){
            Emit @{error='target_integrity_unavailable'} 8};
        if($targetIntegrity -gt $currentIntegrity){
            Emit @{error='target_integrity_denied'} 8};
        $finalTarget=Resolve-TargetElement;
        if(-not [System.Windows.Automation.Automation]::Compare(
            $finalTarget.element,$element)){
            Emit @{error='stale_element';
                resolutionStage=[string]$finalTarget.resolutionStage} 8};
        if($root.Current.NativeWindowHandle -ne [int]$expected.hwnd -or
            $root.Current.ProcessId -ne [int]$expected.pid -or
            ($resolutionStage -cne 'stable_geometry_unique' -and
                $element.Current.AutomationId -cne [string]$request.automationId) -or
            ($element.GetRuntimeId() -join ',') -cne $runtimeId){
            Emit @{error='stale_element'} 8};
        if($request.elementRef){
            $expectedType=([string]$request.elementRef.control_type).ToLowerInvariant();
            $expectedClass=([string]$request.elementRef.class).ToLowerInvariant();
            $expectedAncestor=[string]$request.elementRef.ancestor_signature;
            $elementType=Control-Type-Name $element;
            $elementTypeCompatible=Compatible-Control-Type `
                $elementType $expectedType $expectedClass;
            if((($expectedType) -and -not $elementTypeCompatible) -or
                (($expectedClass) -and
                    ([string]$element.Current.ClassName).ToLowerInvariant() -cne
                    $expectedClass) -or
                (-not $expectedAncestor) -or
                ($resolutionStage -cne 'stable_geometry_unique' -and
                    (Ancestor-Signature $element) -cne $expectedAncestor)){
                Emit @{error='stale_element';resolutionStage=$resolutionStage} 8}
        };
        if($element.Current.IsPassword){Emit @{error='password_control'} 9};
        Assert-LegacyUnprotected;
        if(-not $element.Current.IsEnabled){Emit @{error='control_disabled'} 9};
        if($method -eq 'uia.ValuePattern' -or $method -eq 'uia.RangeValuePattern'){
            try{$patternReadOnly=$pattern.Current.IsReadOnly}catch{
                if(Is-ElementUnavailable $_){
                    Emit @{error='stale_element';resolutionStage=$resolutionStage} 4};
                if(Is-PatternUnavailable $_){
                    Emit @{error='pattern_unavailable';method=$method;delivered=$false;
                        resolutionStage=$resolutionStage} 5};
                Emit @{error='target_safety_unavailable'} 8};
            if($patternReadOnly){Emit @{error='control_read_only'} 9}
        };
        $deliveryTarget=Resolve-TargetElement;
        if(-not [System.Windows.Automation.Automation]::Compare(
            $deliveryTarget.element,$element)){
            Emit @{error='stale_element';
                resolutionStage=[string]$deliveryTarget.resolutionStage} 8};
        Assert-FinalWindowIdentity;
        Assert-ReferenceFresh;
        Assert-OwnerAlive;
        Assert-ActionsEnabled;
        if([AgentToolsNativeSafety]::GetForegroundWindow() -ne $handle){
            Emit @{error='target_not_foreground'} 8};
    } catch {
        if(Is-ElementUnavailable $_){
            Emit @{error='stale_element';resolutionStage=$resolutionStage} 4};
        Emit @{error='target_safety_unavailable'} 8
    }
};
function Refresh-TargetElement {
    $freshTarget=Resolve-TargetElement;
    $freshElement=$freshTarget.element;
    $freshRuntimeId=($freshElement.GetRuntimeId() -join ',');
    $freshMismatches=@();
    if($freshRuntimeId -cne $runtimeId){$freshMismatches+='runtime_id'};
    if(-not [System.Windows.Automation.Automation]::Compare(
        $freshElement,$element)){$freshMismatches+='automation_element'};
    if($freshMismatches.Count -gt 0){
        Emit @{error='stale_element';
            mismatchFields=$freshMismatches;
            focusAttempted=$elementFocusAttempted;
            focusDelivered=$elementFocusDelivered;
            resolutionStage=[string]$freshTarget.resolutionStage} 8};
    $script:element=$freshElement;
    $script:resolutionStage=[string]$freshTarget.resolutionStage;
    $script:nativeAutomationId=[string]$script:element.Current.AutomationId
};
$deliveryAttempted=$false;
$keyboardFocusVerified=$null;
$elementFocusAttempted=$false;
$elementFocusDelivered=$false;
try {
    if($action -eq 'invoke'){
        if($method -eq 'uia.InvokePattern'){
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern);
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            $pattern.Invoke()
        } elseif($method -eq 'uia.TogglePattern'){
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern);
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            $pattern.Toggle()
        } elseif($method -eq 'uia.SelectionItemPattern'){
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern(
                [System.Windows.Automation.SelectionItemPattern]::Pattern);
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            $pattern.Select()
        } elseif($method -eq 'uia.ExpandCollapsePattern'){
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern(
                [System.Windows.Automation.ExpandCollapsePattern]::Pattern);
            $expand=$pattern.Current.ExpandCollapseState -eq
                [System.Windows.Automation.ExpandCollapseState]::Collapsed;
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            if($expand){$pattern.Expand()}
            else{$pattern.Collapse()}
        } else {Emit @{error='pattern_unavailable';method=$method} 5}
    } elseif($action -eq 'set-value'){
        if($method -eq 'uia.ValuePattern'){
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern);
            Assert-TargetSafe $pattern;
            if($request.requireKeyboardFocus -eq $true){
                $elementFocusAttempted=$true;
                try{$element.SetFocus();$elementFocusDelivered=$true}catch{
                    Emit @{error='keyboard_focus_unverified';delivered=$false;
                        focusAttempted=$true;focusDelivered=$false;method=$method} 12};
                try{
                    Refresh-TargetElement;
                    $pattern=$element.GetCurrentPattern(
                        [System.Windows.Automation.ValuePattern]::Pattern);
                    if($element.Current.HasKeyboardFocus -ne $true){
                        Emit @{error='keyboard_focus_unverified';delivered=$false;
                            focusAttempted=$true;focusDelivered=$true;method=$method} 12};
                    Assert-TargetSafe $pattern;
                    $keyboardFocusVerified=$true
                }catch{
                    Emit @{error='keyboard_focus_unverified';delivered=$false;
                        focusAttempted=$true;focusDelivered=$true;method=$method} 12}
            };
            $deliveryAttempted=$true;
            $pattern.SetValue([string]$request.value)
        } elseif($method -eq 'uia.RangeValuePattern'){
            $rangeBits=[string]$request.rangeBits;
            if($rangeBits -cnotmatch '^[0-9A-F]{16}$'){
                Emit @{error='invalid_range_value';delivered=$false} 6};
            $bytes=New-Object byte[] 8;
            for($index=0;$index -lt 8;$index++){
                $bytes[$index]=[Convert]::ToByte($rangeBits.Substring($index*2,2),16)};
            if([BitConverter]::IsLittleEndian){[Array]::Reverse($bytes)};
            $number=[BitConverter]::ToDouble($bytes,0);
            if([double]::IsNaN($number) -or [double]::IsInfinity($number)){
                Emit @{error='invalid_range_value';delivered=$false} 6};
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern(
                [System.Windows.Automation.RangeValuePattern]::Pattern);
            Assert-TargetSafe $pattern;
            if($request.requireKeyboardFocus -eq $true){
                $elementFocusAttempted=$true;
                try{$element.SetFocus();$elementFocusDelivered=$true}catch{
                    Emit @{error='keyboard_focus_unverified';delivered=$false;
                        focusAttempted=$true;focusDelivered=$false;method=$method} 12};
                try{
                    Refresh-TargetElement;
                    $pattern=$element.GetCurrentPattern(
                        [System.Windows.Automation.RangeValuePattern]::Pattern);
                    if($element.Current.HasKeyboardFocus -ne $true){
                        Emit @{error='keyboard_focus_unverified';delivered=$false;
                            focusAttempted=$true;focusDelivered=$true;method=$method} 12};
                    Assert-TargetSafe $pattern;
                    $keyboardFocusVerified=$true
                }catch{
                    Emit @{error='keyboard_focus_unverified';delivered=$false;
                        focusAttempted=$true;focusDelivered=$true;method=$method} 12}
            };
            $deliveryAttempted=$true;
            $pattern.SetValue($number)
        } elseif($method -eq 'uia.LegacyIAccessible'){
            Emit @{error='pattern_unavailable';method=$method;delivered=$false} 5
        } else {Emit @{error='pattern_unavailable';method=$method} 5}
    } elseif($action -eq 'scroll'){
        $mode=[string]$request.mode;
        if($mode -eq 'direction'){
            $no=[System.Windows.Automation.ScrollAmount]::NoAmount;
            $inc=[System.Windows.Automation.ScrollAmount]::SmallIncrement;
            $dec=[System.Windows.Automation.ScrollAmount]::SmallDecrement;
            $direction=[string]$request.direction;
            $horizontal=if($direction -eq 'right'){$inc}
                elseif($direction -eq 'left'){$dec}else{$no};
            $vertical=if($direction -eq 'down'){$inc}
                elseif($direction -eq 'up'){$dec}else{$no};
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern(
                [System.Windows.Automation.ScrollPattern]::Pattern);
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            $pattern.Scroll($horizontal,$vertical)
        } else {
            $target=[double]$request.percent;
            $no=[System.Windows.Automation.ScrollPattern]::NoScroll;
            $axis=[string]$request.axis;
            Refresh-TargetElement;
            $pattern=$element.GetCurrentPattern(
                [System.Windows.Automation.ScrollPattern]::Pattern);
            $horizontal=if($axis -eq 'horizontal'){$target}
                else{$no};
            $vertical=if($axis -eq 'vertical'){$target}
                else{$no};
            Assert-TargetSafe $pattern; $deliveryAttempted=$true;
            $pattern.SetScrollPercent($horizontal,$vertical)
        }
    } else {Emit @{error='action_unavailable'} 5};
    Emit @{count=1;automationId=[string]$request.automationId;
        nativeAutomationId=[string]$element.Current.AutomationId;runtimeId=$runtimeId;
        method=$method;delivered=$true;resolutionStage=$resolutionStage;
        keyboardFocusVerified=$keyboardFocusVerified} 0
} catch {
    if(-not $deliveryAttempted){
        Emit-Legacy-Safety $_
    };
    if(-not $deliveryAttempted -and (Is-ElementUnavailable $_)){
        Emit @{error='stale_element';runtimeId=$runtimeId;
            resolutionStage=$resolutionStage} 4};
    if(-not $deliveryAttempted -and (Is-PatternUnavailable $_)){
        Emit @{error='pattern_unavailable';runtimeId=$runtimeId;method=$method;
            delivered=$false;resolutionStage=$resolutionStage} 5};
    Emit @{error='action_failed';runtimeId=$runtimeId;method=$method;
        delivered=$deliveryAttempted} 6
}
} finally {
    if($workerAcquired){[void]$workerMutex.ReleaseMutex()};
    $workerMutex.Dispose()
}
"""
    )
    return script.replace("__HWND__", str(hwnd))


def _compressed_powershell_command(script: str) -> str:
    compressed = gzip.compress(script.encode("utf-8"), compresslevel=9)
    payload = base64.b64encode(compressed).decode("ascii")
    return (
        f"$b=[Convert]::FromBase64String('{payload}');"
        "$m=New-Object IO.MemoryStream(,$b);"
        "$g=New-Object IO.Compression.GzipStream("
        "$m,[IO.Compression.CompressionMode]::Decompress);"
        "$r=New-Object IO.StreamReader($g,[Text.Encoding]::UTF8);"
        "& ([ScriptBlock]::Create($r.ReadToEnd()))"
    )


def _discover_candidate() -> tuple[Path | None, str]:
    explicit = os.environ.get(WINAPP_ENV)
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve(), "environment"
        return None, "environment"
    managed = _discover_managed_candidate()
    if managed is not None:
        return managed, "managed_cache"
    if LOCAL_WINAPP.is_file():
        return LOCAL_WINAPP.resolve(), "local_cache"
    discovered = shutil.which("winapp.exe") or shutil.which("winapp")
    if discovered:
        return Path(discovered).resolve(), "path"
    return None, "missing"


def _managed_winapp_root() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        raise ComputerError(
            "capability_missing",
            "LOCALAPPDATA is unavailable for the managed WinApp backend. " + INSTALL_HINT,
            exit_code=2,
        )
    return Path(local) / "AgentTools" / "computer" / "backends" / "winapp"


def _managed_winapp_version_root() -> Path:
    return _managed_winapp_root() / f"v{WINAPP_VERSION}"


def _managed_winapp_executable() -> Path:
    return _managed_winapp_version_root() / "winapp.exe"


def _managed_cache_public_path() -> str | None:
    try:
        return str(_managed_winapp_executable())
    except ComputerError:
        return None


def _discover_managed_candidate() -> Path | None:
    try:
        root = _managed_winapp_version_root()
    except ComputerError:
        return None
    executable = root / "winapp.exe"
    companion = root / WINAPP_COMPANION
    manifest = root / WINAPP_CACHE_MANIFEST
    if executable.is_file() and companion.is_file() and manifest.is_file():
        return executable.resolve()
    return None


def _install_pinned_winapp() -> Path:
    try:
        root = _managed_winapp_root()
        _prepare_capture_store_root(root)
        with _capture_store_lock(root):
            existing = _discover_managed_candidate()
            if existing is not None:
                if (
                    _sha256(existing) == WINAPP_EXE_SHA256
                    and _sha256(existing.with_name(WINAPP_COMPANION)) == WINAPP_COMPANION_SHA256
                ):
                    return existing
            version_root = _managed_winapp_version_root()
            _prepare_capture_store_root(version_root)
            manifest = version_root / WINAPP_CACHE_MANIFEST
            try:
                manifest.unlink(missing_ok=True)
            except OSError as exc:
                raise ComputerError(
                    "uia_backend_install_failed",
                    "The incomplete managed WinApp backend could not be invalidated.",
                ) from exc
            archive = _temporary_path(version_root, ".winapp-download-", ".zip")
            staged_exe = _temporary_path(version_root, ".winapp-exe-", ".tmp")
            staged_companion = _temporary_path(
                version_root,
                ".winapp-companion-",
                ".tmp",
            )
            try:
                _download_winapp_archive(archive)
                if _sha256(archive) != WINAPP_RELEASE_SHA256:
                    raise ComputerError(
                        "uia_backend_install_failed",
                        "The downloaded WinApp archive failed its pinned integrity check.",
                    )
                _extract_winapp_runtime(
                    archive,
                    executable=staged_exe,
                    companion=staged_companion,
                )
                if (
                    _sha256(staged_exe) != WINAPP_EXE_SHA256
                    or _sha256(staged_companion) != WINAPP_COMPANION_SHA256
                ):
                    raise ComputerError(
                        "uia_backend_install_failed",
                        "The extracted WinApp runtime failed its pinned integrity check.",
                    )
                executable = version_root / "winapp.exe"
                companion = version_root / WINAPP_COMPANION
                os.replace(staged_exe, executable)
                os.replace(staged_companion, companion)
                _set_owner_only_acl(executable, directory=False)
                _set_owner_only_acl(companion, directory=False)
                _atomic_private_write(
                    manifest,
                    json.dumps(
                        {
                            "version": WINAPP_VERSION,
                            "release_sha256": WINAPP_RELEASE_SHA256,
                            "executable_sha256": WINAPP_EXE_SHA256,
                            "companion_sha256": WINAPP_COMPANION_SHA256,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii"),
                )
                return executable.resolve()
            finally:
                for temporary in (archive, staged_exe, staged_companion):
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
    except ComputerError as exc:
        if exc.code == "capability_missing":
            raise
        raise ComputerError(
            "capability_missing",
            "The pinned WinApp backend could not be installed automatically. " + INSTALL_HINT,
            exit_code=2,
            details={"install_error": exc.code},
        ) from exc
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        raise ComputerError(
            "capability_missing",
            "The pinned WinApp backend could not be installed automatically. " + INSTALL_HINT,
            exit_code=2,
            details={"install_error": type(exc).__name__},
        ) from exc


def _temporary_path(root: Path, prefix: str, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=prefix,
        suffix=suffix,
        dir=root,
        delete=False,
    ) as temporary:
        path = Path(temporary.name)
    _set_owner_only_acl(path, directory=False)
    return path


def _download_winapp_archive(output: Path) -> None:
    request = urllib.request.Request(
        WINAPP_RELEASE_URL,
        headers={"User-Agent": f"AgentTools/{WINAPP_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=WINAPP_DOWNLOAD_TIMEOUT_SECONDS) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in {
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        }:
            raise ComputerError(
                "uia_backend_install_failed",
                "The WinApp download redirected outside the pinned trusted hosts.",
            )
        declared = response.headers.get("Content-Length")
        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError as exc:
                raise ComputerError(
                    "uia_backend_install_failed",
                    "The WinApp download returned an invalid size.",
                ) from exc
            if declared_bytes <= 0 or declared_bytes > MAX_WINAPP_ARCHIVE_BYTES:
                raise ComputerError(
                    "uia_backend_install_failed",
                    "The WinApp download exceeded its bounded archive size.",
                )
        written = 0
        with output.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_WINAPP_ARCHIVE_BYTES:
                    raise ComputerError(
                        "uia_backend_install_failed",
                        "The WinApp download exceeded its bounded archive size.",
                    )
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
    if written <= 0:
        raise ComputerError(
            "uia_backend_install_failed",
            "The WinApp download was empty.",
        )


def _extract_winapp_runtime(
    archive: Path,
    *,
    executable: Path,
    companion: Path,
) -> None:
    expected = {"winapp.exe": executable, WINAPP_COMPANION: companion}
    with zipfile.ZipFile(archive) as bundle:
        infos = [info for info in bundle.infolist() if info.filename in expected]
        if len(infos) != len(expected) or {info.filename for info in infos} != set(expected):
            raise ComputerError(
                "uia_backend_install_failed",
                "The WinApp archive did not contain one exact pinned runtime set.",
            )
        total = sum(info.file_size for info in infos)
        if total <= 0 or total > MAX_WINAPP_EXTRACTED_BYTES:
            raise ComputerError(
                "uia_backend_install_failed",
                "The WinApp runtime exceeded its bounded extracted size.",
            )
        for info in infos:
            written = 0
            destination_path = expected[info.filename]
            with bundle.open(info, "r") as source, destination_path.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > info.file_size or written > MAX_WINAPP_EXTRACTED_BYTES:
                        raise ComputerError(
                            "uia_backend_install_failed",
                            "The WinApp runtime member exceeded its declared size.",
                        )
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if written != info.file_size:
                raise ComputerError(
                    "uia_backend_install_failed",
                    "The WinApp runtime member was truncated.",
                )


def _verify_winapp_companion(
    executable: Path,
    *,
    deadline: float | None = None,
) -> None:
    companion = executable.with_name(WINAPP_COMPANION)
    if not companion.is_file() or _sha256(companion, deadline=deadline) != WINAPP_COMPANION_SHA256:
        raise ComputerError(
            "backend_version_mismatch",
            "The WinApp runtime companion does not match the pinned release. " + INSTALL_HINT,
            exit_code=2,
        )


def _trusted_powershell() -> Path | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetSystemDirectoryW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint)
    kernel32.GetSystemDirectoryW.restype = ctypes.c_uint
    buffer = ctypes.create_unicode_buffer(32_768)
    length = int(kernel32.GetSystemDirectoryW(buffer, len(buffer)))
    if length <= 0 or length >= len(buffer):
        return None
    system_directory = Path(buffer.value)
    candidate = system_directory / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(system_directory.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _sha256(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                _require_deadline(deadline)
                digest.update(chunk)
    except OSError as exc:
        raise ComputerError(
            "uia_backend_unreadable",
            "The WinApp backend exists but cannot be read.",
            exit_code=2,
        ) from exc
    _require_deadline(deadline)
    return digest.hexdigest()


def _require_deadline(deadline: float | None) -> None:
    if deadline is not None and deadline - time.monotonic() <= 0:
        raise ComputerError(
            "uia_timeout",
            "The UI Automation backend did not respond within the safety deadline.",
        )


def _deadline_timeout(timeout_seconds: float, deadline: float | None) -> float:
    if deadline is None:
        return timeout_seconds
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _require_deadline(deadline)
    return min(timeout_seconds, remaining)


def _parse_version(result: ProcessResult) -> str | None:
    combined = result.stdout.decode("utf-8", errors="replace")
    matches = _VERSION_RE.findall(combined)
    return matches[-1] if matches else None


def _run_process(
    executable: Path,
    arguments: list[str],
    *,
    timeout_seconds: float,
    max_output_bytes: int,
    input_bytes: bytes | None = None,
    deadline: float | None = None,
) -> ProcessResult:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    effective_deadline = min(
        time.monotonic() + timeout_seconds,
        deadline if deadline is not None else math.inf,
    )
    input_stream = None
    process: subprocess.Popen[bytes] | None = None
    start_error: ComputerError | None = None
    try:
        if input_bytes is not None:
            input_stream = preloaded_process_stdin(input_bytes)
        if effective_deadline - time.monotonic() <= MIN_PROCESS_LIFECYCLE_SECONDS:
            raise ComputerError(
                "uia_timeout",
                "The UI Automation backend did not have enough lifecycle budget to start safely.",
            )
        process = subprocess.Popen(
            [str(executable), *arguments],
            cwd=executable.parent,
            env=_sanitized_environment(),
            stdin=input_stream if input_stream is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            shell=False,
            creationflags=creationflags,
        )
    except ProcessInputStagingError as exc:
        staging_details: dict[str, Any] = {"secondary_errors": [f"staging:{exc.stage_error}"]}
        if exc.cleanup_error is not None:
            staging_details["steps"] = [f"staged_stdin:{exc.cleanup_error}"]
            start_error = ComputerError(
                "uia_cleanup_failed",
                "The failed WinApp input stage could not be closed safely.",
                details=staging_details,
            )
        else:
            start_error = ComputerError(
                "uia_backend_start_failed",
                "The WinApp input could not be staged.",
                exit_code=2,
                details=staging_details,
            )
        start_error.__cause__ = exc
    except ComputerError as exc:
        start_error = exc
    except OSError as exc:
        start_error = ComputerError(
            "uia_backend_start_failed",
            "The WinApp backend could not be started.",
            exit_code=2,
        )
        start_error.__cause__ = exc
    staged_close_error = close_preloaded_process_stdin(input_stream)
    if process is None:
        if staged_close_error is not None:
            staged_close_details: dict[str, Any] = {
                "steps": [f"staged_stdin:{staged_close_error}"],
                "process_started": False,
                "lifecycle_stage": "pre_spawn_cleanup",
            }
            if start_error is not None:
                staged_close_details["primary_error"] = start_error.code
            raise ComputerError(
                "uia_cleanup_failed",
                "The staged WinApp input could not be closed safely.",
                details=staged_close_details,
            ) from start_error
        if start_error is not None:
            raise _with_process_lifecycle(
                start_error,
                process_started=False,
                stage="pre_spawn",
            )
        raise ComputerError(
            "uia_backend_start_failed",
            "The WinApp backend could not be started.",
            exit_code=2,
            details={
                "process_started": False,
                "lifecycle_stage": "pre_spawn",
            },
        )
    cleanup_reserve = min(0.25, max(0.01, timeout_seconds / 2))
    io_deadline = effective_deadline - cleanup_reserve
    if process.stdout is None or process.stderr is None:
        try:
            _terminate_process(process, effective_deadline)
        except ComputerError as exc:
            raise _with_process_lifecycle(
                exc,
                process_started=True,
                stage="post_spawn_termination",
            ) from exc
        raise ComputerError(
            "uia_pipe_unavailable",
            "The UI Automation backend pipe failed.",
            details={
                "process_started": True,
                "lifecycle_stage": "post_spawn_pre_io",
            },
        )

    streams = {"stdout": process.stdout, "stderr": process.stderr}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    closed: set[str] = set()
    total = 0
    failure: ComputerError | None = None
    if staged_close_error is not None:
        failure = ComputerError(
            "uia_cleanup_failed",
            "The staged WinApp input could not be closed safely.",
            details={
                "steps": [f"staged_stdin:{staged_close_error}"],
                "process_started": True,
                "lifecycle_stage": "post_spawn_cleanup",
            },
        )
    termination_error: ComputerError | None = None
    cleanup_errors: list[str] = []
    result: ProcessResult | None = None
    try:
        while process.poll() is None:
            if failure is not None:
                break
            for name, stream in streams.items():
                if name in closed:
                    continue
                chunk, stream_closed = _read_pipe_available(stream)
                if stream_closed:
                    closed.add(name)
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_output_bytes:
                    failure = ComputerError(
                        "uia_output_too_large",
                        "The UI Automation backend exceeded its output safety limit.",
                    )
                    break
                buffers[name].extend(chunk)
            if failure is not None:
                break
            remaining = io_deadline - time.monotonic()
            if remaining <= 0:
                failure = ComputerError(
                    "uia_timeout",
                    "The UI Automation backend did not respond within the safety deadline.",
                )
                break
            time.sleep(min(0.01, remaining))

        if process.poll() is None:
            try:
                _terminate_process(process, effective_deadline)
            except ComputerError as exc:
                termination_error = exc
        else:
            process.wait(timeout=max(0.0, effective_deadline - time.monotonic()))
        for name, stream in streams.items():
            while name not in closed:
                if time.monotonic() >= effective_deadline:
                    break
                chunk, stream_closed = _read_pipe_available(stream)
                if stream_closed:
                    closed.add(name)
                if chunk:
                    total += len(chunk)
                    if total > max_output_bytes:
                        failure = failure or ComputerError(
                            "uia_output_too_large",
                            "The UI Automation backend exceeded its output safety limit.",
                        )
                        break
                    buffers[name].extend(chunk)
                if not chunk and not stream_closed:
                    break
        if failure is None:
            result = ProcessResult(
                int(process.returncode or 0),
                bytes(buffers["stdout"]),
                bytes(buffers["stderr"]),
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if failure is None:
            failure = ComputerError(
                "uia_process_failed", "The UI Automation backend process failed."
            )
            failure.__cause__ = exc
        if process.poll() is None and termination_error is None:
            try:
                _terminate_process(process, effective_deadline)
            except ComputerError as termination:
                termination_error = termination
    finally:
        if process.poll() is None and termination_error is None:
            try:
                _terminate_process(process, effective_deadline)
            except ComputerError as exc:
                termination_error = exc
        for name in ("stdin", "stdout", "stderr"):
            pipe_stream = getattr(process, name, None)
            if pipe_stream is None or pipe_stream.closed:
                continue
            close_error = close_stream_os_enforced(pipe_stream)
            if close_error is not None:
                cleanup_errors.append(f"{name}:{close_error}")

    if termination_error is not None:
        _with_process_lifecycle(
            termination_error,
            process_started=True,
            stage="post_spawn_termination",
        )
        secondary = list(cleanup_errors)
        if failure is not None:
            secondary.extend(
                f"primary_cleanup:{step}"
                for step in failure.details.get("steps", [])
                if isinstance(step, str)
            )
            secondary.append(f"primary:{failure.code}")
        if secondary:
            termination_error.details["secondary_cleanup_errors"] = secondary[:8]
        raise termination_error
    if cleanup_errors:
        failure_steps = (
            [step for step in failure.details.get("steps", []) if isinstance(step, str)]
            if failure is not None
            else []
        )
        details: dict[str, Any] = {
            "steps": [*failure_steps, *cleanup_errors][:8],
            "process_started": True,
            "lifecycle_stage": "post_spawn_cleanup",
        }
        if failure is not None:
            details["primary_error"] = failure.code
        raise ComputerError(
            "uia_cleanup_failed",
            "The UI Automation backend pipes could not be closed safely.",
            details=details,
        )
    if failure is not None:
        raise _with_process_lifecycle(
            failure,
            process_started=True,
            stage="post_spawn_io",
        )
    if result is None:
        raise ComputerError(
            "uia_process_failed",
            "The UI Automation backend returned no process result.",
            details={
                "process_started": True,
                "lifecycle_stage": "post_spawn_io",
            },
        )
    return result


def _with_process_lifecycle(
    error: ComputerError,
    *,
    process_started: bool,
    stage: str,
) -> ComputerError:
    error.details.setdefault("process_started", process_started)
    error.details.setdefault("lifecycle_stage", stage)
    return error


def _terminate_process(process: subprocess.Popen[bytes], deadline: float) -> None:
    try:
        process.kill()
    except OSError as exc:
        raise ComputerError(
            "uia_termination_failed",
            "The UI Automation backend could not be terminated safely.",
        ) from exc
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ComputerError(
            "uia_termination_failed",
            "The UI Automation backend termination could not be verified before the deadline.",
        )
    try:
        process.wait(timeout=remaining)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComputerError(
            "uia_termination_failed",
            "The UI Automation backend could not be terminated safely.",
        ) from exc


def _read_pipe_available(stream: Any) -> tuple[bytes, bool]:
    if sys.platform == "win32":
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.PeekNamedPipe.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
            wintypes.LPDWORD,
        )
        kernel32.PeekNamedPipe.restype = wintypes.BOOL
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        available = wintypes.DWORD()
        if not kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            error = ctypes.get_last_error()
            if error in {109, 232}:
                return b"", True
            raise OSError(error, "PeekNamedPipe failed")
        if available.value == 0:
            return b"", False
        chunk = os.read(stream.fileno(), min(4096, available.value))
        return chunk, not chunk
    readable, _, _ = select.select([stream], [], [], 0)
    if not readable:
        return b"", False
    chunk = os.read(stream.fileno(), 4096)
    return chunk, not chunk


def _sanitized_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in (
        "SystemRoot",
        "WINDIR",
        "ComSpec",
        "TEMP",
        "TMP",
        "LOCALAPPDATA",
        "PROGRAMDATA",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "WINAPP_CLI_UPDATE_CHECK": "0",
            "WINAPP_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
            "NO_COLOR": "1",
        }
    )
    return environment


def _single_window(payload: dict[str, Any], identity: WindowIdentity) -> dict[str, Any]:
    windows = payload.get("windows")
    if not isinstance(windows, list) or len(windows) != 1 or not isinstance(windows[0], dict):
        raise ComputerError(
            "ambiguous_window_target",
            "The UI Automation backend did not resolve exactly one target window.",
        )
    window = windows[0]
    if type(window.get("hwnd")) is not int or window["hwnd"] != identity.hwnd:
        raise ComputerError(
            "stale_window",
            "The UI Automation backend resolved a different target window.",
        )
    return window


def _flatten_elements(
    raw_elements: list[Any],
    *,
    max_elements: int,
    provider_filtered_hierarchy: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    traversal_truncated = False
    traversal_limit = max_elements + 1

    def visit(
        raw: Any,
        depth: int,
        parent: str | None,
        ancestor_segments: list[str],
        ancestor_identity_exact: bool,
    ) -> None:
        nonlocal traversal_truncated
        if len(output) >= traversal_limit:
            traversal_truncated = True
            return
        if not isinstance(raw, dict):
            return
        selector, selector_overflow = _exact_optional_text(raw.get("selector"), MAX_SELECTOR_CHARS)
        control_type, control_type_overflow = _exact_optional_text(raw.get("type"), 64)
        automation_id, automation_id_overflow = _exact_optional_text(raw.get("automationId"), 160)
        class_name, class_name_overflow = _exact_optional_text(
            raw.get("className"), MAX_CLASS_NAME_CHARS
        )
        identity_exact = not any(
            (
                selector_overflow,
                control_type_overflow,
                automation_id_overflow,
                class_name_overflow,
            )
        )
        filtered_hierarchy = provider_filtered_hierarchy or "ancestorPath" in raw
        provider_ancestor_path = _filtered_provider_ancestor_path(raw.get("ancestorPath"))
        reported_depth: int | None = depth
        reported_parent = parent
        if filtered_hierarchy:
            reported_depth = (
                len(provider_ancestor_path) if provider_ancestor_path is not None else None
            )
            reported_parent = None
        actionable = bool(raw.get("isInvokable")) or control_type in _ACTIONABLE_TYPES
        item = {
            "element": selector,
            "automation_id": automation_id,
            "name": _safe_optional_text(raw.get("name"), MAX_TITLE_CHARS),
            "control_type": control_type,
            "class": class_name,
            "bounds": _bounds_from_element(raw).to_jsonable(),
            "depth": reported_depth,
            "parent": reported_parent,
            "enabled": _parse_bool(raw.get("isEnabled")),
            "offscreen": _parse_bool(raw.get("isOffscreen")),
            "actionable": actionable,
            "scroll_direction": _safe_optional_text(raw.get("scrollDir"), 4),
            "_ancestor_signature": (
                "sha256:"
                + hashlib.sha256("\n".join(ancestor_segments).encode("utf-8")).hexdigest()[:24]
                if ancestor_identity_exact and not filtered_hierarchy
                else None
            ),
            "_identity_exact": identity_exact,
            "_ancestor_identity_exact": (
                ancestor_identity_exact and not filtered_hierarchy
            ),
            "_provider_filtered_hierarchy": filtered_hierarchy,
            "_provider_ancestor_path": provider_ancestor_path,
            "_order": len(output),
        }
        output.append(item)
        children = raw.get("children")
        if isinstance(children, list):
            segment = "|".join(
                (
                    automation_id or "",
                    (control_type or "").casefold(),
                    (class_name or "").casefold(),
                )
            )
            for child in children:
                if traversal_truncated:
                    break
                child_ancestor_identity_exact = ancestor_identity_exact and identity_exact
                visit(
                    child,
                    (reported_depth if reported_depth is not None else depth) + 1,
                    selector or parent,
                    [segment, *ancestor_segments],
                    child_ancestor_identity_exact and not filtered_hierarchy,
                )

    for element in raw_elements:
        if traversal_truncated:
            break
        visit(element, 0, None, [], True)
    return output, traversal_truncated


def _filtered_provider_ancestor_path(value: object) -> list[str] | None:
    if not isinstance(value, list) or len(value) > 12:
        return None
    output: list[str] = []
    for raw_segment in value:
        segment, overflow = _exact_optional_text(raw_segment, 64)
        if not segment or overflow:
            return None
        output.append(segment)
    return output


def _deduplicate_candidates(matches: list[Any]) -> list[dict[str, Any]]:
    bounded: list[tuple[str, dict[str, Any]]] = []
    selector_counts: dict[str, int] = {}
    for raw in matches:
        if not isinstance(raw, dict):
            continue
        selector, _overflow = _exact_optional_text(raw.get("selector"), MAX_SELECTOR_CHARS)
        if not selector:
            continue
        bounded.append((selector, raw))
        selector_counts[selector] = selector_counts.get(selector, 0) + 1
    return [raw for selector, raw in bounded if selector_counts[selector] == 1]


def _normalize_scroll_area(
    candidate: dict[str, Any],
    properties: dict[str, Any],
    native: dict[str, Any],
    identity: WindowIdentity,
    binding: WinAppBinding,
) -> dict[str, Any] | None:
    horizontal_scrollable = _parse_bool(properties.get("HorizontallyScrollable"))
    vertical_scrollable = _parse_bool(properties.get("VerticallyScrollable"))
    horizontal_percent = _parse_percent(properties.get("ScrollHorizontalPercent"))
    vertical_percent = _parse_percent(properties.get("ScrollVerticalPercent"))
    horizontal_scrollable = bool(horizontal_scrollable or horizontal_percent is not None)
    vertical_scrollable = bool(vertical_scrollable or vertical_percent is not None)
    if not horizontal_scrollable and not vertical_scrollable:
        return None
    bounds = _bounds_from_element(candidate)
    warning = None
    if (horizontal_scrollable and _parse_percent(properties.get("HorizontalViewSize")) is None) or (
        vertical_scrollable and _parse_percent(properties.get("VerticalViewSize")) is None
    ):
        warning = "view_size_percent_unavailable"
    selector, _selector_overflow = _exact_optional_text(
        candidate.get("selector"), MAX_SELECTOR_CHARS
    )
    automation_id, _automation_id_overflow = _exact_optional_text(native.get("automation_id"), 160)
    control_type, _control_type_overflow = _exact_optional_text(native.get("control_type"), 64)
    class_name, _class_name_overflow = _exact_optional_text(
        native.get("class"), MAX_CLASS_NAME_CHARS
    )
    ancestor_signature, _ancestor_signature_overflow = _exact_optional_text(
        native.get("ancestor_signature"), 96
    )
    return {
        "element": selector,
        "provider": "winapp",
        "automation_id": automation_id,
        "name": _safe_optional_text(candidate.get("name"), MAX_TITLE_CHARS),
        "control_type": control_type,
        "class": class_name,
        "hwnd": identity.hwnd,
        "pid": identity.pid,
        "bounds": bounds.to_jsonable(),
        "viewport": {"width": bounds.width, "height": bounds.height},
        "horizontal": _axis_state(
            horizontal_scrollable,
            horizontal_percent,
            _parse_percent(properties.get("HorizontalViewSize")),
        ),
        "vertical": _axis_state(
            vertical_scrollable,
            vertical_percent,
            _parse_percent(properties.get("VerticalViewSize")),
        ),
        "backend": binding.path.name,
        "backend_version": binding.version,
        "warning": warning,
        "enabled": native.get("enabled"),
        "read_only": None,
        "password": native.get("is_password"),
        "locator": (
            {
                "provider": "winapp",
                "strategy": "selector_v1",
                "selector": selector,
                "runtime_id": native.get("runtime_id"),
                "automation_id": automation_id,
                "ancestor_signature": ancestor_signature,
            }
            if not any(
                (
                    _selector_overflow,
                    _automation_id_overflow,
                    _control_type_overflow,
                    _class_name_overflow,
                    _ancestor_signature_overflow,
                )
            )
            else None
        ),
        "hierarchy": {"ancestor_signature": ancestor_signature},
    }


def _has_semantic_scroll_state(properties: dict[str, Any]) -> bool:
    return bool(
        _parse_bool(properties.get("HorizontallyScrollable"))
        or _parse_bool(properties.get("VerticallyScrollable"))
        or _parse_percent(properties.get("ScrollHorizontalPercent")) is not None
        or _parse_percent(properties.get("ScrollVerticalPercent")) is not None
    )


def _scroll_provider_identity(
    candidate: dict[str, Any],
    properties: dict[str, Any],
) -> dict[str, Any]:
    control_type, control_type_overflow = _exact_optional_text(
        properties.get("ControlType") or candidate.get("type"),
        64,
    )
    class_name, class_name_overflow = _exact_optional_text(
        properties.get("ClassName") or candidate.get("className"),
        MAX_CLASS_NAME_CHARS,
    )
    bounds = _bounds_from_element(candidate)
    if (
        not control_type
        or control_type_overflow
        or class_name_overflow
        or bounds.width <= 0
        or bounds.height <= 0
    ):
        raise ComputerError(
            "uia_identity_unavailable",
            "The provider scroll element has incomplete exact identity metadata.",
        )
    return {
        "controlType": control_type,
        "className": class_name,
        "bounds": bounds.to_jsonable(),
    }


def _axis_state(
    scrollable: bool, percent: float | None, view_size_percent: float | None
) -> dict[str, Any]:
    if not scrollable:
        return {
            "scrollable": False,
            "percent": None,
            "percent_status": "no_scroll",
            "view_size_percent": None,
            "moreBefore": False,
            "moreAfter": False,
        }
    if percent is None:
        return {
            "scrollable": True,
            "percent": None,
            "percent_status": "unavailable",
            "view_size_percent": view_size_percent,
            "moreBefore": None,
            "moreAfter": None,
        }
    return {
        "scrollable": True,
        "percent": percent,
        "percent_status": "available",
        "view_size_percent": view_size_percent,
        "moreBefore": percent > 0.01,
        "moreAfter": percent < 99.99,
    }


def _bounds_from_element(value: dict[str, Any]) -> Bounds:
    try:
        x = int(str(value.get("x")))
        y = int(str(value.get("y")))
        width = max(0, int(str(value.get("width"))))
        height = max(0, int(str(value.get("height"))))
    except (TypeError, ValueError) as exc:
        raise _invalid_response() from exc
    return Bounds(x=x, y=y, width=width, height=height)


def _parse_bool(value: object) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, (int, str)):
        normalized = str(value).casefold()
        if normalized in {"true", "1", "0x1"}:
            return True
        if normalized in {"false", "0", "0x0"}:
            return False
    return None


def _optional_bool(value: object) -> bool | None:
    return value if type(value) is bool else None


def _parse_percent(value: object) -> float | None:
    try:
        percent = float(str(value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(percent) or percent < 0 or percent > 100:
        return None
    return round(percent, 4)


def _safe_optional_text(value: object, limit: int) -> str | None:
    if value in (None, ""):
        return None
    return bounded_text(value, limit)


def _exact_optional_text(value: object, limit: int) -> tuple[str | None, bool]:
    if value in (None, ""):
        return None, False
    text = str(value)
    if len(text) > limit:
        return None, True
    return text, False


def _exact_element_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ELEMENT_ID_CHARS:
        raise _invalid_response()
    return value


def _password_state(payload: dict[str, Any]) -> tuple[str, bool]:
    properties = payload.get("properties")
    if not isinstance(properties, dict):
        raise ComputerError(
            "uia_sensitive_state_unavailable",
            "The element's password state could not be verified; no text was read.",
        )
    try:
        element_id = _exact_element_id(payload.get("elementId"))
    except ComputerError as exc:
        raise ComputerError(
            "uia_sensitive_state_unavailable",
            "The element's password state could not be verified; no text was read.",
        ) from exc
    is_password = _parse_bool(properties.get("IsPassword"))
    if is_password is None:
        raise ComputerError(
            "uia_sensitive_state_unavailable",
            "The element's password state could not be verified; no text was read.",
        )
    return element_id, is_password


def _password_redacted(selector: str, element_id: str) -> dict[str, Any]:
    return {
        "element": selector,
        "resolved_element": element_id,
        "text": None,
        "char_count": 0,
        "truncated": False,
        "redacted": True,
        "redaction_reason": "password_control",
    }


def _validate_selector(value: str) -> str:
    if (
        not value
        or len(value) > MAX_SELECTOR_CHARS
        or value.startswith("-")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ComputerError("invalid_element", "The UI Automation element selector is invalid.")
    return value


def _reject_read_only_provider_locator(value: str) -> None:
    if value.startswith("nuia-"):
        suffix = value[len("nuia-") :]
        if len(suffix) != 24 or any(character not in "0123456789abcdef" for character in suffix):
            raise ComputerError(
                "invalid_element",
                "The UI Automation element selector is invalid.",
            )
        raise _semantic_action_unavailable(
            "semantic_provider_action_unsupported",
            "The selected inspection provider has no safe semantic action path.",
        )


def _is_native_provider_locator(value: str) -> bool:
    if not value.startswith("nuia-"):
        return False
    suffix = value[len("nuia-") :]
    if len(suffix) != 24 or any(character not in "0123456789abcdef" for character in suffix):
        raise ComputerError(
            "invalid_element",
            "The UI Automation element selector is invalid.",
        )
    return True


def _semantic_action_unavailable(code: str, message: str) -> ComputerError:
    return ComputerError(
        code,
        message,
        details={
            "fallback": {
                "kind": "guarded_physical_input",
                "status": "available",
                "reason": "requires_fresh_capture_and_explicit_allow_physical",
            }
        },
    )


def _action_preflight_unavailable(error: ComputerError) -> ComputerError:
    fallback_error = _semantic_action_unavailable(error.code, error.message)
    return ComputerError(
        error.code,
        error.message,
        exit_code=error.exit_code,
        details={
            **error.details,
            **fallback_error.details,
            "outcome": "rejected",
        },
    )


def _resolution_details(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"resolution_stage": "native_worker"}
    stage = payload.get("resolutionStage")
    if stage not in {
        "automation_id_unique",
        "provider_identity_unique",
        "runtime_id_exact",
        "stable_metadata_missing",
        "stable_metadata_unique",
        "stable_geometry_unique",
        "bounded_geometry_candidates",
        "reference_metadata",
        "reference_freshness",
    }:
        stage = "native_worker"
    details: dict[str, Any] = {"resolution_stage": stage}
    count = payload.get("count")
    if type(count) is int and 0 <= count <= MAX_INSPECT_ELEMENTS:
        details["candidate_count"] = count
    return details


def _title_transaction_hashes(before: str, after: str) -> dict[str, str]:
    key = secrets.token_bytes(32)

    def fingerprint(value: str) -> str:
        return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    return {
        "title_before_hmac_sha256": fingerprint(before),
        "title_after_hmac_sha256": fingerprint(after),
    }


def _revalidate_wrapper_identity(
    backend: Win32Backend,
    identity: WindowIdentity,
    *,
    require_title_match: bool,
) -> WindowIdentity:
    if require_title_match:
        return backend.revalidate_identity(identity, require_title_match=True)
    return backend.revalidate_identity(identity)


def _map_backend_error(stderr: bytes) -> ComputerError:
    try:
        payload = json.loads(stderr.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    code = str(error.get("code") or "").casefold() if isinstance(error, dict) else ""
    message = str(error.get("message") or "").casefold() if isinstance(error, dict) else ""
    details = str(error.get("details") or "").casefold() if isinstance(error, dict) else ""
    if "appnotfound" in details or "not accessible" in message:
        return ComputerError(
            "uia_provider_unavailable",
            "The target window is not accessible through UI Automation.",
        )
    if "ambiguous" in code or "multiple" in message or "more than one" in message:
        return ComputerError("ambiguous_element", "The UI Automation selector is ambiguous.")
    if "not_found" in code or "not found" in message:
        return ComputerError("element_not_found", "The UI Automation element was not found.")
    if "changed" in message or "stale" in code:
        return ComputerError("stale_element", "The UI Automation element changed; inspect again.")
    return ComputerError("uia_backend_error", "The UI Automation backend rejected the request.")


def _invalid_response() -> ComputerError:
    return ComputerError(
        "uia_invalid_response",
        "The UI Automation backend returned an invalid bounded response.",
    )
