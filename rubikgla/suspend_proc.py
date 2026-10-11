"""可逆挂起/恢复指定 PID (NtSuspendProcess/NtResumeProcess via ctypes).
用法: python suspend_proc.py <pid> suspend|resume
"""
import ctypes
import sys

pid = int(sys.argv[1])
action = sys.argv[2]

ntdll = ctypes.WinDLL("ntdll")
kernel32 = ctypes.WinDLL("kernel32")
PROCESS_SUSPEND_RESUME = 0x0800
h = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
if not h:
    print("OpenProcess 失败 err=%d" % ctypes.get_last_error())
    sys.exit(1)
if action == "suspend":
    r = ntdll.NtSuspendProcess(h)
elif action == "resume":
    r = ntdll.NtResumeProcess(h)
else:
    print("action 必须是 suspend|resume")
    sys.exit(1)
print("PID %d %s → NTSTATUS=0x%X" % (pid, action, r & 0xFFFFFFFF))
