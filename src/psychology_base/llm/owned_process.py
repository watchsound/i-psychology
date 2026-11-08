"""Windows job ownership for diagnostic subprocess trees, never shared services."""
import ctypes
from ctypes import wintypes
import time


class _Limits(ctypes.Structure):
    _fields_ = [('process_time',ctypes.c_longlong), ('job_time',ctypes.c_longlong),
        ('flags',wintypes.DWORD), ('min_working',ctypes.c_size_t), ('max_working',ctypes.c_size_t),
        ('active_limit',wintypes.DWORD), ('affinity',ctypes.c_size_t),
        ('priority',wintypes.DWORD), ('scheduling',wintypes.DWORD)]


class _IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('reads','writes','other','read_bytes','write_bytes','other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic',_Limits), ('io',_IO), ('process_memory',ctypes.c_size_t),
        ('job_memory',ctypes.c_size_t), ('peak_process',ctypes.c_size_t), ('peak_job',ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [(name,ctypes.c_longlong) for name in ('user','kernel','period_user','period_kernel')] + [
        (name,wintypes.DWORD) for name in ('faults','total','active','terminated')]


class WindowsProcessJob:
    """An unnamed kill-on-close job holding only our newly created subprocess."""
    def __init__(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE,wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE,wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel, self.handle = kernel, kernel.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel.CloseHandle(self.handle)
            self.handle = None
            raise error

    def assign(self, process):
        if self.handle is None or not self.kernel.AssignProcessToJobObject(self.handle,int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def stop(self):
        """Terminate the owned tree even if its original parent has already exited."""
        if self.handle is None:
            raise RuntimeError('Owned job has already been closed')
        try:
            if not self.kernel.TerminateJobObject(self.handle,1):
                raise ctypes.WinError(ctypes.get_last_error())
            until = time.monotonic()+10
            while True:
                value = _Accounting()
                if not self.kernel.QueryInformationJobObject(self.handle,1,ctypes.byref(value),ctypes.sizeof(value),None):
                    raise ctypes.WinError(ctypes.get_last_error())
                if value.active == 0:
                    return dict(verified_exit=True, job_verified_empty=True, tracked_processes=value.total)
                if time.monotonic() >= until:
                    raise RuntimeError('Owned job still contains live processes after cleanup deadline')
                time.sleep(0.05)
        finally:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
