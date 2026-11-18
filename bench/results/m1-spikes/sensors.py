"""Log every temperature sensor the HID event system exposes, once a second, without root.

    python sensors.py <out.jsonl>

Line 1: {"names": [...]}. Then one line a second: {"t": unix seconds, "v": [degrees C or null]}.
Private, undocumented interface: a hint about heat, never proof of it.
"""

import ctypes
import json
import signal
import sys
import time
from ctypes import c_bool, c_char_p, c_double, c_int32, c_int64, c_long, c_uint32, c_void_p

cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
io = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
UTF8 = 0x08000100
TEMPERATURE = 15

cf.CFStringCreateWithCString.restype = c_void_p
cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
cf.CFNumberCreate.restype = c_void_p
cf.CFNumberCreate.argtypes = [c_void_p, c_long, c_void_p]
cf.CFDictionaryCreate.restype = c_void_p
cf.CFDictionaryCreate.argtypes = [c_void_p, c_void_p, c_void_p, c_long, c_void_p, c_void_p]
cf.CFArrayGetCount.restype = c_long
cf.CFArrayGetCount.argtypes = [c_void_p]
cf.CFArrayGetValueAtIndex.restype = c_void_p
cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_long]
cf.CFStringGetCString.restype = c_bool
cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_long, c_uint32]
cf.CFRelease.restype = None
cf.CFRelease.argtypes = [c_void_p]
io.IOHIDEventSystemClientCreate.restype = c_void_p
io.IOHIDEventSystemClientCreate.argtypes = [c_void_p]
io.IOHIDEventSystemClientSetMatching.restype = c_int32
io.IOHIDEventSystemClientSetMatching.argtypes = [c_void_p, c_void_p]
io.IOHIDEventSystemClientCopyServices.restype = c_void_p
io.IOHIDEventSystemClientCopyServices.argtypes = [c_void_p]
io.IOHIDServiceClientCopyProperty.restype = c_void_p
io.IOHIDServiceClientCopyProperty.argtypes = [c_void_p, c_void_p]
io.IOHIDServiceClientCopyEvent.restype = c_void_p
io.IOHIDServiceClientCopyEvent.argtypes = [c_void_p, c_int64, c_int32, c_int64]
io.IOHIDEventGetFloatValue.restype = c_double
io.IOHIDEventGetFloatValue.argtypes = [c_void_p, c_int32]


def _cfstr(text: str) -> int:
    return cf.CFStringCreateWithCString(None, text.encode(), UTF8)


def _cfnum(n: int) -> int:
    value = c_int32(n)
    return cf.CFNumberCreate(None, 3, ctypes.byref(value))


def services() -> list[tuple[str, int]]:
    keys = (c_void_p * 2)(_cfstr("PrimaryUsagePage"), _cfstr("PrimaryUsage"))
    vals = (c_void_p * 2)(_cfnum(0xFF00), _cfnum(5))
    kcb = ctypes.addressof(c_void_p.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
    vcb = ctypes.addressof(c_void_p.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
    match = cf.CFDictionaryCreate(None, keys, vals, 2, kcb, vcb)
    client = io.IOHIDEventSystemClientCreate(None)
    io.IOHIDEventSystemClientSetMatching(client, match)
    found = io.IOHIDEventSystemClientCopyServices(client)
    product = _cfstr("Product")
    out = []
    for i in range(cf.CFArrayGetCount(found) if found else 0):
        svc = cf.CFArrayGetValueAtIndex(found, i)
        ref = io.IOHIDServiceClientCopyProperty(svc, product)
        buf = ctypes.create_string_buffer(256)
        name = buf.value.decode() if ref and cf.CFStringGetCString(ref, buf, 256, UTF8) else "?"
        out.append((buf.value.decode(errors="replace") or name, svc))
    return out


def read(svc: int) -> float | None:
    event = io.IOHIDServiceClientCopyEvent(svc, TEMPERATURE, 0, 0)
    if not event:
        return None
    value = io.IOHIDEventGetFloatValue(event, TEMPERATURE << 16)
    cf.CFRelease(event)
    return round(value, 2)


def main() -> None:
    found = services()
    names = [f"{name}#{i}" for i, (name, _) in enumerate(found)]
    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    with open(sys.argv[1], "w") as out:
        out.write(json.dumps({"names": names}) + "\n")
        next_at = time.time()
        while not stop:
            out.write(json.dumps({"t": round(time.time(), 3), "v": [read(s) for _, s in found]}) + "\n")
            out.flush()
            next_at += 1.0
            time.sleep(max(0.0, next_at - time.time()))


if __name__ == "__main__":
    main()
