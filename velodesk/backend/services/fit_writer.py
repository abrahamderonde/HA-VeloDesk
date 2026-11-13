"""
FIT workout file generator for TrainIQ.
FIT SDK message numbers: file_id=0, workout=26, workout_step=27
Duration value for DURATION_TIME is in SECONDS (not milliseconds).
"""
import struct
import logging
from datetime import datetime, date
from typing import List, Dict

logger = logging.getLogger(__name__)

FIT_EPOCH = 631065600  # seconds between Unix epoch (1970) and FIT epoch (1989-12-31)

# Correct FIT global message numbers from the official SDK
MESG_FILE_ID      = 0
MESG_WORKOUT      = 26   # NOT 190 (that's golf_course)
MESG_WORKOUT_STEP = 27   # NOT 196

# Base types
ENUM   = 0x00
UINT8  = 0x02
UINT16 = 0x84
UINT32 = 0x86
STRING = 0x07

# Duration types
DURATION_TIME = 0  # value = seconds

# Target types
TARGET_OPEN  = 0
TARGET_POWER = 6   # use custom_target_value_low/high for watt range

# Intensity
INTENSITY_ACTIVE   = 0
INTENSITY_REST     = 1
INTENSITY_WARMUP   = 2
INTENSITY_COOLDOWN = 3

SPORT_CYCLING = 2


def _fit_time(dt) -> int:
    if isinstance(dt, date) and not isinstance(dt, datetime):
        dt = datetime(dt.year, dt.month, dt.day)
    return int(dt.timestamp()) - FIT_EPOCH


def _crc16(data: bytes, crc: int = 0) -> int:
    t = [0x0000,0xCC01,0xD801,0x1400,0xF001,0x3C00,0x2800,0xE401,
         0xA001,0x6C00,0x7800,0xB401,0x5000,0x9C01,0x8801,0x4400]
    for b in data:
        crc = (crc>>4)&0x0FFF ^ t[crc&0xF] ^ t[b&0xF]
        crc = (crc>>4)&0x0FFF ^ t[crc&0xF] ^ t[(b>>4)&0xF]
    return crc


def _str_bytes(s: str, size: int) -> bytes:
    """Encode string to exactly `size` bytes, null-terminated and null-padded."""
    b = (s or '').encode('utf-8')[:size-1]
    return b + b'\x00' * (size - len(b))


def _defn(local: int, global_num: int, fields: list) -> bytes:
    """Build a FIT definition message."""
    buf = bytearray()
    buf.append(0x40 | (local & 0x0F))   # definition header
    buf.append(0)                         # reserved
    buf.append(0)                         # little-endian
    buf += struct.pack('<H', global_num)
    buf.append(len(fields))
    for fnum, fsize, ftype in fields:
        buf += bytes([fnum, fsize, ftype])
    return bytes(buf)


def _data(local: int, fields: list, values: list) -> bytes:
    """Build a FIT data message."""
    buf = bytearray([local & 0x0F])
    for (fnum, fsize, ftype), val in zip(fields, values):
        if ftype == STRING:
            buf += _str_bytes(str(val) if val else '', fsize)
        elif fsize == 1:
            buf.append(int(val if val is not None else 0xFF) & 0xFF)
        elif fsize == 2:
            buf += struct.pack('<H', int(val if val is not None else 0xFFFF) & 0xFFFF)
        elif fsize == 4:
            buf += struct.pack('<I', int(val if val is not None else 0xFFFFFFFF) & 0xFFFFFFFF)
    return bytes(buf)


def generate_workout_fit(workout) -> bytes:
    """Generate a valid Garmin FIT workout file from a PlannedWorkout."""
    now = _fit_time(datetime.now())
    name = (workout.title or 'Workout')[:16]
    intervals = workout.intervals or []
    num_steps = _count_steps(intervals)

    rec = bytearray()

    # ── file_id ───────────────────────────────────────────────────────────
    F0 = [(0,1,ENUM),(1,2,UINT16),(2,2,UINT16),(4,4,UINT32)]
    rec += _defn(0, MESG_FILE_ID, F0)
    rec += _data(0, F0, [5, 255, 0, now])

    # ── workout ───────────────────────────────────────────────────────────
    F1 = [(4,1,ENUM),(5,2,UINT16),(8,16,STRING)]
    rec += _defn(1, MESG_WORKOUT, F1)
    rec += _data(1, F1, [SPORT_CYCLING, num_steps, name])

    # ── workout_step ──────────────────────────────────────────────────────
    F2 = [
        (1,  16, STRING), # wkt_step_name
        (2,  1, ENUM),    # duration_type  (0 = time, value in seconds)
        (3,  4, UINT32),  # duration_value (seconds)
        (4,  1, ENUM),    # target_type    (6 = power)
        (5,  4, UINT32),  # target_value   (0 = use custom low/high)
        (6,  4, UINT32),  # custom_target_value_low  (watts)
        (7,  4, UINT32),  # custom_target_value_high (watts)
        (11, 1, ENUM),    # intensity
    ]
    rec += _defn(2, MESG_WORKOUT_STEP, F2)

    if not intervals:
        dur = (workout.target_duration_minutes or 60) * 60
        rec += _data(2, F2, ['Ride',DURATION_TIME,dur,TARGET_OPEN,0,0,0,INTENSITY_ACTIVE])
    else:
        idx = 0
        for iv in intervals:
            for step in _expand(iv, idx):
                rec += _data(2, F2, [
                    step['name'],
                    DURATION_TIME, step['dur'],
                    step['ttype'], 0,
                    step['tlow'], step['thigh'],
                    step['intensity'],
                ])
                idx = step['idx'] + 1

    # ── assemble ──────────────────────────────────────────────────────────
    data_bytes = bytes(rec)
    hdr = bytearray()
    hdr.append(14)
    hdr.append(0x10)
    hdr += struct.pack('<H', 2132)
    hdr += struct.pack('<I', len(data_bytes))
    hdr += b'.FIT'
    hdr += struct.pack('<H', _crc16(bytes(hdr)))
    return bytes(hdr) + data_bytes + struct.pack('<H', _crc16(data_bytes))


def _count_steps(intervals: list) -> int:
    if not intervals:
        return 1
    n = 0
    for iv in intervals:
        reps = int(iv.get('repeats', 1))
        has_rest = int(iv.get('rest_seconds', 0)) > 0
        n += reps * (2 if has_rest else 1)
    return n


def _expand(iv: dict, start_idx: int) -> list:
    itype    = iv.get('type', 'work')
    dur      = int(iv.get('duration_seconds', 300))
    reps     = int(iv.get('repeats', 1))
    rest     = int(iv.get('rest_seconds', 0))
    p_low    = iv.get('power_low')
    p_high   = iv.get('power_high')

    ttype    = TARGET_POWER if p_low is not None else TARGET_OPEN
    t_low    = int(p_low)  if p_low  is not None else 0
    t_high   = int(p_high) if p_high is not None else 0

    imap = {'work':INTENSITY_ACTIVE,'recovery':INTENSITY_REST,
            'warmup':INTENSITY_WARMUP,'cooldown':INTENSITY_COOLDOWN}
    intensity = imap.get(itype, INTENSITY_ACTIVE)

    steps = []
    idx = start_idx
    for r in range(reps):
        label = f'Int {r+1}/{reps}' if reps > 1 else itype.capitalize()
        steps.append({'idx':idx,'name':label[:16],'dur':dur,
                      'ttype':ttype,'tlow':t_low,'thigh':t_high,'intensity':intensity})
        idx += 1
        if rest > 0:
            r_low, r_high = iv.get('rest_power_low'), iv.get('rest_power_high')
            steps.append({'idx':idx,'name':'Rest','dur':rest,
                          'ttype':TARGET_POWER if r_low else TARGET_OPEN,
                          'tlow':int(r_low) if r_low else 0,
                          'thigh':int(r_high) if r_high else 0,
                          'intensity':INTENSITY_REST})
            idx += 1
    return steps
