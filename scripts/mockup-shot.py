#!/usr/bin/env python3
"""Crop the uniform background off the bottom of a PNG, with nothing but the standard library.

Used to turn a tall `chromium --headless --screenshot` into a full-page photograph whose height is
the page's own, the way `docs/ui/` photographs are. Assumes a non-interlaced 8-bit RGB/RGBA PNG,
which is exactly what headless Chromium writes.
"""
import struct
import sys
import zlib


def read_png(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    pos, idat, meta = 8, b'', None
    while pos < len(data):
        length, kind = struct.unpack('>I4s', data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b'IHDR':
            meta = struct.unpack('>IIBBBBB', chunk)
        elif kind == b'IDAT':
            idat += chunk
        elif kind == b'IEND':
            break
        pos += 12 + length
    width, height, depth, colour, *_ = meta
    assert depth == 8 and colour in (2, 6) and not _[-1], 'need 8-bit non-interlaced RGB/RGBA'
    channels = 3 if colour == 2 else 4
    stride = width * channels
    raw = zlib.decompress(idat)
    rows, previous, pos = [], bytearray(stride), 0
    for _ in range(height):
        filter_type = raw[pos]; pos += 1
        line = bytearray(raw[pos:pos + stride]); pos += stride
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            x = line[i]
            if filter_type == 1: line[i] = (x + a) & 255
            elif filter_type == 2: line[i] = (x + b) & 255
            elif filter_type == 3: line[i] = (x + ((a + b) >> 1)) & 255
            elif filter_type == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (x + pred) & 255
        rows.append(bytes(line)); previous = line
    return width, height, channels, rows


def write_png(path, width, height, channels, rows):
    stride = width * channels
    raw = b''.join(b'\x00' + row for row in rows)
    body = zlib.compress(raw, 6)
    def chunk(kind, payload):
        return (struct.pack('>I', len(payload)) + kind + payload
                + struct.pack('>I', zlib.crc32(kind + payload) & 0xffffffff))
    colour = 2 if channels == 3 else 6
    with open(path, 'wb') as handle:
        handle.write(b'\x89PNG\r\n\x1a\n')
        handle.write(chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, colour, 0, 0, 0)))
        handle.write(chunk(b'IDAT', body))
        handle.write(chunk(b'IEND', b''))


def main(source, target, background_rows=1, pad=34):
    width, height, channels, rows = read_png(source)
    background = rows[0]
    last = 0
    for index, row in enumerate(rows):
        if row != background:
            last = index
    bottom = min(height, last + 1 + pad)
    write_png(target, width, bottom, channels, rows[:bottom])
    print('%s: %dx%d -> %dx%d' % (source, width, height, width, bottom))


if __name__ == '__main__':
    main(*sys.argv[1:])
