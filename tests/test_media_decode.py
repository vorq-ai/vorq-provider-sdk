"""Reading a reference's real dimensions out of its own bytes.

The client *declares* what a reference is, and the input side of the order is
priced on that declaration. Nothing on the wire holds it to the truth, so this is
where the truth is established: after decrypting, the daemon reads the header the
reference actually carries and refuses a job whose reference is larger than the
units bought for it.

Everything here is header arithmetic on stdlib bytes. The daemon has no image or
video dependency and must not grow one — a decoder that renders is a decoder that
can be made to allocate, on attacker-supplied bytes, before anything is paid for.
The fixtures are built byte by byte rather than checked in, so the format each
reader relies on is written down beside the reader.
"""

from __future__ import annotations

import struct

import pytest

from vorqd.errors import BackendError
from vorqd.media import decode_dimensions


def png(width: int, height: int) -> bytes:
    """Signature, then an IHDR whose first eight bytes are the dimensions."""
    ihdr = struct.pack(">II", width, height) + bytes([8, 6, 0, 0, 0])
    return (b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + b"\x00\x00\x00\x00")


def jpeg(width: int, height: int, *, marker: bytes = b"\xff\xc0") -> bytes:
    """SOI, a segment to skip over, then a start-of-frame carrying height then width."""
    comment = b"\xff\xfe" + struct.pack(">H", 2 + 4) + b"skip"
    sof = marker + struct.pack(">H", 2 + 7) + bytes([8]) + struct.pack(">HH", height, width) + b"\x03"
    return b"\xff\xd8" + comment + sof + b"\xff\xd9"


def webp_vp8(width: int, height: int) -> bytes:
    """A lossy VP8 chunk: dimensions are 14-bit, after a 3-byte start code."""
    body = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", width, height)
    chunk = b"VP8 " + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_vp8l(width: int, height: int) -> bytes:
    """Lossless: 14-bit dimensions minus one, packed little-endian after a 0x2f byte."""
    packed = (width - 1) | ((height - 1) << 14)
    body = b"\x2f" + struct.pack("<I", packed)[:4]
    chunk = b"VP8L" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def webp_vp8x(width: int, height: int) -> bytes:
    """Extended: canvas dimensions as 24-bit values minus one."""
    body = b"\x10\x00\x00\x00" + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    chunk = b"VP8X" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def _timing(version: int, timescale: int, ticks: int) -> bytes:
    """The run `mvhd` and `mdhd` share: version, two timestamps, timescale, duration."""
    if version == 0:
        return bytes([0, 0, 0, 0]) + struct.pack(">II", 0, 0) + struct.pack(">II", timescale, ticks)
    return bytes([1, 0, 0, 0]) + struct.pack(">QQ", 0, 0) + struct.pack(">IQ", timescale, ticks)


def tkhd(width: int, height: int, *, version: int = 0) -> bytes:
    if version == 0:
        times = struct.pack(">II", 0, 0) + struct.pack(">II", 1, 0) + struct.pack(">I", 0)
    else:
        times = struct.pack(">QQ", 0, 0) + struct.pack(">II", 1, 0) + struct.pack(">Q", 0)
    return _box(b"tkhd", bytes([version, 0, 0, 0]) + times + b"\x00" * 52
                + struct.pack(">II", width << 16, height << 16))


def stsd(width: int, height: int) -> bytes:
    """A sample description whose one visual entry states the *coded* frame size —
    the number a decoder believes, whatever the track header says for display."""
    entry = (b"\x00" * 6 + struct.pack(">H", 1)          # reserved, data reference index
             + b"\x00" * 16                              # pre_defined / reserved
             + struct.pack(">HH", width, height) + b"\x00" * 50)
    return _box(b"stsd", bytes(4) + struct.pack(">I", 1) + _box(b"avc1", entry))


def media(*, handler: bytes = b"vide", timescale: int = 600, ticks: int = 0,
          coded: tuple[int, int] | None = None) -> bytes:
    """A track's `mdia`: its own clock, what kind of track it is, and what it codes."""
    hdlr = _box(b"hdlr", bytes(4) + bytes(4) + handler + bytes(12) + b"\x00")
    inner = _box(b"mdhd", _timing(0, timescale, ticks)) + hdlr
    if coded is not None:
        inner += _box(b"minf", _box(b"stbl", stsd(*coded)))
    return _box(b"mdia", inner)


def movie(*traks: bytes, timescale: int = 600, ticks: int = 2400, version: int = 0,
          extra: bytes = b"", tail: bytes = b"") -> bytes:
    return (_box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2")
            + _box(b"moov", _box(b"mvhd", _timing(version, timescale, ticks))
                   + b"".join(_box(b"trak", t) for t in traks) + extra)
            + tail)


def mp4(width: int, height: int, *, seconds: int = 4, timescale: int = 600,
        version: int = 0) -> bytes:
    """ftyp, then a moov carrying an mvhd (how long) and a tkhd (how large).

    Both boxes come in a 32-bit and a 64-bit flavour and real files use both, so
    the reader has to branch on the version byte rather than assume an offset.
    """
    return movie(tkhd(width, height, version=version),
                 timescale=timescale, ticks=seconds * timescale, version=version)


# --- the happy paths ----------------------------------------------------------


@pytest.mark.parametrize("build,media_type", [
    (png, "image/png"),
    (jpeg, "image/jpeg"),
    (webp_vp8, "image/webp"),
    (webp_vp8l, "image/webp"),
    (webp_vp8x, "image/webp"),
])
def test_a_still_reports_its_own_dimensions(build, media_type):
    assert decode_dimensions(build(1280, 720), media_type) == (1280, 720, None)


@pytest.mark.parametrize("marker", [b"\xff\xc0", b"\xff\xc1", b"\xff\xc2"])
def test_every_start_of_frame_a_jpeg_may_use_is_read(marker):
    """Baseline, extended and progressive are three markers for one thing, and a
    reader that knew only `\\xff\\xc0` would refuse perfectly ordinary photographs."""
    assert decode_dimensions(jpeg(640, 480, marker=marker), "image/jpeg") == (640, 480, None)


@pytest.mark.parametrize("version", [0, 1])
def test_a_clip_reports_its_dimensions_and_its_length(version):
    """Duration is the timescale divided out — a number of ticks means nothing
    without the ticks-per-second beside it."""
    assert decode_dimensions(mp4(1920, 1080, seconds=7, version=version),
                             "video/mp4") == (1920, 1080, 7)


def test_a_clip_shorter_than_a_second_still_counts_as_one():
    """Pricing floors a still at one pixel-second, and a 400 ms clip cannot be
    cheaper than a photograph of the same size."""
    assert decode_dimensions(movie(tkhd(64, 64), ticks=240), "video/mp4")[2] == 1


def test_a_clips_length_rounds_up_rather_than_down():
    """Rounding down would let a caller buy 1 second and hand over 1.9."""
    raw = mp4(64, 64, seconds=1, timescale=1000)
    patched = raw.replace(struct.pack(">I", 1000) + struct.pack(">I", 1000),
                          struct.pack(">I", 1000) + struct.pack(">I", 1900))
    assert decode_dimensions(patched, "video/mp4")[2] == 2


# --- the refusals -------------------------------------------------------------


@pytest.mark.parametrize("raw,why", [
    (b"", "empty"),
    (b"\x89PNG\r\n\x1a\n", "a signature and nothing else"),
    (png(8, 8)[:20], "an IHDR cut in half"),
    (b"\xff\xd8\xff\xc0\x00", "a JPEG segment cut in half"),
    (b"\xff\xd8" + b"\xff\xfe\x00\x04ab", "a JPEG with no start of frame at all"),
    (b"RIFF\x04\x00\x00\x00WEBP", "a WebP with no chunk"),
    (b"RIFF\x0c\x00\x00\x00WEBPVP8 \x01\x00\x00\x00", "a VP8 chunk too short to read"),
    (mp4(8, 8)[:16], "an MP4 truncated inside its first box"),
    (_box(b"ftyp", b"isom"), "an MP4 with no moov"),
    (b"not an image at all, just prose", "prose"),
])
def test_bytes_that_are_not_the_media_they_claim_are_refused(raw, why):
    """A clean refusal, never an IndexError or a struct.error.

    These bytes are attacker-supplied: they arrive sealed inside a container this
    daemon has already claimed and paid to open. A decoder that raises something
    the caller does not catch takes the whole sweep down rather than the one job.
    """
    with pytest.raises(BackendError):
        decode_dimensions(raw, "image/png" if raw[:4] != b"RIFF" else "image/webp")


def test_a_media_type_this_daemon_cannot_read_is_named_in_the_refusal():
    with pytest.raises(BackendError, match="image/gif"):
        decode_dimensions(png(8, 8), "image/gif")


def test_a_declared_type_that_the_bytes_contradict_is_refused():
    """The media type is the client's claim too. PNG bytes labelled as a video are
    not a video, and reading them as one would be reading attacker-chosen offsets.
    """
    with pytest.raises(BackendError):
        decode_dimensions(png(1280, 720), "video/mp4")


def test_a_dimension_of_zero_is_refused_rather_than_priced_as_free():
    with pytest.raises(BackendError):
        decode_dimensions(png(0, 720), "image/png")


def test_a_deeply_nested_box_structure_does_not_run_away():
    """A box declaring its own size as zero is the classic MP4 parser hang: the
    walk never advances and the loop never ends. It must terminate, refusing.
    """
    zero_sized = struct.pack(">I", 0) + b"moov" + b"\x00" * 16
    with pytest.raises(BackendError):
        decode_dimensions(_box(b"ftyp", b"isom") + zero_sized, "video/mp4")


# --- headers that understate what the bytes are --------------------------------
#
# Each of these is a file an ordinary decoder reads as large and a careless header
# reader reads as small. The reference leg of an order is priced on this reader's
# answer, so an under-read is a provider doing work nobody paid for.


def test_fill_bytes_before_a_jpeg_marker_are_padding_not_a_segment():
    """Any number of `FF` may precede a marker. Read as a marker of its own, `FF FF`
    takes the next two bytes for a length and jumps wherever the file's author
    likes — past the real frame header and onto a smaller one hidden in a comment.
    """
    real = jpeg(640, 480)
    fake_sof = b"\xff\xc0" + struct.pack(">H", 9) + bytes([8]) + struct.pack(">HH", 16, 16) + b"\x03"
    # `FF` `FF E1 0004 xxxx`: padding then a 4-byte APP1 to a decoder; a segment of
    # length 0xE100 to a reader that takes the second `FF` for a marker kind.
    lead = b"\xff" + b"\xff\xe1" + struct.pack(">H", 4) + b"xx"
    jump = 4 + 0xE100                      # where the misreading lands
    body = lead + real[2:-2]
    pad = jump - (2 + len(body)) - 4       # a comment holding the decoy at `jump`
    comment = b"\xff\xfe" + struct.pack(">H", 2 + pad + len(fake_sof)) + b"\x00" * pad + fake_sof
    crafted = b"\xff\xd8" + body + comment + b"\xff\xd9"
    assert crafted[jump:jump + 2] == b"\xff\xc0"
    assert decode_dimensions(crafted, "image/jpeg") == (640, 480, None)


def test_an_honest_jpeg_with_a_fill_byte_is_read():
    raw = jpeg(640, 480)
    assert decode_dimensions(raw[:2] + b"\xff" + raw[2:], "image/jpeg") == (640, 480, None)


def test_the_largest_track_is_the_one_measured_not_the_last():
    """A subtitle or thumbnail track after the video must not be what gets priced."""
    raw = movie(tkhd(1920, 1080), tkhd(16, 16))
    assert decode_dimensions(raw, "video/mp4")[:2] == (1920, 1080)


def test_a_coded_size_larger_than_the_display_size_is_the_one_priced():
    """`tkhd` is a presentation hint a decoder ignores; `stsd` is what it decodes."""
    raw = movie(tkhd(16, 16) + media(coded=(1920, 1080)))
    assert decode_dimensions(raw, "video/mp4")[:2] == (1920, 1080)


def test_an_audio_tracks_sample_entry_is_not_read_as_a_frame():
    """The same offsets in an audio entry hold a sample rate. 44100 x anything is
    not a picture, and an honest clip with sound must still be read as itself."""
    raw = movie(tkhd(320, 240) + media(coded=(320, 240)),
                tkhd(0, 0) + media(handler=b"soun", coded=(44100, 2)))
    assert decode_dimensions(raw, "video/mp4")[:2] == (320, 240)


def test_the_longest_clock_in_the_file_is_the_clips_length():
    """A movie header saying one second over a track that runs ten is ten."""
    raw = movie(tkhd(64, 64) + media(timescale=1000, ticks=10_000), ticks=600)
    assert decode_dimensions(raw, "video/mp4")[2] == 10


def test_a_fragmented_clip_is_as_long_as_its_fragments_say():
    """Fragmented files legitimately write a zero movie duration; the real one is
    in `mvex/mehd`, in the movie's own timescale."""
    mehd = _box(b"mehd", bytes(4) + struct.pack(">I", 600 * 60))
    raw = movie(tkhd(64, 64), ticks=0, extra=_box(b"mvex", mehd), tail=_box(b"moof", b""))
    assert decode_dimensions(raw, "video/mp4")[2] == 60


def test_a_fragmented_clip_that_does_not_say_how_long_it_is_is_refused():
    """With fragments and no `mehd` the length is only knowable by reading every
    fragment, and a header reader that guessed would be guessing one second."""
    raw = movie(tkhd(64, 64), ticks=600, tail=_box(b"moof", b""))
    with pytest.raises(BackendError, match="fragment"):
        decode_dimensions(raw, "video/mp4")


def test_a_clip_with_no_length_at_all_is_refused_rather_than_priced_as_a_second():
    with pytest.raises(BackendError, match="duration"):
        decode_dimensions(movie(tkhd(64, 64), ticks=0), "video/mp4")


def test_a_file_of_nothing_but_box_headers_is_refused_before_it_is_walked():
    """Eight bytes a box, a reference may carry millions, and the walk runs on the
    event loop every other job shares."""
    raw = _box(b"ftyp", b"isom") + _box(b"free", b"") * 200_000
    with pytest.raises(BackendError, match="boxes"):
        decode_dimensions(raw, "video/mp4")


def test_a_frame_of_encoder_overshoot_does_not_cost_a_whole_second():
    """Measured off a real generated clip: 97 frames at 24 fps is 4.042 s, from a
    model asked for 4. Every encoder does this, so a strict round-up reads every
    honest "four second" clip as five and refuses it after the claim.
    """
    raw = movie(tkhd(752, 560) + media(timescale=12288, ticks=49664, coded=(752, 560)),
                timescale=1000, ticks=4042)
    assert decode_dimensions(raw, "video/mp4") == (752, 560, 4)


def test_the_allowance_is_a_few_frames_and_not_a_discount():
    assert decode_dimensions(movie(tkhd(64, 64), timescale=1000, ticks=4100), "video/mp4")[2] == 4
    assert decode_dimensions(movie(tkhd(64, 64), timescale=1000, ticks=4101), "video/mp4")[2] == 5
