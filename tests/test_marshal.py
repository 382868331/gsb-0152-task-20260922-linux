"""Call marshalling tests: register file, stack image, fallback rules.

Expected register/stack bytes are hand-computed with int.to_bytes and
struct.pack directly in this file -- never via the core's encoder.
"""

import struct
import unittest

from abi_marshal import (
    F64,
    I8,
    I64,
    PTR,
    ArrayType,
    CallImage,
    ImageError,
    I16,
    RecordType,
    pack_call,
    plan_call,
    unpack_call,
)

ZERO8 = b"\x00" * 8


def i64le(v):
    return v.to_bytes(8, "little", signed=True)


def u64le(v):
    return v.to_bytes(8, "little")


def f64le(v):
    return struct.pack("<d", v)


class ScalarRegisterTest(unittest.TestCase):
    def test_int_and_float_scalars_take_registers(self):
        img = pack_call([I16, F64], [0x0102, 1.5])
        self.assertEqual(img.int_regs[0], b"\x02\x01" + b"\x00" * 6)
        self.assertEqual(img.float_regs[0], f64le(1.5))
        self.assertEqual(img.int_regs[1:], (None, None, None))
        self.assertEqual(img.float_regs[1], None)
        self.assertEqual(img.stack, b"")

    def test_four_int_registers_then_spill_to_stack(self):
        img = pack_call([I64] * 5, [1, 2, 3, 4, 5])
        self.assertEqual(
            img.int_regs, (i64le(1), i64le(2), i64le(3), i64le(4))
        )
        # 5th scalar spills to an 8-byte-aligned stack slot at offset 0
        self.assertEqual(img.stack, i64le(5))
        self.assertEqual(img.placements[4].chunks, (("stack", 0),))

    def test_two_float_registers_then_spill(self):
        img = pack_call([F64] * 3, [1.0, 2.0, 3.0])
        self.assertEqual(img.float_regs, (f64le(1.0), f64le(2.0)))
        self.assertEqual(img.stack, f64le(3.0))

    def test_pointer_uses_integer_register(self):
        img = pack_call([PTR], [0xDEADBEEF])
        self.assertEqual(img.int_regs[0], u64le(0xDEADBEEF))


class AggregateRegisterTest(unittest.TestCase):
    def test_mixed_class_aggregate_split_across_register_kinds(self):
        mixed = RecordType("M", [("tag", I8), ("weight", F64)])
        img = pack_call([I64, I64, I64, mixed], [1, 2, 3, {"tag": 0x41, "weight": 2.5}])
        # chunk 0 (tag + padding) is INTEGER -> int reg 3
        self.assertEqual(img.int_regs[3], b"\x41" + b"\x00" * 7)
        # chunk 1 (weight) is FLOAT -> float reg 0
        self.assertEqual(img.float_regs[0], f64le(2.5))
        self.assertEqual(img.stack, b"")

    def test_all_float_aggregate_uses_two_float_registers(self):
        point = RecordType("P", [("x", F64), ("y", F64)])
        img = pack_call([point], [{"x": 1.0, "y": -2.0}])
        self.assertEqual(img.float_regs, (f64le(1.0), f64le(-2.0)))
        self.assertEqual(img.int_regs, (None,) * 4)

    def test_over_16_byte_aggregate_goes_to_stack(self):
        wide = RecordType("W", [("a", I64), ("b", I64), ("c", I64)])
        img = pack_call([wide], [{"a": 10, "b": 11, "c": 12}])
        self.assertEqual(img.int_regs, (None,) * 4)
        self.assertEqual(img.stack, i64le(10) + i64le(11) + i64le(12))
        self.assertEqual(
            img.placements[0].chunks,
            (("stack", 0), ("stack", 8), ("stack", 16)),
        )


class WholeAggregateFallbackTest(unittest.TestCase):
    def test_shortfall_sends_whole_aggregate_to_stack_and_keeps_registers(self):
        pair = RecordType("P", [("a", I64), ("b", I64)])
        # 3 int registers used; pair needs 2 but only 1 remains.
        img = pack_call([I64, I64, I64, pair, I64], [1, 2, 3, {"a": 10, "b": 11}, 4])
        # the pair consumes NO registers...
        self.assertEqual(img.int_regs[:3], (i64le(1), i64le(2), i64le(3)))
        # ...so the trailing scalar still gets int register 3
        self.assertEqual(img.int_regs[3], i64le(4))
        # and the pair sits whole on the stack, 16-byte aligned
        self.assertEqual(img.placements[3].chunks, (("stack", 0), ("stack", 8)))
        self.assertEqual(img.stack, i64le(10) + i64le(11))

    def test_float_shortfall_fallback(self):
        point = RecordType("P", [("x", F64), ("y", F64)])
        img = pack_call([F64, point], [0.5, {"x": 1.0, "y": 2.0}])
        # only 1 float register left, point needs 2 -> whole to stack
        self.assertEqual(img.float_regs[0], f64le(0.5))
        self.assertEqual(img.float_regs[1], None)
        self.assertEqual(img.stack, f64le(1.0) + f64le(2.0))

    def test_stack_aggregate_slots_are_16_byte_aligned(self):
        pair = RecordType("P", [("a", I64), ("b", I64)])
        # 5 scalars: 4 registers + 1 spilled at stack offset 0 (8 bytes).
        # The pair then falls back; its slot must start at offset 16.
        img = pack_call(
            [I64] * 5 + [pair], [1, 2, 3, 4, 5, {"a": 10, "b": 11}]
        )
        self.assertEqual(img.placements[5].chunks[0], ("stack", 16))
        self.assertEqual(img.placements[5].chunks[0][1] % 16, 0)
        self.assertEqual(
            img.stack,
            i64le(5) + ZERO8 + i64le(10) + i64le(11),
        )

    def test_memory_aggregate_slot_is_16_byte_aligned_after_scalar_spill(self):
        wide = RecordType("W", [("a", I64), ("b", I64), ("c", I64)])
        img = pack_call([I64] * 5 + [wide], [1, 2, 3, 4, 5, {"a": 6, "b": 7, "c": 8}])
        self.assertEqual(img.placements[5].chunks[0], ("stack", 16))
        self.assertEqual(len(img.stack), 16 + 24)


class EmptyAndNestedTest(unittest.TestCase):
    def test_empty_record_consumes_no_registers_no_stack(self):
        empty = RecordType("E")
        img = pack_call([empty, I64], [{}, 7])
        self.assertEqual(img.placements[0].chunks, ())
        self.assertEqual(img.int_regs[0], i64le(7))
        self.assertEqual(img.stack, b"")
        self.assertEqual(unpack_call([empty, I64], img), [{}, 7])

    def test_array_of_empty_records_consumes_nothing(self):
        empty_arr = ArrayType(RecordType("E"), 3)
        img = pack_call([empty_arr, F64], [[{}, {}, {}], 9.0])
        self.assertEqual(img.placements[0].chunks, ())
        self.assertEqual(img.float_regs[0], f64le(9.0))
        self.assertEqual(img.stack, b"")

    def test_nested_array_roundtrip_bytes(self):
        rec = RecordType("N", [("vals", ArrayType(I16, 3)), ("flag", I8)])
        img = pack_call([rec], [{"vals": [0x0201, 0x0403, 0x0605], "flag": 0x07}])
        # layout: vals at 0..6, flag at 6, size 8 -> one INTEGER eightbyte
        expect = (
            b"\x01\x02\x03\x04\x05\x06\x07\x00"
        )
        self.assertEqual(img.int_regs[0], expect)
        self.assertEqual(img.stack, b"")


class RoundTripTest(unittest.TestCase):
    def test_roundtrip_mixed_signature(self):
        mixed = RecordType("M", [("tag", I8), ("weight", F64)])
        pair = RecordType("P", [("a", I64), ("b", I64)])
        wide = RecordType("W", [("x", I64), ("y", I64), ("z", I64)])
        empty = RecordType("E")
        types = [I64, I64, I64, mixed, empty, pair, F64, wide, ArrayType(I16, 2)]
        values = [
            1,
            -2,
            3,
            {"tag": 9, "weight": 2.5},
            {},
            {"a": 10, "b": -11},
            3.25,
            {"x": 100, "y": 200, "z": 300},
            [0x7FFF, -2],
        ]
        img = pack_call(types, values)
        got = unpack_call(types, img)
        self.assertEqual(got, values)

    def test_roundtrip_is_bit_exact_for_floats(self):
        vals = [0.1, -0.0, 1e300, -1e-300, 2.0**-1074]
        img = pack_call([F64] * 5, vals)
        got = unpack_call([F64] * 5, img)
        for want, have in zip(vals, got):
            self.assertEqual(struct.pack("<d", want), struct.pack("<d", have))


class UnpackImageTest(unittest.TestCase):
    def setUp(self):
        self.pair = RecordType("P", [("a", I64), ("b", I64)])
        self.types = [I64, I64, I64, self.pair]
        self.values = [1, 2, 3, {"a": 10, "b": 11}]
        self.img = pack_call(self.types, self.values)

    def test_truncated_stack_rejected(self):
        bad = CallImage(self.img.int_regs, self.img.float_regs, b"\x00" * 8, ())
        with self.assertRaises(ImageError):
            unpack_call(self.types, bad)

    def test_missing_register_rejected(self):
        regs = list(self.img.int_regs)
        regs[0] = None
        bad = CallImage(tuple(regs), self.img.float_regs, self.img.stack, ())
        with self.assertRaises(ImageError):
            unpack_call(self.types, bad)

    def test_wrong_register_width_rejected(self):
        regs = list(self.img.int_regs)
        regs[0] = b"\x01" * 4
        bad = CallImage(tuple(regs), self.img.float_regs, self.img.stack, ())
        with self.assertRaises(ImageError):
            unpack_call(self.types, bad)

    def test_wrong_register_count_rejected(self):
        bad = CallImage((ZERO8,) * 3, self.img.float_regs, self.img.stack, ())
        with self.assertRaises(ImageError):
            unpack_call(self.types, bad)

    def test_non_image_rejected(self):
        with self.assertRaises(ImageError):
            unpack_call(self.types, object())


class PlanTest(unittest.TestCase):
    def test_plan_reports_stack_size(self):
        pair = RecordType("P", [("a", I64), ("b", I64)])
        plan = plan_call([I64] * 4 + [pair])
        self.assertEqual(plan.stack_size, 16)
        self.assertEqual(len(plan.placements), 5)


if __name__ == "__main__":
    unittest.main()
