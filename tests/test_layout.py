"""Layout and classification tests.

The expected offsets come from two sources independent of the core:
  1. ``reference_record_layout`` below, a standalone re-implementation of
     the alignment formula written for this test file (it never imports
     or calls abi_marshal's layout machinery), and
  2. hand-computed constants for the representative cases.
"""

import unittest

from abi_marshal import (
    F64,
    I8,
    I16,
    I32,
    I64,
    MEMORY,
    PTR,
    ArrayType,
    CycleError,
    RecordType,
    SizeLimitError,
    classify,
    layout,
)

# Independent scalar facts, fixed by the ABI contract (not by the core).
SCALAR_SIZE_ALIGN = {
    "i8": (1, 1),
    "i16": (2, 2),
    "i32": (4, 4),
    "i64": (8, 8),
    "f64": (8, 8),
    "ptr": (8, 8),
}
SCALARS = {"i8": I8, "i16": I16, "i32": I32, "i64": I64, "f64": F64, "ptr": PTR}


def reference_record_layout(specs):
    """Standalone per-field offset reference.

    ``specs`` is a list of (size, align) pairs; returns
    (offsets, total_size, record_align) computed straight from the
    natural-alignment rule.  Empty records are ( [], 0, 1 ).
    """
    offsets = []
    offset = 0
    max_align = 1
    for size, align in specs:
        offset += -offset % align
        offsets.append(offset)
        offset += size
        if align > max_align:
            max_align = align
    if not specs:
        return offsets, 0, 1
    return offsets, offset + (-offset % max_align), max_align


class ScalarLayoutTest(unittest.TestCase):
    def test_scalar_sizes_and_aligns(self):
        for name, (size, align) in SCALAR_SIZE_ALIGN.items():
            lay = layout(SCALARS[name])
            self.assertEqual(lay.size, size, name)
            self.assertEqual(lay.align, align, name)

    def test_alignment_never_exceeds_8(self):
        for typ in SCALARS.values():
            self.assertLessEqual(layout(typ).align, 8)


class RecordOffsetReferenceTest(unittest.TestCase):
    """Every ordered pair/triple of scalars is checked against the
    independent reference, field by field."""

    def check_record(self, names):
        rec = RecordType("R_" + "_".join(names) or "R")
        for i, n in enumerate(names):
            rec.add_field(f"f{i}_{n}", SCALARS[n])
        lay = layout(rec)
        specs = [SCALAR_SIZE_ALIGN[n] for n in names]
        exp_offsets, exp_size, exp_align = reference_record_layout(specs)
        got_offsets = [f.offset for f in lay.fields]
        self.assertEqual(got_offsets, exp_offsets, names)
        self.assertEqual(lay.size, exp_size, names)
        self.assertEqual(lay.align, exp_align, names)

    def test_all_ordered_pairs(self):
        for a in SCALARS:
            for b in SCALARS:
                with self.subTest(pair=(a, b)):
                    self.check_record([a, b])

    def test_selected_triples(self):
        for names in (
            ["i8", "i64", "i16"],
            ["i32", "i8", "f64"],
            ["i16", "i8", "i32"],
            ["ptr", "i8", "i8"],
            ["i8", "i8", "i8"],
        ):
            with self.subTest(triple=names):
                self.check_record(names)


class HandComputedLayoutTest(unittest.TestCase):
    def offsets(self, rec):
        return {f.name: f.offset for f in layout(rec).fields}

    def test_i8_then_f64(self):
        rec = RecordType("A", [("a", I8), ("b", F64)])
        lay = layout(rec)
        self.assertEqual(self.offsets(rec), {"a": 0, "b": 8})
        self.assertEqual((lay.size, lay.align), (16, 8))

    def test_tail_padding_i32_i8(self):
        rec = RecordType("B", [("a", I32), ("b", I8)])
        lay = layout(rec)
        self.assertEqual(self.offsets(rec), {"a": 0, "b": 4})
        # end offset 5 rounded up to align 4 -> tail padding of 3
        self.assertEqual((lay.size, lay.align), (8, 4))

    def test_tail_padding_i16_i8(self):
        rec = RecordType("C", [("a", I16), ("b", I8)])
        lay = layout(rec)
        self.assertEqual(self.offsets(rec), {"a": 0, "b": 2})
        self.assertEqual((lay.size, lay.align), (4, 2))

    def test_no_tail_padding_when_already_aligned(self):
        rec = RecordType("D", [("a", I32), ("b", I32)])
        lay = layout(rec)
        self.assertEqual((lay.size, lay.align), (8, 4))

    def test_empty_record_convention(self):
        rec = RecordType("E")
        lay = layout(rec)
        self.assertEqual((lay.size, lay.align), (0, 1))
        self.assertEqual(lay.fields, ())

    def test_record_of_empty_records_is_empty(self):
        inner = RecordType("E")
        rec = RecordType("F", [("e1", inner), ("e2", inner)])
        lay = layout(rec)
        self.assertEqual((lay.size, lay.align), (0, 1))
        self.assertEqual([f.offset for f in lay.fields], [0, 0])

    def test_empty_record_field_does_not_shift_neighbors(self):
        empty = RecordType("E")
        rec = RecordType("G", [("a", I8), ("e", empty), ("b", I32)])
        lay = layout(rec)
        self.assertEqual(self.offsets(rec), {"a": 0, "e": 1, "b": 4})
        self.assertEqual((lay.size, lay.align), (8, 4))


class ArrayLayoutTest(unittest.TestCase):
    def test_array_stride_is_full_element_size(self):
        elem = RecordType("E", [("a", I8), ("b", I32)])  # size 8, align 4
        arr = ArrayType(elem, 3)
        lay = layout(arr)
        self.assertEqual((lay.size, lay.align), (24, 4))
        self.assertEqual(lay.count, 3)

    def test_nested_array_offsets(self):
        # i16[3] then i8: elements at 0,2,4; flag at 6; size 8 (align 2).
        rec = RecordType("N", [("vals", ArrayType(I16, 3)), ("flag", I8)])
        lay = layout(rec)
        offs = {f.name: f.offset for f in lay.fields}
        self.assertEqual(offs, {"vals": 0, "flag": 6})
        self.assertEqual((lay.size, lay.align), (8, 2))

    def test_array_of_records_field_offsets(self):
        elem = RecordType("E", [("a", I8), ("b", I32)])
        arr = ArrayType(elem, 2)
        outer = RecordType("O", [("items", arr), ("tail", I16)])
        lay = layout(outer)
        offs = {f.name: f.offset for f in lay.fields}
        # items: 2 * 8 = 16 bytes at 0; tail at 16; size 20 -> align 4 -> 20
        self.assertEqual(offs, {"items": 0, "tail": 16})
        self.assertEqual((lay.size, lay.align), (20, 4))

    def test_array_of_empty_records_has_size_zero(self):
        arr = ArrayType(RecordType("E"), 7)
        lay = layout(arr)
        self.assertEqual((lay.size, lay.align), (0, 1))

    def test_zero_length_array(self):
        arr = ArrayType(I64, 0)
        lay = layout(arr)
        self.assertEqual((lay.size, lay.align), (0, 8))


class ClassifyTest(unittest.TestCase):
    def test_mixed_class_aggregate(self):
        # chunk 0 holds the i8 -> INTEGER; chunk 1 holds the f64 -> FLOAT
        rec = RecordType("M", [("tag", I8), ("weight", F64)])
        self.assertEqual(classify(rec), ("integer", "float"))

    def test_float_then_int(self):
        rec = RecordType("M2", [("w", F64), ("tag", I64)])
        self.assertEqual(classify(rec), ("float", "integer"))

    def test_int_in_first_bytes_dominates_chunk(self):
        rec = RecordType("M3", [("a", I32), ("b", F64)])
        self.assertEqual(classify(rec), ("integer", "float"))

    def test_all_float_aggregate(self):
        rec = RecordType("P", [("x", F64), ("y", F64)])
        self.assertEqual(classify(rec), ("float", "float"))

    def test_over_16_bytes_is_memory(self):
        rec = RecordType("W", [("a", I64), ("b", I64), ("c", I64)])
        self.assertEqual(classify(rec), MEMORY)
        self.assertEqual(classify(ArrayType(F64, 3)), MEMORY)

    def test_exactly_16_bytes_stays_in_registers(self):
        rec = RecordType("S", [("a", I64), ("b", I64)])
        self.assertEqual(classify(rec), ("integer", "integer"))

    def test_empty_record_consumes_nothing(self):
        self.assertEqual(classify(RecordType("E")), ())
        self.assertEqual(classify(ArrayType(RecordType("E"), 4)), ())

    def test_scalar_classification(self):
        self.assertEqual(classify(I32), ("integer",))
        self.assertEqual(classify(F64), ("float",))
        self.assertEqual(classify(PTR), ("integer",))

    def test_nested_array_classification(self):
        inner = RecordType("E", [("a", I8), ("b", I32)])  # 8 bytes, INTEGER
        arr = ArrayType(inner, 2)  # 16 bytes
        self.assertEqual(classify(arr), ("integer", "integer"))
        mixed = RecordType("X", [("head", F64), ("items", ArrayType(I16, 3))])
        # chunk0: f64 -> FLOAT; chunk1: i16s -> INTEGER
        self.assertEqual(classify(mixed), ("float", "integer"))


class CycleAndLimitTest(unittest.TestCase):
    def test_direct_cycle_rejected(self):
        rec = RecordType("node")
        rec.add_field("self", rec)
        with self.assertRaises(CycleError):
            layout(rec)

    def test_indirect_cycle_rejected(self):
        a = RecordType("a")
        b = RecordType("b")
        a.add_field("b", b)
        b.add_field("a", a)
        with self.assertRaises(CycleError):
            layout(a)
        with self.assertRaises(CycleError):
            layout(b)

    def test_cycle_error_is_locatable(self):
        a = RecordType("a")
        b = RecordType("b")
        a.add_field("b", b)
        b.add_field("a", a)
        with self.assertRaises(CycleError) as ctx:
            layout(a)
        self.assertTrue(str(ctx.exception))

    def test_shared_subrecord_is_not_a_cycle(self):
        shared = RecordType("S", [("x", I32)])
        rec = RecordType("T", [("p", shared), ("q", shared)])
        lay = layout(rec)
        self.assertEqual([f.offset for f in lay.fields], [0, 4])

    def test_size_limit(self):
        # 24-byte element * 2**20 elements = 24 MiB > 16 MiB limit,
        # with the count itself still inside the array-count bound.
        elem = RecordType("W", [("a", I64), ("b", I64), ("c", I64)])
        big = ArrayType(elem, 1 << 20)
        with self.assertRaises(SizeLimitError):
            layout(big)


if __name__ == "__main__":
    unittest.main()
