"""Input validation and error-semantics tests."""

import unittest

from abi_marshal import (
    F64,
    I8,
    I32,
    I64,
    PTR,
    U8,
    U64,
    ArityError,
    ArrayType,
    CycleError,
    IntegerOverflowError,
    IntType,
    NonFiniteFloatError,
    RecordType,
    TypeDefinitionError,
    ValueTypeError,
    pack_call,
)


class IntegerOverflowTest(unittest.TestCase):
    def test_signed_boundaries(self):
        # in-range extremes pass
        pack_call([I8], [127])
        pack_call([I8], [-128])
        # one past each end is rejected
        with self.assertRaises(IntegerOverflowError):
            pack_call([I8], [128])
        with self.assertRaises(IntegerOverflowError):
            pack_call([I8], [-129])

    def test_unsigned_boundaries(self):
        pack_call([U8], [0])
        pack_call([U8], [255])
        with self.assertRaises(IntegerOverflowError):
            pack_call([U8], [256])
        with self.assertRaises(IntegerOverflowError):
            pack_call([U8], [-1])

    def test_u64_and_pointer_boundaries(self):
        pack_call([U64], [(1 << 64) - 1])
        with self.assertRaises(IntegerOverflowError):
            pack_call([U64], [1 << 64])
        pack_call([PTR], [(1 << 64) - 1])
        with self.assertRaises(IntegerOverflowError):
            pack_call([PTR], [1 << 64])
        with self.assertRaises(IntegerOverflowError):
            pack_call([PTR], [-1])

    def test_overflow_inside_nested_value_is_locatable(self):
        rec = RecordType("R", [("x", I8)])
        arr = ArrayType(rec, 2)
        with self.assertRaises(IntegerOverflowError) as ctx:
            pack_call([arr], [[{"x": 1}, {"x": 200}]])
        self.assertEqual(ctx.exception.path, ("arg0", 1, "x"))
        self.assertIn("arg0[1].x", str(ctx.exception))


class BoolRejectionTest(unittest.TestCase):
    def test_bool_rejected_for_int(self):
        with self.assertRaises(ValueTypeError):
            pack_call([I32], [True])

    def test_bool_rejected_for_pointer(self):
        with self.assertRaises(ValueTypeError):
            pack_call([PTR], [False])

    def test_bool_rejected_for_float(self):
        with self.assertRaises(ValueTypeError):
            pack_call([F64], [True])

    def test_bool_rejected_as_array_count(self):
        with self.assertRaises(TypeDefinitionError):
            ArrayType(I8, True)


class FloatValidationTest(unittest.TestCase):
    def test_nan_rejected(self):
        with self.assertRaises(NonFiniteFloatError):
            pack_call([F64], [float("nan")])

    def test_infinity_rejected(self):
        for v in (float("inf"), float("-inf")):
            with self.assertRaises(NonFiniteFloatError):
                pack_call([F64], [v])

    def test_non_numeric_rejected(self):
        with self.assertRaises(ValueTypeError):
            pack_call([F64], ["1.0"])

    def test_int_accepted_only_if_exactly_representable(self):
        pack_call([F64], [2**53])  # exact
        with self.assertRaises(IntegerOverflowError):
            pack_call([F64], [2**53 + 1])  # would round silently
        with self.assertRaises(IntegerOverflowError):
            pack_call([F64], [10**400])  # overflows conversion


class ArityAndShapeTest(unittest.TestCase):
    def test_top_level_arity(self):
        with self.assertRaises(ArityError):
            pack_call([I8, I8], [1])
        with self.assertRaises(ArityError):
            pack_call([I8], [1, 2])

    def test_array_length_must_match_exactly(self):
        arr = ArrayType(I8, 2)
        pack_call([arr], [[1, 2]])
        with self.assertRaises(ArityError):
            pack_call([arr], [[1]])
        with self.assertRaises(ArityError):
            pack_call([arr], [[1, 2, 3]])

    def test_record_missing_and_extra_fields(self):
        rec = RecordType("R", [("x", I8), ("y", I8)])
        with self.assertRaises(ValueTypeError):
            pack_call([rec], [{"x": 1}])
        with self.assertRaises(ValueTypeError):
            pack_call([rec], [{"x": 1, "y": 2, "z": 3}])

    def test_wrong_container_kinds(self):
        rec = RecordType("R", [("x", I8)])
        with self.assertRaises(ValueTypeError):
            pack_call([rec], [("not-a-mapping",)])
        with self.assertRaises(ValueTypeError):
            pack_call([ArrayType(I8, 1)], [b"\x01"])  # bytes are not a list
        with self.assertRaises(ValueTypeError):
            pack_call([I8], [1.5])

    def test_values_must_be_a_sequence(self):
        with self.assertRaises(ValueTypeError):
            pack_call([I8], 7)


class TypeDefinitionTest(unittest.TestCase):
    def test_int_size_domain(self):
        for bad in (0, 3, 5, 16, -1, "4"):
            with self.assertRaises(TypeDefinitionError):
                IntType("bad", bad, True)

    def test_array_count_domain(self):
        for bad in (-1, 2**20 + 1, 1.5, "3"):
            with self.assertRaises(TypeDefinitionError):
                ArrayType(I8, bad)
        ArrayType(I8, 0)
        ArrayType(I8, 2**20)

    def test_duplicate_field_rejected(self):
        rec = RecordType("R", [("x", I8)])
        with self.assertRaises(TypeDefinitionError):
            rec.add_field("x", I8)

    def test_non_type_field_rejected(self):
        with self.assertRaises(TypeDefinitionError):
            RecordType("R", [("x", 42)])

    def test_cycle_rejected_through_pack_call(self):
        rec = RecordType("node")
        rec.add_field("self", rec)
        with self.assertRaises(CycleError):
            pack_call([rec], [{}])


class AtomicityTest(unittest.TestCase):
    def test_failed_pack_returns_no_partial_image(self):
        # A failure deep in a later argument must raise, not return a
        # partially-filled image; pack_call is pure, so we simply assert
        # the exception escapes and nothing is produced.
        rec = RecordType("R", [("x", I8)])
        with self.assertRaises(IntegerOverflowError):
            pack_call([I64, I64, rec], [1, 2, {"x": 1000}])


if __name__ == "__main__":
    unittest.main()
