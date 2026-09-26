"""位域（BitfieldType）的单元测试。

独立性说明：本文件的期望字节由 set_bits/get_bits 按题内规则
（小端、低位先分配）逐位独立组装/取出，不调用 abi_marshal 的布局或
编解码实现；往返一致仅作为补充。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abi_marshal as am


def set_bits(size, writes):
    """writes: [(byte_off, bit_off, width, value)] -> bytes（逐位独立组装）。"""
    bits = [0] * (size * 8)
    for byte_off, bit_off, width, value in writes:
        assert 0 <= value < (1 << width)
        for i in range(width):
            bits[byte_off * 8 + bit_off + i] = (value >> i) & 1
    out = bytearray(size)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def get_bits(data, byte_off, bit_off, width):
    """从字节串中按小端位序独立取出 width 位的无符号值。"""
    value = 0
    for i in range(width):
        pos = byte_off * 8 + bit_off + i
        bit = (data[pos // 8] >> (pos % 8)) & 1
        value |= bit << i
    return value


def bf(base, width):
    return am.BitfieldType(base, width)


def rec(name, fields):
    r = am.RecordType(name)
    r.define(fields)
    return r


class TestBitfieldTypeValidation(unittest.TestCase):
    def test_signed_base_rejected(self):
        with self.assertRaises(am.InvalidTypeError):
            am.BitfieldType(am.INT8, 3)

    def test_64bit_unit_rejected(self):
        with self.assertRaises(am.InvalidTypeError):
            am.BitfieldType(am.UINT64, 3)

    def test_width_out_of_range_rejected(self):
        with self.assertRaises(am.InvalidTypeError):
            am.BitfieldType(am.UINT8, 9)
        with self.assertRaises(am.InvalidTypeError):
            am.BitfieldType(am.UINT8, -1)
        with self.assertRaises(am.InvalidTypeError):
            am.BitfieldType(am.UINT8, True)

    def test_full_width_ok(self):
        t = am.BitfieldType(am.UINT32, 32)
        self.assertEqual(t.range, (0, (1 << 32) - 1))


class TestBitfieldLayout(unittest.TestCase):
    def test_cross_unit_when_not_fitting(self):
        # 5 + 6 > 8：第二个字段必须开新单元，不能横跨
        r = rec("Cross", [("a", bf(am.UINT8, 5)), ("b", bf(am.UINT8, 6))])
        lay = am.layout(r)
        self.assertEqual(lay.size, 2)
        self.assertEqual(lay.align, 1)
        fa, fb = lay.fields
        self.assertEqual((fa.offset, fa.bit_offset, fa.mask), (0, 0, 0b11111))
        self.assertEqual((fb.offset, fb.bit_offset, fb.mask), (1, 0, 0b111111))

    def test_lsb_first_packing_within_unit(self):
        r = rec("Pack", [("a", bf(am.UINT8, 3)), ("b", bf(am.UINT8, 4))])
        lay = am.layout(r)
        self.assertEqual(lay.size, 1)
        fa, fb = lay.fields
        self.assertEqual((fa.offset, fa.bit_offset, fa.mask), (0, 0, 0b0000111))
        self.assertEqual((fb.offset, fb.bit_offset, fb.mask), (0, 3, 0b1111000))

    def test_zero_width_ends_unit_and_realigns(self):
        r = rec("ZW", [("a", bf(am.UINT8, 3)), ("z", bf(am.UINT8, 0)),
                       ("b", bf(am.UINT16, 4))])
        lay = am.layout(r)
        # a 占字节 0；零宽结束单元；b 重新对齐到 2 字节边界（偏移 2）
        self.assertEqual(lay.size, 4)
        self.assertEqual(lay.align, 2)
        fa, fz, fb = lay.fields
        self.assertEqual((fa.offset, fa.bit_offset), (0, 0))
        self.assertIsNone(fz.bit_offset)
        self.assertEqual((fb.offset, fb.bit_offset, fb.mask), (2, 0, 0b1111))

    def test_zero_width_same_unit_realigns_to_same_size(self):
        r = rec("ZW2", [("a", bf(am.UINT8, 3)), ("z", bf(am.UINT8, 0)),
                        ("b", bf(am.UINT8, 2))])
        lay = am.layout(r)
        self.assertEqual(lay.size, 2)
        self.assertEqual(lay.fields[2].offset, 1)

    def test_mixed_unit_sizes(self):
        r = rec("Mix", [("a", bf(am.UINT8, 8)), ("b", bf(am.UINT32, 20))])
        lay = am.layout(r)
        self.assertEqual(lay.size, 8)
        self.assertEqual(lay.align, 4)
        self.assertEqual(lay.fields[1].offset, 4)

    def test_nested_record_with_bitfields(self):
        inner = rec("Inner", [("x", bf(am.UINT16, 10)), ("y", bf(am.UINT16, 6))])
        outer = rec("Outer", [("tag", am.UINT8), ("in", inner),
                              ("tail", bf(am.UINT8, 2))])
        lay = am.layout(outer)
        self.assertEqual(lay.size, 6)
        self.assertEqual(lay.align, 2)
        ftag, fin, ftail = lay.fields
        self.assertEqual(ftag.offset, 0)
        self.assertEqual(fin.offset, 2)
        self.assertEqual(ftail.offset, 4)
        inner_lay = am.layout(inner)
        self.assertEqual(inner_lay.size, 2)
        self.assertEqual(inner_lay.fields[1].bit_offset, 10)

    def test_non_bitfield_layout_unchanged(self):
        # 不含位域的旧类型布局必须与旧实现一致（硬编码期望）
        r = rec("Old", [("a", am.UINT8), ("b", am.UINT32), ("c", am.UINT8)])
        lay = am.layout(r)
        self.assertEqual(lay.size, 12)
        self.assertEqual(lay.align, 4)
        self.assertEqual([f.offset for f in lay.fields], [0, 4, 8])
        for f in lay.fields:
            self.assertIsNone(f.bit_offset)
            self.assertIsNone(f.mask)
        empty = rec("Empty", [])
        self.assertEqual((am.layout(empty).size, am.layout(empty).align), (0, 1))


class TestBitfieldEncodeDecode(unittest.TestCase):
    def test_independent_byte_assembly(self):
        r = rec("Enc", [("a", bf(am.UINT8, 3)), ("z", bf(am.UINT8, 0)),
                        ("b", bf(am.UINT8, 5)), ("c", bf(am.UINT16, 9)),
                        ("d", am.UINT32)])
        lay = am.layout(r)
        self.assertEqual(lay.size, 8)
        value = {"a": 5, "b": 17, "c": 300, "d": 0xDEADBEEF}
        data = am.encode(r, value)
        # 独立逐位组装期望字节
        expected = set_bits(8, [(0, 0, 3, 5), (1, 0, 5, 17), (2, 0, 9, 300)])
        expected = expected[:4] + (0xDEADBEEF).to_bytes(4, "little")
        self.assertEqual(data, expected)
        # 独立反向取出核对
        self.assertEqual(get_bits(data, 0, 0, 3), 5)
        self.assertEqual(get_bits(data, 1, 0, 5), 17)
        self.assertEqual(get_bits(data, 2, 0, 9), 300)
        self.assertEqual(am.decode(r, data), value)

    def test_unused_bits_and_padding_are_zero(self):
        r = rec("Pad", [("a", bf(am.UINT8, 3)), ("b", am.UINT32)])
        lay = am.layout(r)
        self.assertEqual(lay.size, 8)
        data = am.encode(r, {"a": 0b111, "b": 1})
        self.assertEqual(data, bytes([0b00000111, 0, 0, 0, 1, 0, 0, 0]))

    def test_adjacent_fields_do_not_clobber(self):
        r = rec("Adj", [("a", bf(am.UINT8, 4)), ("b", bf(am.UINT8, 4))])
        data = am.encode(r, {"a": 0b1010, "b": 0b0101})
        self.assertEqual(data, bytes([0b01011010]))
        self.assertEqual(am.decode(r, data), {"a": 0b1010, "b": 0b0101})

    def test_max_values_roundtrip(self):
        r = rec("Max", [("a", bf(am.UINT16, 16)), ("b", bf(am.UINT32, 32))])
        value = {"a": 0xFFFF, "b": 0xFFFFFFFF}
        self.assertEqual(am.decode(r, am.encode(r, value)), value)

    def test_overflow_rejected_wholesale(self):
        r = rec("Ov", [("a", bf(am.UINT8, 3))])
        with self.assertRaises(am.IntegerOverflowError):
            am.encode(r, {"a": 8})
        # pack_call 整次拒绝：即使前一个参数合法也没有部分结果
        with self.assertRaises(am.IntegerOverflowError) as ctx:
            am.pack_call([(am.UINT8, 1), (r, {"a": 8})])
        self.assertEqual(ctx.exception.path, "arg[1].a")

    def test_bool_and_type_mismatch_rejected(self):
        r = rec("Bool", [("a", bf(am.UINT8, 3))])
        with self.assertRaises(am.InvalidValueError):
            am.encode(r, {"a": True})
        with self.assertRaises(am.InvalidValueError):
            am.encode(r, {"a": 1.0})

    def test_zero_width_field_not_in_value_dict(self):
        r = rec("ZV", [("a", bf(am.UINT8, 3)), ("z", bf(am.UINT8, 0)),
                       ("b", bf(am.UINT8, 2))])
        data = am.encode(r, {"a": 1, "b": 2})
        self.assertEqual(am.decode(r, data), {"a": 1, "b": 2})
        # 提供零宽字段的值属于多余字段，拒绝
        with self.assertRaises(am.InvalidValueError):
            am.encode(r, {"a": 1, "z": 0, "b": 2})


class TestBitfieldClassificationAndMarshalling(unittest.TestCase):
    def test_bitfield_bytes_classify_as_integer(self):
        r = rec("Cls", [("a", bf(am.UINT8, 3)), ("f", am.DOUBLE)])
        lay = am.layout(r)
        self.assertEqual(lay.size, 16)
        self.assertEqual(am.classify(r), (am.INTEGER, am.FLOAT))

    def test_exactly_16_bytes_goes_to_registers(self):
        # 恰好 16 字节：寄存器/栈分界，应进寄存器
        fields = [("b%d" % i, bf(am.UINT32, 32)) for i in range(4)]
        r = rec("Edge16", fields)
        lay = am.layout(r)
        self.assertEqual(lay.size, 16)
        value = {"b%d" % i: 0x11111111 * (i + 1) for i in range(4)}
        img = am.pack_call([(r, value)])
        self.assertEqual(img.placements[0].kind, "registers")
        self.assertEqual(img.placements[0].int_regs, (0, 1))
        self.assertEqual(img.stack, b"")
        self.assertEqual(am.unpack_call([r], img), [value])

    def test_17_bytes_goes_to_stack(self):
        # 恰超 16 字节：整体走栈
        fields = [("b%d" % i, bf(am.UINT32, 32)) for i in range(4)]
        fields.append(("extra", bf(am.UINT8, 1)))
        r = rec("Edge17", fields)
        lay = am.layout(r)
        self.assertEqual(lay.size, 20)
        self.assertEqual(am.classify(r), (am.MEMORY,))
        value = {"b%d" % i: i for i in range(4)}
        value["extra"] = 1
        img = am.pack_call([(r, value)])
        self.assertEqual(img.placements[0].kind, "stack")
        self.assertEqual(img.placements[0].stack_offset, 0)
        self.assertEqual(len(img.stack), 32)  # 16 字节对齐
        self.assertEqual(am.unpack_call([r], img), [value])

    def test_register_pressure_falls_back_to_stack(self):
        r = rec("FB", [(n, bf(am.UINT32, 32)) for n in "xyzw"])
        args = [(am.UINT64, 1), (am.UINT64, 2), (am.UINT64, 3),
                (r, {"x": 0xAA, "y": 0xBB, "z": 0xCC, "w": 0xDD})]
        img = am.pack_call(args)
        # 3 个整数寄存器已用，16 字节记录需 2 个 -> 整体走栈且不消耗寄存器
        self.assertEqual(img.placements[3].kind, "stack")
        self.assertIsNone(img.int_regs[3])
        types = [t for t, _ in args]
        self.assertEqual(am.unpack_call(types, img),
                         [1, 2, 3, {"x": 0xAA, "y": 0xBB,
                                    "z": 0xCC, "w": 0xDD}])

    def test_mixed_call_roundtrip(self):
        inner = rec("In", [("f", bf(am.UINT16, 12))])
        r = rec("Mix2", [("a", bf(am.UINT8, 1)), ("n", inner),
                         ("g", bf(am.UINT8, 7))])
        args = [(r, {"a": 1, "n": {"f": 0xABC}, "g": 0x55}),
                (am.DOUBLE, 2.5)]
        img = am.pack_call(args)
        types = [t for t, _ in args]
        self.assertEqual(am.unpack_call(types, img),
                         [{"a": 1, "n": {"f": 0xABC}, "g": 0x55}, 2.5])


if __name__ == "__main__":
    unittest.main()
