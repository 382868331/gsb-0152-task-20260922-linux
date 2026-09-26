"""位域迭代演示：正常编组 + 边界拒绝，全部现场真实计算（远小于 90 秒）。

运行：python demo_iteration.py
"""

import abi_marshal as am


def bf(base, width):
    return am.BitfieldType(base, width)


def show_layout(rec):
    lay = am.layout(rec)
    print(f"记录 {rec.name}: size={lay.size} align={lay.align}")
    for f in lay.fields:
        if f.bit_offset is None:
            print(f"  {f.name}: byte={f.offset} (非位域或零宽)")
        else:
            print(f"  {f.name}: byte={f.offset} bit={f.bit_offset} "
                  f"mask={f.mask:#x}")


def main():
    # --- 正常情形：跨单元 + 零宽分隔 + 嵌套，编组后重建 ---
    inner = am.RecordType("Flags")
    inner.define([("ready", bf(am.UINT8, 1)), ("mode", bf(am.UINT8, 3))])
    pkt = am.RecordType("Packet")
    pkt.define([
        ("ver", bf(am.UINT8, 4)), ("type", bf(am.UINT8, 4)),
        ("_sep", bf(am.UINT8, 0)),                 # 零宽：结束当前单元
        ("len", bf(am.UINT16, 9)),                 # 跨入新单元
        ("flags", inner),                          # 嵌套记录
        ("crc", am.UINT32),
    ])
    show_layout(pkt)
    value = {"ver": 3, "type": 5, "len": 400,
             "flags": {"ready": 1, "mode": 6}, "crc": 0xDEADBEEF}
    data = am.encode(pkt, value)
    print("编码字节:", data.hex())
    back = am.decode(pkt, data)
    print("解码重建:", back)
    assert back == value

    img = am.pack_call([(pkt, value), (am.DOUBLE, 2.5)])
    print("分类:", am.classify(pkt), "安放:", img.placements[0].kind)
    rebuilt = am.unpack_call([pkt, am.DOUBLE], img)
    assert rebuilt == [value, 2.5]
    print("pack/unpack 往返一致")

    # --- 边界 1：恰好 16 字节 -> 寄存器；再加 1 位域单元 -> 整体走栈 ---
    edge16 = am.RecordType("Edge16")
    edge16.define([("b%d" % i, bf(am.UINT32, 32)) for i in range(4)])
    v16 = {"b%d" % i: 0x11111111 * (i + 1) for i in range(4)}
    img16 = am.pack_call([(edge16, v16)])
    print("16 字节记录安放:", img16.placements[0].kind,
          "int_regs:", img16.placements[0].int_regs)

    edge20 = am.RecordType("Edge20")
    edge20.define([("b%d" % i, bf(am.UINT32, 32)) for i in range(4)]
                  + [("extra", bf(am.UINT8, 1))])
    v20 = dict(v16, extra=1)
    img20 = am.pack_call([(edge20, v20)])
    print("20 字节记录安放:", img20.placements[0].kind,
          "stack_offset:", img20.placements[0].stack_offset)

    # --- 边界 2：字段值超范围，整次拒绝 ---
    try:
        am.pack_call([(am.UINT8, 1), (pkt, dict(value, len=512))])  # 9 位最大 511
    except am.IntegerOverflowError as exc:
        print("越界拒绝:", exc)


if __name__ == "__main__":
    main()

