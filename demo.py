"""demo.py — abi_marshal 固定输入演示（离线、确定性、约 1 秒内完成）。

演示内容：
1. 构造一组聚合类型并打印逐字段字节布局与 8 字节块分类；
2. pack_call 编组一次混合调用，打印寄存器与栈映像；
3. unpack_call 从映像重建参数并校验一致；
4. 实际触发一次越界拒绝（u8 赋值 300）。
"""

import abi_marshal as am


def show_layout(t):
    lay = am.layout(t)
    print(f"  {t.name}: size={lay.size} align={lay.align} "
          f"classify={am.classify(t)}")
    for f in lay.fields:
        print(f"    +{f.offset:<3} {f.name}: {f.type.name} "
              f"(size={f.layout.size}, align={f.layout.align})")


def main():
    print("== 1. 类型布局与分类 ==")
    empty = am.RecordType("Empty")
    empty.define([])
    header = am.RecordType("Header")
    header.define([("tag", am.UINT8), ("flags", am.UINT8), ("count", am.UINT32)])
    point = am.RecordType("Point")
    point.define([("x", am.DOUBLE), ("y", am.DOUBLE)])
    packet = am.RecordType("Packet")
    packet.define([("hdr", header),
                   ("pos", point),
                   ("samples", am.ArrayType(am.INT16, 4)),
                   ("reserved", empty)])
    mixed = am.RecordType("Mixed")
    mixed.define([("weight", am.DOUBLE), ("id", am.INT32)])
    for t in (empty, header, point, packet, mixed):
        show_layout(t)

    print("\n== 2. pack_call：寄存器与栈映像 ==")
    args = [
        (am.UINT64, 0x1122334455667788),          # -> 整数寄存器 0
        (am.DOUBLE, 2.5),                          # -> 浮点寄存器 0
        (mixed, {"weight": 0.5, "id": 42}),        # 混合类：FLOAT+INTEGER
        (packet, {"hdr": {"tag": 1, "flags": 2, "count": 4},
                  "pos": {"x": 1.25, "y": -0.5},
                  "samples": [10, -20, 30, -40],
                  "reserved": {}}),                # size 32 > 16 -> 栈
        (empty, {}),                               # 空记录：不占任何资源
        (am.INT64, -1),                            # -> 整数寄存器
    ]
    image = am.pack_call(args)
    for i, (t, _) in enumerate(args):
        p = image.placements[i]
        print(f"  arg[{i}] {t.name}: kind={p.kind} eightbytes={p.eightbytes}"
              f" int_regs={p.int_regs} float_regs={p.float_regs}"
              f" stack_offset={p.stack_offset} size={p.size}")
    print(f"  int_regs   = {[hex(v) if v is not None else None for v in image.int_regs]}")
    print(f"  float_regs = {list(image.float_regs)}")
    print(f"  stack ({len(image.stack)} bytes) = {image.stack.hex()}")

    print("\n== 3. unpack_call：从映像重建参数 ==")
    types = [t for t, _ in args]
    rebuilt = am.unpack_call(types, image)
    ok = rebuilt == [v for _, v in args]
    print(f"  rebuilt == original : {ok}")
    print(f"  rebuilt[3]['samples'] = {rebuilt[3]['samples']}")
    print(f"  rebuilt[3]['pos']     = {rebuilt[3]['pos']}")
    assert ok, "round-trip mismatch"

    print("\n== 4. 越界拒绝演示 ==")
    try:
        am.pack_call([(am.UINT8, 300)])
        print("  ERROR: 未触发拒绝！")
    except am.IntegerOverflowError as exc:
        print(f"  触发 IntegerOverflowError: {exc}")
    try:
        am.pack_call([(am.DOUBLE, float("nan"))])
        print("  ERROR: 未触发拒绝！")
    except am.NonFiniteFloatError as exc:
        print(f"  触发 NonFiniteFloatError: {exc}")

    print("\ndemo 完成。")


if __name__ == "__main__":
    main()
