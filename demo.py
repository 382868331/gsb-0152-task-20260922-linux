"""demo.py — abi_marshal 固定输入演示（离线、确定性、约 1 秒内完成）。

演示内容：
1. 构造一组聚合类型并打印逐字段字节布局与 8 字节块分类；
2. pack_call 编组一次混合调用，打印寄存器与栈映像；
3. unpack_call 从映像重建参数并校验一致；
4. 演示小返回值进寄存器、大返回值写入调用方空间且隐藏地址顺延参数；
5. 实际触发一次越界拒绝（u8 赋值 300）。
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

    print("\n== 4. pack_return / unpack_return：返回值编组 ==")
    small_ret = am.RecordType("SmallReturn")
    small_ret.define([("weight", am.DOUBLE), ("id", am.INT32)])
    small_value = {"weight": 1.75, "id": 99}
    small_image = am.pack_return(small_ret, small_value)
    print(f"  small {small_ret.name}: kind={small_image.kind} "
          f"eightbytes={small_image.eightbytes} "
          f"int_regs={small_image.int_regs} float_regs={small_image.float_regs}")
    print(f"  rebuilt small return = {am.unpack_return(small_ret, small_image)}")

    item_t = am.RecordType("ResultItem")
    item_t.define([("a", am.UINT8), ("b", am.UINT32)])
    big_ret = am.RecordType("BigResult")
    big_ret.define([("items", am.ArrayType(item_t, 3)),
                    ("count", am.UINT32)])
    big_value = {
        "items": [{"a": 1, "b": 100}, {"a": 2, "b": 200},
                  {"a": 3, "b": 300}],
        "count": 3,
    }
    result_buf = am.ResultBuffer(big_ret, 0x7F0000002000)
    call_image = am.pack_call([(am.INT64, 11), (am.DOUBLE, 0.5)],
                              return_type=big_ret, result_buffer=result_buf)
    print("  caller-side hidden result pointer: "
          f"placement={call_image.sret_pointer} address="
          f"0x{call_image.result_address:x}")
    print("  visible args after hidden pointer: "
          f"{[(p.kind, p.int_regs, p.float_regs) for p in call_image.placements]}")
    big_image = am.pack_return(big_ret, big_value, result_buf)
    rebuilt_return = am.unpack_return(big_ret, big_image, result_buf)
    print(f"  big {big_ret.name}: kind={big_image.kind} size={big_image.size} "
          f"int_regs[0]=0x{big_image.int_regs[0]:x}")
    print(f"  result buffer raw bytes = {result_buf.contents.hex()}")
    print(f"  rebuilt big return = {rebuilt_return}")
    assert rebuilt_return == big_value

    print("\n== 5. 越界拒绝演示 ==")
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
