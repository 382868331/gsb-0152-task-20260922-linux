"""demo.py — abi_marshal 固定输入演示（离线、确定性、约 1 秒内完成）。

演示内容：
1. 构造一组聚合类型并打印逐字段字节布局与 8 字节块分类；
2. pack_call 编组一次混合调用，打印寄存器与栈映像；
3. unpack_call 从映像重建参数并校验一致；
4. 模拟返回值传递：小结果进寄存器、大结果走调用方缓冲区（隐式首参）；
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

    print("\n== 4. 返回值传递：寄存器与调用方缓冲区 ==")
    # 4a. 小结果：{f64, i32} 恰好 16 字节以内，逐块进结果寄存器
    small_ret = am.RecordType("Score")
    small_ret.define([("mean", am.DOUBLE), ("count", am.UINT32)])
    site = am.pack_callsite(small_ret, [(am.UINT64, 0x77), (am.DOUBLE, 0.5)])
    rp = site.result_placement
    print(f"  小结果安放: kind={rp.kind} int_regs={rp.int_regs} "
          f"float_regs={rp.float_regs} (显式实参位置不变)")
    returned = am.write_result(small_ret, site, {"mean": -1.75, "count": 42})
    print(f"  返回点 int_regs   = {list(returned.int_regs)}")
    print(f"  返回点 float_regs = {list(returned.float_regs)}")
    print(f"  调用方重建返回值  = {am.unpack_result(small_ret, returned)}")

    # 4b. 大结果：嵌套记录 + 数组（>16 字节），调用方提供缓冲区，
    #     缓冲区地址作为隐式指针占据第一个参数位置，显式实参整体后移。
    big_ret = am.RecordType("Report")
    big_ret.define([("head", header),
                    ("pos", point),
                    ("samples", am.ArrayType(am.INT16, 4))])
    big_val = {"head": {"tag": 9, "flags": 1, "count": 2},
               "pos": {"x": 10.5, "y": -3.25},
               "samples": [100, -200, 300, -400]}
    site2 = am.pack_callsite(big_ret, [(am.INT64, 1234)])
    rp2 = site2.result_placement
    print(f"  大结果安放: kind={rp2.kind} size={rp2.size} "
          f"align={rp2.align} buffer_addr=0x{rp2.address:x}")
    p_sret = site2.placements[0]
    p_arg = site2.placements[1]
    print(f"  placements[0] 隐式 sret: kind={p_sret.kind} "
          f"int_regs={p_sret.int_regs} sret={p_sret.sret}")
    print(f"  placements[1] 显式 i64 : kind={p_arg.kind} "
          f"int_regs={p_arg.int_regs} (无 sret 时本应在 0 号槽)")
    print(f"  进入时显式实参重建 = {am.unpack_call([am.INT64], site2)}")
    returned2 = am.write_result(big_ret, site2, big_val)
    print(f"  result_buffer ({len(returned2.result_buffer)} bytes) = "
          f"{returned2.result_buffer.hex()}")
    rebuilt_ret = am.unpack_result(big_ret, returned2)
    print(f"  调用方按同一套布局重建 = {rebuilt_ret}")
    assert rebuilt_ret == big_val

    # 4c. 空记录返回：不占寄存器也不占缓冲区
    site3 = am.pack_callsite(empty, [])
    out3 = am.write_result(empty, site3, {})
    print(f"  空记录返回: kind={out3.result_placement.kind} "
          f"value={am.unpack_result(empty, out3)}")

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
