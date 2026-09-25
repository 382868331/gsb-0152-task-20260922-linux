"""demo.py — abi_marshal 固定输入演示（离线、确定性、约 1 秒内完成）。

演示内容：
1. 构造一组聚合类型并打印逐字段字节布局与 8 字节块分类；
2. pack_call 编组一次混合调用，打印寄存器与栈映像；
3. unpack_call 从映像重建参数并校验一致；
4. 实际触发一次越界拒绝（u8 赋值 300）；
5. 返回值安放方案（返回寄存器 / 间接缓冲区 / 空记录）；
6. 完整调用流程：参数编码 -> 模拟计算 -> 返回编码 -> 调用方读取
   （混合整浮记录经返回寄存器）；
7. 间接返回：嵌套数组记录经有界缓冲区，隐藏指针把第 4 个整数参数
   刚好挤到栈上；
8. 返回值越界与非法缓冲区的实际拒绝。
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

    print("\n== 5. 返回值安放方案 ==")
    big = am.RecordType("Vec3")
    big.define([("x", am.DOUBLE), ("y", am.DOUBLE), ("z", am.DOUBLE)])
    nested = am.RecordType("Grid")
    nested.define([("m", am.ArrayType(am.ArrayType(am.INT16, 2), 2)),
                   ("x", am.DOUBLE), ("tag", am.UINT8)])
    for t in (am.UINT8, am.DOUBLE, mixed, empty, big, nested):
        p = am.plan_return(t)
        print(f"  {t.name}: kind={p.kind} eightbytes={p.eightbytes} "
              f"size={p.size} align={p.align}")

    print("\n== 6. 完整调用流程：返回寄存器（混合整浮记录）==")

    def weighted_avg(a, b):
        return {"weight": (a + b) / 2.0, "id": 7}

    call_args = [(am.DOUBLE, 1.0), (am.DOUBLE, 3.0)]
    call_img = am.pack_call(call_args, ret_type=mixed)
    values = am.unpack_call([t for t, _ in call_args], call_img)
    print(f"  被调方收到参数: {values}")
    ret_img = am.pack_return(mixed, weighted_avg(*values),
                             call_img.ret_plan)
    print(f"  返回整数寄存器: {ret_img.int_regs}")
    print(f"  返回浮点寄存器: {ret_img.float_regs}")
    result = am.unpack_return(mixed, ret_img, call_img.ret_plan)
    print(f"  调用方读到返回值: {result}")
    assert result == {"weight": 2.0, "id": 7}

    print("\n== 7. 间接返回：嵌套数组 + 隐藏指针挤占寄存器 ==")
    buf = am.ReturnBuffer(0x4000, 64)
    four_ints = [(am.INT64, 1), (am.INT64, 2), (am.INT64, 3), (am.INT64, 4)]
    img7 = am.pack_call(four_ints, ret_type=nested, ret_buffer=buf)
    print(f"  隐藏指针安放: kind={img7.hidden_ptr.kind} "
          f"int_regs={img7.hidden_ptr.int_regs}")
    print(f"  int_regs = {[hex(v) if v is not None else None for v in img7.int_regs]}")
    for i, p in enumerate(img7.placements):
        print(f"  arg[{i}] INT64: kind={p.kind} int_regs={p.int_regs} "
              f"stack_offset={p.stack_offset}")
    print(f"  stack = {img7.stack.hex()}  # 第 4 个参数被挤到栈上")

    def make_grid(a, b, c, d):
        s = a + b + c + d
        return {"m": [[s, 2 * s], [3 * s, 4 * s]], "x": s / 2.0, "tag": 0xAB}

    vals7 = am.unpack_call([am.INT64] * 4, img7)
    ret7 = am.pack_return(nested, make_grid(*vals7), img7.ret_plan)
    print(f"  写入缓冲区的 {len(ret7.memory)} 字节: {ret7.memory.hex()}")
    result7 = am.unpack_return(nested, ret7, img7.ret_plan)
    print(f"  调用方读到返回值: {result7}")
    assert result7 == {"m": [[10, 20], [30, 40]], "x": 5.0, "tag": 0xAB}
    # simulate_call 把上面整条链路合成一步：
    assert am.simulate_call(make_grid, four_ints, nested,
                            ret_buffer=buf) == result7
    print("  simulate_call 一步完成同一流程，结果一致: True")

    print("\n== 8. 返回值边界拒绝演示 ==")
    try:
        am.pack_return(am.UINT8, 300)
        print("  ERROR: 未触发拒绝！")
    except am.IntegerOverflowError as exc:
        print(f"  返回值越界 IntegerOverflowError: {exc}")
    try:
        am.pack_call([(am.UINT8, 999)], ret_type=big,
                     ret_buffer=am.ReturnBuffer(0x1004, 64))
        print("  ERROR: 未触发拒绝！")
    except am.InvalidValueError as exc:
        print(f"  缓冲区未对齐（先于参数编码拒绝）: {exc}")

    print("\ndemo 完成。")


if __name__ == "__main__":
    main()
