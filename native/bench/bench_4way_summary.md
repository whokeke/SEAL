# 4-way Mulmod Benchmark: GCC 15 vs LLVM 22

## 测试环境

- **CPU**: HiSilicon HIP12, 2.3 GHz, 256-bit SVE (svcntd=4), SVE2
- **GCC**: spec-study gcc-15.1.0 (`/home/hukeke/spec/toolchains/gcc-15`)
- **LLVM**: clang 22.1.8 (`/home/hukeke/spec/toolchains/LLVM-22.1.8-release`)
- **编译参数**: `-O3 -march=armv9-a+sve2 -D_GNU_SOURCE`
- **数据**: N=65536, ITERS=1000, p=0xFFFFFFFFFFFFFFC5

## 四种变体

| # | 变体 | mulhi 实现 | 关键指令 |
|---|------|-----------|---------|
| 1 | scalar schoolbook | 32-bit 分解 (16 条指令) | `and`+`lsr`+4×`mul`+carry chain |
| 2 | scalar `__uint128_t` | 编译器生成 `umulh` | `umulh`+`mul`+`msub` (3 条) |
| 3 | vector schoolbook (SVE32) | `svmullb`/`svmullt` + carry chain | 4×`umullb/t`+13 条 carry (19 条/vector) |
| 4 | vector int128 (`svmulh`) | SVE 原生 `umulh` | `umulh z.d`+`mls`+`cmphs`+`sub` (5 条/vector) |

## Benchmark 结果

| 变体 | GCC 15 (cyc/elem) | LLVM 22 (cyc/elem) | 差距 |
|------|-------------------|-------------------|------|
| scalar schoolbook | 3.579 | 3.446 | LLVM 快 4% |
| scalar `__uint128_t` | 2.699 | **2.252** | **LLVM 快 17%** |
| vector schoolbook (SVE32) | 3.202 | 3.204 | 持平 |
| vector int128 (`svmulh`) | 1.205 | 1.191 | 持平 |

## 汇编分析

### (1) scalar schoolbook — 都没用 umulh

**GCC 15**: 自动向量化了标量循环。检测到循环可并行，用 SVE `mul z.d` + `umulh z.d` 向量化 schoolbook。461 条指令（含 vector+scalar 两条路径），代码膨胀。

**LLVM 22**: 保持标量。用 `umull`（标量 32×32→64 宽化乘）做 4 个 partial product。74 条指令，紧凑。

都没用 `umulh` 的原因：schoolbook 源码写的是 32 位分解（`a_lo = a & 0xFFFFFFFF`、`a_hi = a >> 32`、`p_ll = a_lo * b_lo`...），编译器看到的不是 `a*b >> 64` 而是拆分后的运算，无法识别为 umulh。

### (2) scalar `__uint128_t` — 都生成 umulh，但向量化策略不同

**GCC 15**:
- 标量核心 = `umulh x5, x7, x3` + `msub x5, x4, x5, x7`（3 条）
- 也自动向量化：`ptrue p7.b, all` = **全宽 4 元素/vector**
- 391 条指令（含 vector+scalar 路径）

**LLVM 22**:
- 自动向量化：`ptrue p0.d, vl2` = **窄 2 元素/vector**（128-bit SVE）
- 46 条指令，极紧凑

**差距原因 (17%)**: GCC 用全宽 4 元素向量化，寄存器压力大、循环开销大；LLVM 用窄 2 元素向量化，代码更紧凑、寄存器压力更小 → 更快。

### (3) vector schoolbook (SVE32) — 都用 umullb/umullt

两个编译器都正确识别 `svmullb_u64`/`svmullt_u64` ACLE intrinsic，生成 `umullb z.d, z.s, z.s` 指令。性能完全相同（3.20），因为 intrinsic 几乎不留给编译器优化空间。

### (4) vector int128 (svmulh) — 都用 SVE umulh

两个编译器都生成 `umulh z.d, p/m, z.d, z.d` + `mls z.d` + `cmphs p.d` + `sub z.d`。性能基本相同（1.21 vs 1.19），这是 intrinsic 的理想情况——编译器差异最小。

## 性能差距分析

### 排名（快→慢）

```
1. vector int128 (svmulh)    1.2 cyc/elem   ← 1 条 SVE umulh / N=4 元素 = 0.3 条/元素
2. scalar int128 (umulh)     2.3 cyc/elem   ← 1 条标量 umulh / 1 元素，但编译器向量化后 ~2 元素/vector
3. vector schoolbook (SVE32) 3.2 cyc/elem   ← 19 条 SVE 指令 / N=4 = 4.75 条/元素
4. scalar schoolbook         3.5 cyc/elem   ← 16 条标量指令 / 1 元素
```

### 为什么 vector schoolbook 比 scalar int128 慢？

- SVE32 schoolbook：19 条指令 / 4 元素 = **4.75 条/元素**
- scalar int128 (向量化后)：约 5-6 条指令 / 2 元素 = **2.5-3 条/元素**
- schoolbook 的 carry chain（13 条依赖链）是瓶颈，N=4 不够大无法摊薄

### 为什么 GCC 和 LLVM 在 scalar int128 上差距 17%？

| | GCC 15 | LLVM 22 |
|---|---|---|
| 向量化宽度 | 4 元素 (ptrue p7.b, all) | 2 元素 (ptrue p0.d, vl2) |
| 指令数 | 391 | 46 |
| 寄存器压力 | 高 (4 元素需要更多 Z 寄存器) | 低 (2 元素只需 2-3 个 Z 寄存器) |
| 循环开销 | 大 (4 元素一组，tail 处理复杂) | 小 (2 元素一组，简单) |

LLVM 的窄向量化（vl2）是更优的策略：2 元素刚好用 128-bit，寄存器压力低，代码紧凑，循环开销小。

## 结论

1. **有 `svmulh` 就别用 schoolbook**——无论标量还是向量，schoolbook 都比 umulh 慢 2-3x
2. **`__uint128_t` 是标量路径的最优修复**——让编译器生成 `umulh`，比 schoolbook 快 30-50%
3. **SVE intrinsic 消除编译器差异**——vector int128 和 vector schoolbook 的 GCC/LLVM 性能基本相同
4. **LLVM 22 的自动向量化更优**——在 scalar `__uint128_t` 上比 GCC 15 快 17%，因为选择了更窄的向量化宽度（vl2 vs full VL）
5. **schoolbook 的 carry chain 是根本瓶颈**——19 条指令中 13 条是串行依赖链，interleave 和向量化都无法有效摊薄（N=4 太小）

## 文件

| 文件 | 说明 |
|------|------|
| `bench_4way.c` | 4 变体 benchmark 源码 |
| `bench_4way_gcc15.s` | GCC 15 生成的汇编 |
| `bench_4way_llvm22.s` | LLVM 22 生成的汇编 |

### 编译命令

```bash
# GCC 15
/home/hukeke/spec/toolchains/gcc-15/bin/aarch64-unknown-linux-gnu-gcc \
  -O3 -march=armv9-a+sve2 -D_GNU_SOURCE -o bench_4way_gcc15 bench_4way.c

# LLVM 22
/home/hukeke/spec/toolchains/LLVM-22.1.8-release/bin/clang \
  -O3 -march=armv9-a+sve2 -D_GNU_SOURCE \
  --gcc-toolchain=/home/hukeke/spec/toolchains/gcc-15 \
  -o bench_4way_llvm22 bench_4way.c
```
