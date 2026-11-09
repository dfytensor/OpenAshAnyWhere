# softmax_v1.py
import torch, triton, triton.language as tl
from triton.testing import do_bench, assert_close

@triton.jit
def softmax_v1_kernel(X, Y, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)          # 数值稳定
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(Y + row * stride + cols, num / den, mask=mask)


def softmax_v1(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    softmax_v1_kernel[(n_rows,)](x, y, x.stride(0), n_cols, BLOCK_SIZE=BLOCK)
    return y





@triton.jit
def softmax_kernel(X, Y, stride, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(Y + row * stride + cols, (num / den).to(Y.dtype.element_ty), mask=mask)


def softmax_opt_a(x, num_warps=None, BLOCK_SIZE=None):
    n_rows, N = x.shape
    BLOCK_SIZE = BLOCK_SIZE or triton.next_power_of_2(N)
    if num_warps is None:
        num_warps = max(1, min(BLOCK_SIZE // 256, 16))   # 经验公式 + 上限
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, y, x.stride(0), N,
                              BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return y







def autotune_num_warps(x, candidates=(1, 2, 4, 8, 16, 32), warmup=10, rep=50):
    n_rows, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)
    best = None
    results = []
    for nw in candidates:
        # 先确认不会超过硬件线程上限（num_warps * 32 ≤ 1024）
        if nw * 32 > 1024:
            continue
        try:
            ms = do_bench(
                lambda nw=nw: softmax_kernel[(n_rows,)](
                    x, y, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE, num_warps=nw
                ),
                warmup=warmup, rep=rep,
            )
        except Exception:
            continue   # OutOfResources 等直接跳过
        results.append((nw, ms))
        if best is None or ms < best[1]:
            best = (nw, ms)
    return best, results

# opt_b_row_packing.py
@triton.jit
def softmax_row_packed_kernel(
    X, Y, stride, n_rows, N,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)   # (R,)
    cols = tl.arange(0, BLOCK_N)                                  # (N,)
    row_mask = rows < n_rows
    mask = row_mask[:, None] & (cols[None, :] < N)                # (R, N)

    ptrs = X + rows[:, None] * stride + cols[None, :]
    x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

    x = x - tl.max(x, axis=1, keep_dims=True)                     # 行内 max
    num = tl.exp(x)
    den = tl.sum(num, axis=1, keep_dims=True)                     # 行内 sum
    out = (num / den).to(Y.dtype.element_ty)

    tl.store(Y + rows[:, None] * stride + cols[None, :], out, mask=mask)


def softmax_opt_b(x, ROWS_PER_BLOCK=4, BLOCK_N=None, num_warps=None):
    n_rows, N = x.shape
    BLOCK_N = BLOCK_N or triton.next_power_of_2(N)
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
    if num_warps is None:
        num_warps = max(1, min(BLOCK_N // 256, 16))
    y = torch.empty_like(x)
    softmax_row_packed_kernel[grid](
        x, y, x.stride(0), n_rows, N,
        ROWS_PER_BLOCK=ROWS_PER_BLOCK, BLOCK_N=BLOCK_N, num_warps=num_warps,
    )
    return y



def compare_row_packing(x, R_list=(1, 2, 4, 8, 16)):
    n_rows, N = x.shape
    y_ref = torch.softmax(x, dim=-1)
    print(f"{'R':>3} {'grid':>8} {'ms':>10} {'GB/s':>10} {'OK':>4}")
    for R in R_list:
        y = softmax_opt_b(x, ROWS_PER_BLOCK=R)
        ok = torch.allclose(y, y_ref, atol=1e-3, rtol=1e-3)
        ms = do_bench(lambda R=R: softmax_opt_b(x, ROWS_PER_BLOCK=R), warmup=10, rep=50)
        bw = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
        print(f"{R:>3} {triton.cdiv(n_rows, R):>8} {ms*1e3:>9.1f}us {bw:>9.0f} {str(ok):>4}")




DEVICE = "cuda"
DTYPE = torch.float16
SHAPES = [
    (1024, 512),
    (4096, 1000),    # 非 2 的幂，检验 mask
    (8192, 2047),    # 接近 2 的幂
    (512, 4097),     # 非 2 的幂 + 中等行宽
]

def check(name, y_tri, y_ref, atol=1e-2, rtol=1e-2):
    assert_close(y_ref, y_tri, atol=atol, rtol=rtol)
    print(f"  [{name}] assert_close OK")


def bench(name, fn, x, atol=1e-2, rtol=1e-2):
    y_ref = torch.softmax(x, dim=-1)
    y = fn(x)
    check(name, y, y_ref, atol, rtol)
    ms = do_bench(lambda: fn(x), warmup=25, rep=200)
    bw = 2 * x.numel() * x.element_size() / (ms * 1e-3) / 1e9
    print(f"  [{name}] {ms*1e3:8.1f} us   {bw:7.0f} GB/s")
    return ms


def main():
    print("=" * 70)
    print("优化 A: num_warps 扫描（每行一个 program，只调 launch 参数）")
    print("=" * 70)
    for shape in SHAPES:
        x = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
        best, results = autotune_num_warps(x)
        print(f"\nshape={shape}  BLOCK_SIZE={triton.next_power_of_2(shape[1])}")
        for nw, ms in results:
            mark = "  ← best" if nw == best[0] else ""
            print(f"  num_warps={nw:>2}  {ms*1e3:8.1f} us{mark}")

    print("\n" + "=" * 70)
    print("优化 B: 多行打包（R 行/tile，比较 R=1/2/4/8/16）")
    print("=" * 70)
    for shape in SHAPES:
        x = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
        print(f"\nshape={shape}")
        compare_row_packing(x)

    print("\n" + "=" * 70)
    print("A + B 叠加：在最佳 R 下再扫 num_warps")
    print("=" * 70)
    for shape in SHAPES:
        x = torch.randn(*shape, device=DEVICE, dtype=DTYPE)
        # 先找最佳 R（固定经验 num_warps）
        best_R, best_ms = 1, float("inf")
        for R in (1, 2, 4, 8, 16):
            ms = do_bench(lambda R=R: softmax_opt_b(x, ROWS_PER_BLOCK=R), warmup=10, rep=50)
            if ms < best_ms:
                best_R, best_ms = R, ms
        # 在最佳 R 下扫 num_warps
        print(f"\nshape={shape}  best R={best_R}")
        for nw in (1, 2, 4, 8, 16):
            if nw * 32 > 1024: continue
            try:
                ms = do_bench(
                    lambda nw=nw: softmax_opt_b(x, ROWS_PER_BLOCK=best_R, num_warps=nw),
                    warmup=10, rep=50,
                )
            except Exception:
                continue
            print(f"  num_warps={nw:>2}  {ms*1e3:8.1f} us")


if __name__ == "__main__":
    main()