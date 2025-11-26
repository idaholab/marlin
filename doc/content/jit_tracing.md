# JIT Tracing System

Marlin supports Just-In-Time (JIT) compilation of tensor compute sequences using PyTorch's tracing infrastructure. This feature can significantly reduce GPU memory usage and improve performance by eliminating intermediate tensor allocations and enabling graph-level optimizations.

## Overview

When JIT tracing is enabled, sequences of tensor computations within a `ComputeGroup` are captured into a PyTorch JIT graph during the first execution. Subsequent executions use the optimized, compiled graph rather than executing individual compute operations. This approach:

- **Reduces memory allocations**: Intermediate tensors can be eliminated or reused
- **Enables fusion**: Multiple operations can be fused into single GPU kernels
- **Improves cache efficiency**: Better memory access patterns through graph optimization
- **Supports shape generalizability**: Graphs traced with small tensors work with large tensors

## Architecture

The JIT tracing system consists of three main components:

### TraceSchema

A cache key that determines when a traced graph can be reused. Importantly, traces are keyed by **dimension count** (ndim), not concrete sizes, enabling a single trace to work across different grid sizes.

```cpp
struct TraceSchema {
  std::vector<int64_t> tensor_ndims;  // Number of dimensions per tensor (NOT sizes!)
  at::DispatchKey dispatch_key;        // Device (CPU/CUDA) information
};
```

This design follows the NEML2 approach, where `torch::jit::tracer::getSizeOf()` creates symbolic dimension references in the traced graph that evaluate to actual sizes at runtime.

### TracedComputeSequence

Manages a contiguous sequence of JIT-traceable compute operations. Handles:

- Collecting input/output buffer information
- Performing the actual tracing via PyTorch's `torch::jit::tracer::trace()`
- Caching traced graphs by schema
- Executing traced graphs with current tensor data

### JITExecutor

Builds and manages the execution plan for a `ComputeGroup`. It:

- Analyzes compute operations for JIT compatibility via `supportsJIT()`
- Splits the execution into segments around non-JITable operations
- Creates `TracedComputeSequence` instances for contiguous JITable sections
- Coordinates execution of traced and non-traced segments

## Execution Flow

```
ComputeGroup::computeBuffer()
    │
    ▼
JITExecutor::execute()
    │
    ├─► For each TracedComputeSequence:
    │       │
    │       ▼
    │     Compute TraceSchema from input tensors
    │       │
    │       ▼
    │     Cache lookup by schema
    │       │
    │       ├─► HIT: Execute cached graph with GraphExecutor
    │       │
    │       └─► MISS: Trace operations, optimize graph, cache, then execute
    │
    └─► For each non-JITable compute:
            │
            ▼
          Direct execution via computeBuffer()
```

## Segment Splitting

When a `ComputeGroup` contains non-JITable operations, the JITExecutor automatically splits the execution into segments:

```
Computes: [A, B, C*, D, E, F*, G]  where * = non-JITable

Execution Plan:
  1. TracedComputeSequence([A, B])  ← traced together
  2. Non-JITable: C                  ← direct execution
  3. TracedComputeSequence([D, E])  ← traced together
  4. Non-JITable: F                  ← direct execution
  5. TracedComputeSequence([G])     ← traced alone
```

## Configuration

### Enabling/Disabling JIT

JIT tracing is **enabled by default** for all `ComputeGroup` objects. To disable:

```
[TensorComputes]
  [Solve]
    [my_group]
      type = ComputeGroup
      computes = 'compute1 compute2 compute3'
      enable_jit = false
    []
  []
[]
```

### Per-Compute JIT Support

Individual compute classes can opt out of JIT tracing by overriding `supportsJIT()`:

```cpp
class MyCompute : public TensorOperator<>
{
public:
  // Return false if this compute cannot be JIT traced
  virtual bool supportsJIT() const override { return false; }
};
```

## Non-JITable Operations

The following compute types have `supportsJIT() = false` by default:

| Class | Reason |
|-------|--------|
| `RandomTensor` | Random number generation is non-deterministic |
| `NEML2TensorCompute` | Has its own internal JIT infrastructure |
| `MooseFunctionTensor` | Calls external MOOSE functions not visible to tracer |
| `TensorSolver` (and subclasses) | Iterative algorithms with data-dependent control flow |
| `FFTMechanics` | Contains iterative solving with data-dependent convergence |

### When to Mark a Compute as Non-JITable

Override `supportsJIT()` to return `false` if your compute:

1. **Uses random number generation** with different values each call
2. **Has data-dependent control flow** (e.g., `if (tensor_value > threshold)`)
3. **Calls external libraries** that PyTorch cannot trace
4. **Modifies global state** that affects subsequent computations
5. **Uses operations not supported by the tracer** (see PyTorch JIT limitations)

## Generalized Tracing Across Grid Sizes

A key feature of this implementation is the ability to trace at a small grid size and run at a larger size. This is particularly useful when:

1. **Memory constraints**: The untraced computation doesn't fit in GPU memory, but the traced version (with fused operations) would
2. **Development workflow**: Quickly trace on a small test case, then deploy to production-size grids
3. **Batch processing**: Use the same trace for varying input sizes

### How It Works

Following the NEML2 approach, the tracer uses `torch::jit::tracer::getSizeOf()` to create **symbolic dimension references** instead of hardcoded constants:

```cpp
// During tracing, this creates a graph node that reads the dimension at runtime
TraceableSize size = getTraceableSize(tensor, 0);

// The traced graph contains: tensor.size(0) -> symbolic reference
// NOT: hardcoded value like 64
```

### Example Workflow

```
1. Trace at small size (16×16 grid that fits in memory):
   - Graph captures operations with symbolic size references
   - Cache stores graph keyed by ndim (e.g., [3] for 3D tensor)

2. Run at large size (512×512 grid):
   - Same TraceSchema (ndim = [3], same device)
   - Cached graph retrieved
   - Symbolic size references evaluate to 512 at runtime
   - Fused operations execute with reduced memory footprint
```

### TraceableSize and TraceableTensorShape

Helper types are provided for working with potentially symbolic dimensions:

```cpp
// TraceableSize: variant type holding either int64_t or torch::Tensor (symbolic)
TraceableSize size = getTraceableSize(input_tensor, 0);
int64_t concrete = size.concrete();  // Evaluates symbolic if needed

// TraceableTensorShape: collection of TraceableSize values
TraceableTensorShape shape = getTraceableShape(tensor);
std::vector<int64_t> dims = shape.concrete();  // Get all concrete values
```

These are available via `TraceableUtils.h` and as protected methods on `TensorOperatorBase`.

## Graph Optimizations

Traced graphs automatically receive the following optimizations:

- **Dead Code Elimination**: Removes unused computations
- **Constant Propagation**: Pre-computes constant expressions
- **Common Subexpression Elimination**: Reuses repeated calculations
- **Graph Fusion**: Combines operations into efficient kernels

## Cache Invalidation

Traced graphs are automatically invalidated when:

- **Grid changes**: `TensorProblem::gridChanged()` triggers cache clear
- **Schema changes**: Different tensor shapes create new cached graphs
- **Device changes**: CPU/GPU switches require new traces

Manual invalidation:

```cpp
// In ComputeGroup, JIT caches are cleared on gridChanged()
void ComputeGroup::gridChanged() {
  if (_jit_executor)
    _jit_executor->invalidateAllCaches();
}
```

## Thread Safety

PyTorch JIT tracing is **not thread-safe**. The implementation uses a global mutex to serialize tracing operations:

```cpp
static std::mutex s_trace_mutex;
std::lock_guard<std::mutex> lock(s_trace_mutex);
// ... perform tracing ...
```

Execution of already-traced graphs is thread-safe.

## Debugging

### Checking JIT Status

Enable debug output in the `Domain` block to see JIT executor statistics:

```
[Domain]
  ...
  debug = true
[]
```

This will print:
```
JIT executor built: 2 traced sequences, 1 non-JITable computes
```

### Disabling JIT for Debugging

If you suspect JIT-related issues, disable it to compare behavior:

```
[TensorComputes]
  [Solve]
    [group]
      type = ComputeGroup
      computes = '...'
      enable_jit = false  # Disable for debugging
    []
  []
[]
```

### Common Issues

1. **"Failed to trace compute sequence"**: A compute operation uses PyTorch features not supported by the tracer. Mark it as non-JITable.

2. **Results differ with JIT enabled/disabled**: The traced graph may have captured tensor values as constants. Ensure tensors are passed as inputs, not captured from closure.

3. **Memory not reduced as expected**: Check if non-JITable computes are fragmenting your sequences into many small traced segments.

## Performance Considerations

### When JIT Helps Most

- Long sequences of element-wise operations
- Repeated execution of the same compute pattern
- Operations that can be fused (e.g., multiple additions/multiplications)

### When JIT May Not Help

- Single-operation computes (tracing overhead > benefit)
- Operations already highly optimized (e.g., cuBLAS GEMM)
- Compute sequences fragmented by many non-JITable operations

### Memory vs. Speed Tradeoff

JIT tracing caches graphs for each unique `TraceSchema`. If your simulation uses many different tensor shapes, this can increase memory usage. Consider disabling JIT for such cases.

## Implementation Files

| File | Purpose |
|------|---------|
| `include/utils/TraceSchema.h` | Cache key structure (keyed by ndim, not size) |
| `include/utils/TraceableSize.h` | Variant type for symbolic/concrete dimensions |
| `include/utils/TraceableTensorShape.h` | Collection of traceable dimensions |
| `include/utils/TraceableUtils.h` | Utility functions for extracting traceable sizes |
| `include/utils/TracedComputeSequence.h` | Sequence tracing and execution |
| `include/utils/JITExecutor.h` | Execution plan management |
| `src/utils/TraceSchema.C` | Schema comparison operators |
| `src/utils/TracedComputeSequence.C` | Tracing implementation |
| `src/utils/JITExecutor.C` | Segment building and execution |

## References

This implementation follows patterns established by the NEML2 library's JIT tracing system. Key reference files:

- `moose/framework/contrib/neml2/src/neml2/models/Model.cxx` - `forward_maybe_jit()` implementation
- `moose/framework/contrib/neml2/include/neml2/jit/TraceableTensorShape.h` - Shape handling patterns
