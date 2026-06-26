# Memory Bandwidth Analysis in Memory Chart

## System Context

In the analyze mode, rocprof-compute renders the "Memory Chart", a high-level graphical
representation of the memory hierarchy populated with numeric metrics to aid user understanding
of their workload's memory performance.

In gfx950, rocprof-compute has memory bandwidth analysis metrics describing
bandwidth, latency, and throughput within the memory hierarchy.

Currently, the gfx950 memory bandwidth analysis metrics are hidden behind the tool option
`--experimental`, and while these metrics are relatable to what the memory chart presents,
there is no relationship between the two features.

## Problem Statement

**[PS1] No bottleneck identification.**
The memory chart presents metric results but does not identify which memory levels are
bottlenecks. Users must manually interpret request counts, hit rates, latencies, and stall indicators
across memory levels. There is no uniform method to attain bottlenecks relating to the workload.

**[PS2] Bottleneck analysis exists but is inaccessible.**
Bottleneck detection equations for GL1/GL2/EA exist in the codebase but are gated
behind `--experimental --membw-analysis`, disconnected from the memory chart, and
presented only as flat metric tables (block 30).

**[PS3] No guided analysis.**
Users lack structured guidance when diagnosing memory performance issues. Memory
bandwidth guided analysis has been a long-standing user request.

**[PS4] No structured data export.**
Downstream tools (Optiq) hardcode the memory chart structure rather than consuming it
from the analysis database. Bottleneck results have no database representation at all.
Decoupling these tools requires a structured, data-driven export.

## Scope

1. gfx950 only (initial target)
2. Analyze mode only (profiled data is assumed available)
3. Memory levels: GL1, GL2, EA (where bottleneck equations exist today)

## Assumptions

- **[PR1]** gfx950 memory chart must be updated to reflect current HW topology (gfx115)
- **[PR2]** mem-bw-analysis metrics must be validated and promoted out of experimental

## Requirements

**[FR1]** (PS1) Show bottleneck detection results on the memory chart at the relevant
memory level, showing only detected (active) bottlenecks along with their supporting
metrics.

  - **[FR1.1]** Evaluate bottleneck equations from profiled data to produce boolean results per memory level.
  - **[FR1.2]** Display only active (true) bottlenecks — no visual noise from non-triggered indicators.
  - **[FR1.3]** Show supporting metric values alongside each active bottleneck (e.g., stall rate = 15%).
  - **[FR1.4]** CLI renderer only (initial target). TUI and GUI renderers are out of scope for this effort.

**[FR2]** (PS2) Remove the experimental gate — mem-bw-analysis becomes a standard part
of the analyze flow.

  - **[FR2.1]** Counters required by mem-bw-analysis must be collected in standard profile mode without `--experimental`.
  - **[FR2.2]** Existing `--experimental --membw-analysis` invocation must continue to work or be gracefully deprecated.

**[FR3]** (PS3) When a bottleneck is detected, provide textual guidance to the user.

  - **[FR3.1]** Design the specific content and form of this guidance.
  - **[FR3.2]** Guidance should incorporate actual metric values, not only static text.
  - **[FR3.3]** Define placement of guidance relative to the memory chart (inline, below, separate panel).

**[IR4]** (PS4) Persist results for downstream consumption (Optiq).

  - **[IR4.1]** Coordinate with Optiq on database schema and JSON format.

## Design

### Decision 1: Where do bottleneck equations live?

Bottleneck equations form a dependency tree where child conditions reference parent results
and use negation (e.g., `gl1_bottleneck_tcp_utcl1` requires `gl1_bottleneck_tcp == true`;
`gl1_bottleneck_tcp_other` requires siblings to be false).

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Pure YAML | Define all equations as YAML metrics with a new annotation | Consistent with existing metric authoring | YAML engine cannot reference other metric results, express negation, or model parent/child dependencies |
| **Hybrid (chosen)** | **Raw stall rate metrics in YAML; bottleneck decision tree in Python** | **Reuses existing YAML metrics; tree logic expressed naturally in Python; thresholds remain editable in config** | **Two places to understand the full picture** |
| Pure Python | All definitions (thresholds, tree, boolean logic) in Python | Single location for all logic | Threshold values in code, inconsistent with metric authoring pattern, harder to tune |

### Decision 2: How do bottleneck results reach the memory chart renderer?

`plot_mem_chart(normal_unit, metric_dict)` currently receives a flat dictionary. Bottleneck
results are hierarchical (parent/child booleans with supporting values).

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Extend `metric_dict` | Flatten bottleneck booleans into the existing dict | Minimal change to data flow | Loses hierarchy; mixes raw metrics with derived analysis |
| **Separate data channel (chosen)** | **Pass bottleneck results as a new argument to `plot_mem_chart()`** | **Clean separation; structured representation preserves hierarchy** | **Requires updating function signature and call sites** |

### Decision 3: How are bottleneck indicators displayed to the user?

The CLI renderer (`mem_chart_gfx9.py`) draws on a fixed 234x42 plotille canvas with
absolute coordinates per block.

| Option | Description | Pros | Cons |
|--------|------------|------|------|
| Annotate blocks only | Colored text within or adjacent to block rectangles | Visual proximity to memory level | Limited space; no room for guidance text |
| Dedicated canvas region | New area on the canvas listing active bottlenecks | More space for detail | Canvas size must increase; layout complexity |
| Separate panel only | Rich table or text block after the plotille canvas | Simplest implementation | Loses visual connection to memory levels |
| **Annotate blocks + guidance below (chosen)** | **Active bottleneck labels on the canvas blocks; detailed guidance text printed below the chart** | **Visual proximity on chart; guidance has room for detail and metric values** | **Two rendering concerns (canvas labels + post-chart text)** |
