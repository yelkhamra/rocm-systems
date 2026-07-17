// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline. It reads the embedded
// JSON model (#roofline-model), then keeps this state:
//   * peak         -- memory region for the aggregate view (default HBM); a
//                     single isolated kernel instead shows every level
//   * selected     -- a Set of isolated kernel names
//   * isolatedRoofs -- Set of memory-roof trace indices isolated in the legend

(function () {
  "use strict";

  var modelEl = document.getElementById("roofline-model");
  if (!modelEl) {
    return;
  }

  var model;
  try {
    model = JSON.parse(modelEl.textContent);
  } catch (err) {
    return;
  }

  var gd = document.getElementById(model.divId);
  var peakSelect = document.getElementById("roofline-peak-select");
  var kernelList = document.getElementById("roofline-kernel-list");
  var showAllBtn = document.getElementById("roofline-show-all");
  var kernelCountEl = document.getElementById("roofline-kernel-count");
  var runtimeSlider = document.getElementById("roofline-runtime-threshold");
  var runtimeValueEl = document.getElementById("roofline-runtime-value");
  var runtimeFilterEl = document.getElementById("roofline-runtime-filter");

  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var peakColors = model.peakColors || {};
  var ceilingDenseHi = model.ceilingDenseHi || 0;
  var roofSamples = model.roofSamples || 200;
  var ROOF_EXTREME_MAX_AI = 1e150;

  // Whether any kernel carries a percent-of-runtime, which gates the filter.
  var hasRuntimeData = kernels.some(function (kernel) {
    return kernel.pctRuntime != null && isFinite(kernel.pctRuntime);
  });
  // Names of the kernels within the current cumulative runtime threshold
  var thresholdSet = null;

  var memoryRoofIndices = rooflineTraces.map(function (roof) {
    return roof.traceIndex;
  });
  var computeCeilingIndices = computeTraces.map(function (ceiling) {
    return ceiling.traceIndex;
  });

  var state = {
    // The memory region shown in the aggregate (multi-kernel) view; defaults to
    // HBM. A single isolated kernel ignores this and shows every level.
    peak: model.defaultPeak || "HBM",
    selected: new Set(),
    // Trace indices of the memory roofs currently isolated in the legend.
    isolatedRoofs: new Set(),
    // Cumulative percent of GPU resident time to display.
    runtimeThreshold: runtimeSlider ? Number(runtimeSlider.value) : 100,
  };

  // The plot is framed once on open; after that the user drives pan/zoom.
  var hasFitted = false;

  // The runtime filter keeps only the heaviest kernels whose cumulative percent
  // of GPU resident time reaches the threshold (100% keeps every kernel).
  function recomputeThresholdSet() {
    thresholdSet = new Set();
    if (state.runtimeThreshold >= 100) {
      kernels.forEach(function (kernel) {
        thresholdSet.add(kernel.name);
      });
      return;
    }
    var order = kernels.map(function (_, index) {
      return index;
    });
    order.sort(function (a, b) {
      return (kernels[b].pctRuntime || 0) - (kernels[a].pctRuntime || 0);
    });
    var cumulative = 0;
    for (var i = 0; i < order.length; i++) {
      var kernel = kernels[order[i]];
      thresholdSet.add(kernel.name);
      cumulative += kernel.pctRuntime || 0;
      if (cumulative >= state.runtimeThreshold) {
        break;
      }
    }
  }

  function withinThreshold(kernel) {
    return !thresholdSet || thresholdSet.has(kernel.name);
  }

  function kernelIsVisible(kernel) {
    // An explicit selection overrides the runtime filter; otherwise the filter
    // governs which kernels are drawn.
    if (state.selected.size > 0) {
      return state.selected.has(kernel.name);
    }
    return withinThreshold(kernel);
  }

  function pointsForCurrentPeak(kernel) {
    var points = kernel.points || [];
    // A single isolated kernel shows every memory level (colored by level);
    // otherwise each kernel shows one dot at the selected memory region.
    if (state.selected.size === 1 && state.selected.has(kernel.name)) {
      return points;
    }
    return points.filter(function (point) {
      return point.peak === state.peak;
    });
  }

  function logspace(lo, hi, n) {
    var a = Math.log10(lo);
    var b = Math.log10(hi);
    var steps = Math.max(n - 1, 1);
    var out = [];
    for (var i = 0; i < n; i++) {
      out.push(Math.pow(10, a + ((b - a) * i) / steps));
    }
    return out;
  }

  // Steepest bandwidth among the isolated roofs (or all roofs when none are
  // isolated); the compute ceilings meet the diagonal at peak / this bandwidth.
  function referenceBandwidth() {
    var pool = rooflineTraces;
    if (state.isolatedRoofs.size) {
      pool = rooflineTraces.filter(function (roof) {
        return state.isolatedRoofs.has(roof.traceIndex);
      });
    }
    var bws = pool
      .map(function (roof) {
        return roof.bandwidth;
      })
      .filter(function (bw) {
        return bw > 0;
      });
    if (!bws.length) {
      bws = rooflineTraces
        .map(function (roof) {
          return roof.bandwidth;
        })
        .filter(function (bw) {
          return bw > 0;
        });
    }
    return bws.length ? Math.max.apply(null, bws) : 0;
  }

  // Snap each compute ceiling's left endpoint to the ridge of the isolated
  // slope(s) so the flat cap meets the diagonal, and re-sample densely so the
  // whole line stays hoverable.
  function updateCeilings() {
    if (!gd || typeof Plotly === "undefined" || !computeTraces.length) {
      return;
    }
    var refBw = referenceBandwidth();
    if (!refBw || !(ceilingDenseHi > 0)) {
      return;
    }
    var indices = [];
    var xs = [];
    var ys = [];
    computeTraces.forEach(function (ceiling) {
      var left = ceiling.peakPerf / refBw;
      var pts = logspace(left, Math.max(ceilingDenseHi, left), roofSamples);
      pts.push(ROOF_EXTREME_MAX_AI);
      indices.push(ceiling.traceIndex);
      xs.push(pts);
      ys.push(
        pts.map(function () {
          return ceiling.peakPerf;
        })
      );
    });
    Plotly.restyle(gd, { x: xs, y: ys }, indices);
  }

  // Isolate the clicked memory roof(s) by dimming the others. The horizontal
  // compute peaks cap every roofline, so they always stay fully opaque -- an
  // isolated memory roof keeps its horizontal peak visible too.
  function applyRoofIsolation() {
    if (!gd || typeof Plotly === "undefined") {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var indices = [];
    var opacities = [];
    memoryRoofIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(!isolating || state.isolatedRoofs.has(idx) ? 1 : 0.15);
    });
    computeCeilingIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(1);
    });
    if (indices.length) {
      Plotly.restyle(gd, { opacity: opacities }, indices);
    }
    // The flat cap should meet the isolated slope's ridge, not stay put.
    updateCeilings();
  }

  // Mark the active memory region's roof in the legend with an "(AI axis)"
  // suffix (the dropdown drives which level's AI is on the x-axis).
  function updateRoofLegendLabels() {
    if (!gd || typeof Plotly === "undefined" || !rooflineTraces.length) {
      return;
    }
    var indices = [];
    var names = [];
    rooflineTraces.forEach(function (roof) {
      indices.push(roof.traceIndex);
      names.push(roof.level + (roof.level === state.peak ? " (AI axis)" : ""));
    });
    Plotly.restyle(gd, { name: names }, indices);
  }

  function render() {
    updateRoofLegendLabels();
    // Keep the region dropdown in sync with the selection (single vs aggregate).
    syncPeakControl();
    if (!gd || typeof Plotly === "undefined" || !kernelTraceIndices.length) {
      updatePanel();
      return;
    }

    var xs = [];
    var ys = [];
    var markerColors = [];
    var customdata = [];
    var visibility = [];

    kernels.forEach(function (kernel) {
      var visible = kernelIsVisible(kernel);
      var points = visible ? pointsForCurrentPeak(kernel) : [];
      var singleSel =
        state.selected.size === 1 && state.selected.has(kernel.name);
      var baseColor = kernel.color || "#888888";
      xs.push(
        points.map(function (point) {
          return point.ai;
        })
      );
      ys.push(
        points.map(function (point) {
          return point.perf;
        })
      );
      // Aggregate view colors by kernel; an isolated kernel colors each dot by
      // its memory level so the levels stay distinguishable without shapes.
      markerColors.push(
        points.map(function (point) {
          return singleSel ? peakColors[point.peak] || baseColor : baseColor;
        })
      );
      // The full tooltip body is precomputed server-side; just pass it back.
      customdata.push(
        points.map(function (point) {
          return [point.hover];
        })
      );
      visibility.push(visible && points.length > 0);
    });

    Plotly.restyle(
      gd,
      {
        x: xs,
        y: ys,
        "marker.color": markerColors,
        customdata: customdata,
        visible: visibility,
      },
      kernelTraceIndices
    );
    updatePanel();
    fitView();
  }

  // Frame the log axes on the initial draw only
  function fitView() {
    if (!gd || typeof Plotly === "undefined" || hasFitted) {
      return;
    }
    var ais = [];
    var perfs = [];
    kernels.forEach(function (kernel) {
      if (!kernelIsVisible(kernel)) {
        return;
      }
      pointsForCurrentPeak(kernel).forEach(function (point) {
        if (point.ai > 0) {
          ais.push(point.ai);
        }
        if (point.perf > 0) {
          perfs.push(point.perf);
        }
      });
    });
    // Nothing drawn under the current filters: leave the view untouched.
    if (!ais.length || !perfs.length) {
      return;
    }
    computeTraces.forEach(function (ceiling) {
      if (ceiling.peakPerf > 0) {
        perfs.push(ceiling.peakPerf);
      }
    });

    var pad = 5; // symmetric padding in log space
    var ranges = {
      "xaxis.range": [
        Math.log10(Math.min.apply(null, ais) / pad),
        Math.log10(Math.max.apply(null, ais) * pad),
      ],
      "yaxis.range": [
        Math.log10(Math.min.apply(null, perfs) / pad),
        Math.log10(Math.max.apply(null, perfs) * pad),
      ],
    };

    hasFitted = true;
    Plotly.relayout(gd, ranges);
  }

  function toggleKernel(name, event) {
    var multi = event && (event.ctrlKey || event.metaKey);
    if (multi) {
      if (state.selected.size === 0) {
        // First multi-select from the "all" baseline: start with everything
        // selected, then remove the clicked kernel (hide just that one).
        kernels.forEach(function (kernel) {
          state.selected.add(kernel.name);
        });
        state.selected.delete(name);
      } else if (state.selected.has(name)) {
        state.selected.delete(name);
      } else {
        state.selected.add(name);
      }
    } else if (state.selected.size === 1 && state.selected.has(name)) {
      // Clicking the already-isolated kernel clears the filter.
      state.selected.clear();
    } else {
      state.selected.clear();
      state.selected.add(name);
    }
    render();
  }

  function buildPeakOptions() {
    if (!peakSelect) {
      return;
    }
    (model.peaks || []).forEach(function (peak) {
      var el = document.createElement("option");
      el.value = peak;
      el.textContent = peak;
      peakSelect.appendChild(el);
    });
    // "All peaks" only applies to a single isolated kernel; hidden otherwise.
    var allEl = document.createElement("option");
    allEl.value = "all";
    allEl.textContent = "All peaks";
    allEl.hidden = true;
    peakSelect.appendChild(allEl);
    peakSelect.value = state.peak;
  }

  // The region dropdown only applies to the aggregate view. When
  // exactly one kernel is isolated we show all of its memory levels, so the
  // control switches to a disabled "All peaks" and restores the region after.
  function syncPeakControl() {
    if (!peakSelect) {
      return;
    }
    var single = state.selected.size === 1;
    var allOpt = peakSelect.querySelector('option[value="all"]');
    if (allOpt) {
      allOpt.hidden = !single;
    }
    if (single) {
      peakSelect.value = "all";
      peakSelect.disabled = true;
    } else {
      if (peakSelect.value === "all") {
        peakSelect.value = state.peak;
      }
      peakSelect.disabled = false;
    }
  }

  function buildKernelPanel() {
    if (!kernelList) {
      return;
    }
    // Show the heaviest kernels first, but keep dataset.index pointing at the
    // original kernels[] position so click handling and restyle stay aligned.
    var order = kernels.map(function (_, index) {
      return index;
    });
    order.sort(function (a, b) {
      return (kernels[b].pctRuntime || 0) - (kernels[a].pctRuntime || 0);
    });
    order.forEach(function (index) {
      var kernel = kernels[index];
      var item = document.createElement("li");
      item.className = "roofline-kernel-item";
      item.dataset.index = String(index);
      item.title = kernel.name;

      var swatch = document.createElement("span");
      swatch.className = "roofline-swatch";
      swatch.style.backgroundColor = kernel.color || "#888888";

      var name = document.createElement("span");
      name.className = "roofline-kernel-name";
      name.textContent = kernel.name;

      item.appendChild(swatch);
      item.appendChild(name);

      // Percent of GPU resident time.
      if (kernel.pctRuntime != null && isFinite(kernel.pctRuntime)) {
        var pct = document.createElement("span");
        pct.className = "roofline-kernel-pct";
        pct.textContent = kernel.pctRuntime.toFixed(1) + "%";
        pct.title = "Percent of GPU resident time";
        item.appendChild(pct);
      }

      item.addEventListener("click", function (event) {
        toggleKernel(kernel.name, event);
      });
      kernelList.appendChild(item);
    });
  }

  function updatePanel() {
    var filtering = state.selected.size > 0;
    var shown = 0;
    if (kernelList) {
      Array.prototype.forEach.call(kernelList.children, function (item) {
        var kernel = kernels[Number(item.dataset.index)];
        if (!kernel) {
          return;
        }
        var selected = state.selected.has(kernel.name);
        item.classList.toggle("selected", selected);
        item.classList.toggle("dimmed", filtering && !selected);
        // Trim rows outside the runtime threshold, but never a selected one.
        item.classList.toggle("filtered", !withinThreshold(kernel) && !selected);
      });
    }
    // Count how many kernels are actually drawn under the current peak +
    // selection filters, shown as "(drawn / total)" next to the title.
    kernels.forEach(function (kernel) {
      if (kernelIsVisible(kernel) && pointsForCurrentPeak(kernel).length > 0) {
        shown += 1;
      }
    });
    if (kernelCountEl) {
      kernelCountEl.textContent = "(" + shown + " / " + kernels.length + ")";
    }
    // The reset button only means something while a filter is active; disable
    // it otherwise so it never looks like a no-op click.
    if (showAllBtn) {
      showAllBtn.disabled = !filtering;
    }
  }

  function wireEvents() {
    if (peakSelect) {
      peakSelect.addEventListener("change", function () {
        // "All peaks" is a disabled single-kernel display state, never chosen.
        if (peakSelect.value === "all") {
          return;
        }
        state.peak = peakSelect.value;
        render();
      });
    }
    if (showAllBtn) {
      showAllBtn.addEventListener("click", function () {
        state.selected.clear();
        render();
      });
    }
    if (runtimeSlider) {
      runtimeSlider.addEventListener("input", function () {
        state.runtimeThreshold = Number(runtimeSlider.value);
        if (runtimeValueEl) {
          runtimeValueEl.textContent = state.runtimeThreshold + "%";
        }
        recomputeThresholdSet();
        render();
      });
    }
    if (gd && typeof gd.on === "function") {
      gd.on("plotly_click", function (data) {
        if (!data || !data.points || !data.points.length) {
          return;
        }
        var traceIndex = data.points[0].curveNumber;
        var position = kernelTraceIndices.indexOf(traceIndex);
        if (position < 0 || !kernels[position]) {
          return;
        }
        toggleKernel(kernels[position].name, data.event);
      });
      // Clicking a memory roof in the legend isolates it 
      // Ctrl/Cmd-click adds or removes a roof from the isolation set. 
      // The compute peaks are not individually isolatable; clicking one clears the isolation.
      gd.on("plotly_legendclick", function (ev) {
        if (!ev || typeof ev.curveNumber !== "number") {
          return false;
        }
        var idx = ev.curveNumber;
        if (computeCeilingIndices.indexOf(idx) >= 0) {
          state.isolatedRoofs.clear();
          applyRoofIsolation();
          return false;
        }
        if (memoryRoofIndices.indexOf(idx) < 0) {
          return false;
        }
        var multi = ev.event && (ev.event.ctrlKey || ev.event.metaKey);
        if (multi) {
          if (state.isolatedRoofs.has(idx)) {
            state.isolatedRoofs.delete(idx);
          } else {
            state.isolatedRoofs.add(idx);
          }
        } else if (
          state.isolatedRoofs.size === 1 &&
          state.isolatedRoofs.has(idx)
        ) {
          // Clicking the sole isolated roof clears the isolation.
          state.isolatedRoofs.clear();
        } else {
          state.isolatedRoofs.clear();
          state.isolatedRoofs.add(idx);
        }
        applyRoofIsolation();
        return false;
      });
      // Double-click is the "show everything" gesture: clear any isolation.
      gd.on("plotly_legenddoubleclick", function () {
        state.isolatedRoofs.clear();
        applyRoofIsolation();
        return false;
      });
    }
  }

  function whenPlotReady(callback, attemptsLeft) {
    if (gd && typeof Plotly !== "undefined" && typeof gd.on === "function") {
      callback();
      return;
    }
    if (attemptsLeft <= 0) {
      callback();
      return;
    }
    setTimeout(function () {
      whenPlotReady(callback, attemptsLeft - 1);
    }, 50);
  }

  function resizePlot() {
    if (gd && typeof Plotly !== "undefined" && Plotly.Plots) {
      Plotly.Plots.resize(gd);
    }
  }

  function init() {
    buildPeakOptions();
    buildKernelPanel();
    // The runtime filter is meaningless without per-kernel runtime data.
    if (runtimeFilterEl && !hasRuntimeData) {
      runtimeFilterEl.style.display = "none";
    }
    recomputeThresholdSet();
    whenPlotReady(function () {
      wireEvents();
      resizePlot();
      render();
    }, 40);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
