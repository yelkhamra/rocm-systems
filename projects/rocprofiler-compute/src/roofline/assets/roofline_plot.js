// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline.

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

  // ---- Config forwarded from roofline_shared.py via the model -------------
  var ALL_PEAKS_VALUE = model.allPeaksValue || "all";
  var ALL_PEAKS_LABEL = model.allPeaksLabel || "All peaks";
  var FALLBACK_COLOR = model.fallbackColor || "#888888";
  var ROOF_EXTREME_MAX_AI = model.roofExtremeMaxAi || 1e150;
  var PLOT_DIM_OPACITY =
    model.plotDimOpacity != null ? model.plotDimOpacity : 0.15;
  var ROOF_SAMPLES = model.roofSamples || 200;
  // Float tolerance when comparing a kernel's cumulative runtime to a slider stop.
  var RUNTIME_EPSILON = 1e-6;
  // Poll for Plotly to finish its initial paint before wiring interactivity.
  var PLOT_READY_POLL_MS = 50;
  var PLOT_READY_MAX_ATTEMPTS = 40;

  // ---- DOM handles --------------------------------------------------------
  var gd = document.getElementById(model.divId);
  var peakSelect = document.getElementById("roofline-peak-select");
  var kernelList = document.getElementById("roofline-kernel-list");
  var showAllBtn = document.getElementById("roofline-show-all");
  var kernelCountEl = document.getElementById("roofline-kernel-count");
  var runtimeSlider = document.getElementById("roofline-runtime-threshold");
  var runtimeValueEl = document.getElementById("roofline-runtime-value");
  var runtimeFilterEl = document.getElementById("roofline-runtime-filter");
  var roofList = document.getElementById("roofline-roof-list");
  var roofCountEl = document.getElementById("roofline-roof-count");
  var showAllRoofsBtn = document.getElementById("roofline-show-all-roofs");

  // ---- Model data ---------------------------------------------------------
  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var computeOverlayTraces = model.computeOverlayTraces || [];
  var peakColors = model.peakColors || {};
  var ceilingDenseHi = model.ceilingDenseHi || 0;

  // Whether any kernel carries a percent-of-runtime, which gates the filter.
  var hasRuntimeData = kernels.some(function (kernel) {
    return kernel.pctRuntime != null && isFinite(kernel.pctRuntime);
  });

  // Data-driven runtime filter: each kernel's cumulative percent of runtime
  // and the sorted set of those values used as the slider's stops.
  // Filled by computeRuntimeBreakpoints().
  var kernelCumulativePct = {};
  var runtimeBreakpoints = [];

  var memoryRoofIndices = rooflineTraces.map(function (roof) {
    return roof.traceIndex;
  });
  var computeCeilingIndices = computeTraces.map(function (ceiling) {
    return ceiling.traceIndex;
  });

  var state = {
    // The memory region shown in the aggregate view. A single
    // isolated kernel ignores this and shows every level.
    peak: model.defaultPeak || ALL_PEAKS_VALUE,
    selected: new Set(),
    // Trace indices of the memory roofs currently isolated in the legend.
    isolatedRoofs: new Set(),
    // Cumulative-runtime-percent cutoff; a kernel shows when its cumulative
    // percent is within this. Infinity shows every kernel until init sets it.
    runtimeThreshold: Infinity,
  };

  // ===== Small shared helpers ==============================================

  function isMultiSelectEvent(event) {
    return !!(event && (event.ctrlKey || event.metaKey));
  }

  function plotlyReady() {
    return gd && typeof Plotly !== "undefined";
  }

  function toggleSelection(set, key, multi) {
    if (multi) {
      if (set.has(key)) {
        set.delete(key);
      } else {
        set.add(key);
      }
      return;
    }
    if (set.size === 1 && set.has(key)) {
      set.clear();
    } else {
      set.clear();
      set.add(key);
    }
  }

  function kernelIndicesByRuntime() {
    var order = kernels.map(function (_, index) {
      return index;
    });
    order.sort(function (a, b) {
      return (kernels[b].pctRuntime || 0) - (kernels[a].pctRuntime || 0);
    });
    return order;
  }

  function setRowState(item, selected, dimmed) {
    item.classList.toggle("selected", selected);
    item.classList.toggle("dimmed", dimmed);
  }

  // Shared count shown next to each panel title.
  function formatCount(shown, total) {
    return "(" + shown + " / " + total + ")";
  }

  // Iterate the kernel-list rows, yielding the row element.
  function eachKernelRow(fn) {
    if (!kernelList) {
      return;
    }
    Array.prototype.forEach.call(kernelList.children, function (item) {
      var kernel = kernels[Number(item.dataset.index)];
      if (kernel) {
        fn(item, kernel);
      }
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

  // ===== Runtime-percent filter ============================================

  function computeRuntimeBreakpoints() {
    kernelCumulativePct = {};
    runtimeBreakpoints = [];
    if (!hasRuntimeData) {
      return;
    }
    var order = kernelIndicesByRuntime();
    var cumulative = 0;
    var i = 0;
    while (i < order.length) {
      var pct = kernels[order[i]].pctRuntime || 0;
      var group = [];
      while (i < order.length && (kernels[order[i]].pctRuntime || 0) === pct) {
        group.push(order[i]);
        i += 1;
      }
      group.forEach(function (idx) {
        cumulative += kernels[idx].pctRuntime || 0;
      });
      group.forEach(function (idx) {
        kernelCumulativePct[kernels[idx].name] = cumulative;
      });
      runtimeBreakpoints.push(cumulative);
    }
  }

  function withinThreshold(kernel) {
    if (!hasRuntimeData) {
      return true;
    }
    return (kernelCumulativePct[kernel.name] || 0) <= state.runtimeThreshold + RUNTIME_EPSILON;
  }

  function kernelIsVisible(kernel) {
    // An explicit selection overrides the runtime filter
    if (state.selected.size > 0) {
      return state.selected.has(kernel.name);
    }
    return withinThreshold(kernel);
  }

  // A kernel is actually drawn only if it is visible and has points at the
  // current peak.
  function kernelIsDrawn(kernel) {
    return kernelIsVisible(kernel) && pointsForCurrentPeak(kernel).length > 0;
  }

  function isSoleSelected(kernel) {
    return state.selected.size === 1 && state.selected.has(kernel.name);
  }

  function pointsForCurrentPeak(kernel) {
    var points = kernel.points || [];
    if (state.peak === ALL_PEAKS_VALUE || isSoleSelected(kernel)) {
      return points;
    }
    return points.filter(function (point) {
      return point.peak === state.peak;
    });
  }

  // ===== Compute ceilings / roof isolation =================================

  // Compute ceilings meet the diagonal at the leftmost bandwidth among the isolated roofs.
  function referenceBandwidth() {
    var pool = state.isolatedRoofs.size
      ? rooflineTraces.filter(function (roof) {
          return state.isolatedRoofs.has(roof.traceIndex);
        })
      : rooflineTraces;
    var bws = bandwidthsOf(pool);
    if (!bws.length) {
      bws = bandwidthsOf(rooflineTraces);
    }
    return bws.length ? Math.max.apply(null, bws) : 0;
  }

  function bandwidthsOf(roofs) {
    return roofs
      .map(function (roof) {
        return roof.bandwidth;
      })
      .filter(function (bw) {
        return bw > 0;
      });
  }

  // Highlight overlays are shown only while isolating.
  function updateCeilings() {
    if (!plotlyReady() || !computeOverlayTraces.length) {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var refBw = referenceBandwidth();
    var indices = [];
    var xs = [];
    var ys = [];
    var visibility = [];
    computeOverlayTraces.forEach(function (overlay) {
      indices.push(overlay.traceIndex);
      if (isolating && refBw && ceilingDenseHi > 0) {
        var left = overlay.peakPerf / refBw;
        var pts = logspace(left, Math.max(ceilingDenseHi, left), ROOF_SAMPLES);
        pts.push(ROOF_EXTREME_MAX_AI);
        xs.push(pts);
        ys.push(
          pts.map(function () {
            return overlay.peakPerf;
          })
        );
        visibility.push(true);
      } else {
        xs.push([]);
        ys.push([]);
        visibility.push(false);
      }
    });
    Plotly.restyle(gd, { x: xs, y: ys, visible: visibility }, indices);
  }

  // Isolate the clicked memory roof(s) by dimming the others.
  function applyRoofIsolation() {
    if (!plotlyReady()) {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var indices = [];
    var opacities = [];
    memoryRoofIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(
        !isolating || state.isolatedRoofs.has(idx) ? 1 : PLOT_DIM_OPACITY
      );
    });
    computeCeilingIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(isolating ? PLOT_DIM_OPACITY : 1);
    });
    if (indices.length) {
      Plotly.restyle(gd, { opacity: opacities }, indices);
    }
    updateCeilings();
  }

  // Isolate a memory roof, shared by the roofline panel rows and by clicking a
  // slope in the plot.
  function isolateRoof(traceIndex, multi) {
    if (memoryRoofIndices.indexOf(traceIndex) < 0) {
      return;
    }
    toggleSelection(state.isolatedRoofs, traceIndex, multi);
    applyRoofIsolation();
    updateRoofPanel();
  }

  // ===== Kernel rendering ==================================================

  // Build the per-kernel Plotly restyle payload for the current peak/selection.
  function buildKernelRestylePayload() {
    var xs = [];
    var ys = [];
    var markerColors = [];
    var customdata = [];
    var visibility = [];

    kernels.forEach(function (kernel) {
      var visible = kernelIsVisible(kernel);
      var points = visible ? pointsForCurrentPeak(kernel) : [];
      var colorByLevel = isSoleSelected(kernel);
      var baseColor = kernel.color || FALLBACK_COLOR;
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
      markerColors.push(
        points.map(function (point) {
          return colorByLevel ? peakColors[point.peak] || baseColor : baseColor;
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

    return {
      xs: xs,
      ys: ys,
      markerColors: markerColors,
      customdata: customdata,
      visibility: visibility,
    };
  }

  // Ensure the peak dropdown reflects the current selection.
  function syncPeakControl() {
    if (!peakSelect) {
      return;
    }
    if (state.selected.size === 1) {
      peakSelect.value = ALL_PEAKS_VALUE;
      peakSelect.disabled = true;
      peakSelect.title = "All memory levels are shown for the isolated kernel";
    } else {
      peakSelect.value = state.peak;
      peakSelect.disabled = false;
      peakSelect.title = "";
    }
  }

  function render() {
    syncPeakControl();
    if (!plotlyReady() || !kernelTraceIndices.length) {
      updatePanel();
      updateRoofPanel();
      return;
    }
    var payload = buildKernelRestylePayload();
    Plotly.restyle(
      gd,
      {
        x: payload.xs,
        y: payload.ys,
        "marker.color": payload.markerColors,
        customdata: payload.customdata,
        visible: payload.visibility,
      },
      kernelTraceIndices
    );
    updatePanel();
    updateRoofPanel();
  }

  function toggleKernel(name, event) {
    var multi = isMultiSelectEvent(event);
    if (multi && state.selected.size === 0) {
      kernels.forEach(function (kernel) {
        state.selected.add(kernel.name);
      });
      state.selected.delete(name);
    } else {
      toggleSelection(state.selected, name, multi);
    }
    render();
    // Isolating a single kernel brings its row into view in the kernel table.
    if (state.selected.size === 1) {
      scrollKernelIntoView(name);
    }
  }

  // Scroll a kernel's row into view within the kernel list.
  function scrollKernelIntoView(name) {
    eachKernelRow(function (item, kernel) {
      if (kernel.name === name) {
        item.scrollIntoView({ block: "nearest" });
      }
    });
  }

  // ===== Panels ============================================================

  // Build one panel with a color swatch, a label, and optional trailing
  // nodes. Shared by the kernel and roofline panels.
  function createPanelRow(opts) {
    var item = document.createElement("li");
    item.className = "roofline-panel-item";
    var dataset = opts.dataset || {};
    Object.keys(dataset).forEach(function (key) {
      item.dataset[key] = dataset[key];
    });

    var swatch = document.createElement("span");
    swatch.className = opts.swatchClass || "roofline-swatch";
    swatch.style.backgroundColor = opts.color || FALLBACK_COLOR;

    var label = document.createElement("span");
    label.className = "roofline-panel-name";
    label.textContent = opts.label;

    item.appendChild(swatch);
    item.appendChild(label);
    (opts.extras || []).forEach(function (node) {
      item.appendChild(node);
    });
    item.addEventListener("click", opts.onClick);
    return item;
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
    var allEl = document.createElement("option");
    allEl.value = ALL_PEAKS_VALUE;
    allEl.textContent = ALL_PEAKS_LABEL;
    peakSelect.appendChild(allEl);
    peakSelect.value = state.peak;
  }

  function buildKernelPanel() {
    if (!kernelList) {
      return;
    }
    // Show the heaviest kernels first.
    kernelIndicesByRuntime().forEach(function (index) {
      var kernel = kernels[index];
      var extras = [];
      // Percent of GPU resident time.
      if (kernel.pctRuntime != null && isFinite(kernel.pctRuntime)) {
        var pct = document.createElement("span");
        pct.className = "roofline-kernel-pct";
        pct.textContent = kernel.pctRuntime.toFixed(1) + "%";
        pct.title = "Percent of GPU resident time";
        extras.push(pct);
      }
      kernelList.appendChild(
        createPanelRow({
          color: kernel.color,
          label: kernel.name,
          dataset: { index: String(index) },
          extras: extras,
          onClick: function (event) {
            toggleKernel(kernel.name, event);
          },
        })
      );
    });
  }

  function buildRoofPanel() {
    if (!roofList) {
      return;
    }
    rooflineTraces.forEach(function (roof) {
      var aiaxis = document.createElement("span");
      aiaxis.className = "roofline-roof-aiaxis";
      roofList.appendChild(
        createPanelRow({
          color: peakColors[roof.level],
          label: roof.level,
          swatchClass: "roofline-swatch roofline-roof-swatch",
          dataset: { trace: String(roof.traceIndex), level: roof.level },
          extras: [aiaxis],
          onClick: function (event) {
            isolateRoof(roof.traceIndex, isMultiSelectEvent(event));
          },
        })
      );
    });
  }

  // Reflect isolation state and the active region in the roofline
  // panel rows, plus the "(shown / total)" count and the reset button.
  function updateRoofPanel() {
    var isolating = state.isolatedRoofs.size > 0;
    // While a single kernel is isolated every level is shown
    // so no single level owns the AI axis -> blank the marker.
    var axisPeak = state.selected.size === 1 ? ALL_PEAKS_VALUE : state.peak;
    if (roofList) {
      Array.prototype.forEach.call(roofList.children, function (item) {
        var idx = Number(item.dataset.trace);
        var isolated = state.isolatedRoofs.has(idx);
        setRowState(item, isolated, isolating && !isolated);
        var aiaxis = item.querySelector(".roofline-roof-aiaxis");
        if (aiaxis) {
          aiaxis.textContent = item.dataset.level === axisPeak ? "(AI axis)" : "";
        }
      });
    }
    if (roofCountEl) {
      var total = rooflineTraces.length;
      var shown = isolating ? state.isolatedRoofs.size : total;
      roofCountEl.textContent = formatCount(shown, total);
    }
    if (showAllRoofsBtn) {
      showAllRoofsBtn.disabled = !isolating;
    }
  }

  function updatePanel() {
    var filtering = state.selected.size > 0;
    eachKernelRow(function (item, kernel) {
      var selected = state.selected.has(kernel.name);
      setRowState(item, selected, filtering && !selected);
      // Trim rows outside the runtime threshold, but never a selected one.
      item.classList.toggle("filtered", !withinThreshold(kernel) && !selected);
    });
    // Count how many kernels are actually drawn under the current peak +
    // selection filters, shown as "(drawn / total)" next to the title.
    var shown = 0;
    kernels.forEach(function (kernel) {
      if (kernelIsDrawn(kernel)) {
        shown += 1;
      }
    });
    if (kernelCountEl) {
      kernelCountEl.textContent = formatCount(shown, kernels.length);
    }
    if (showAllBtn) {
      showAllBtn.disabled = !filtering;
    }
  }

  // ===== Wiring / lifecycle ================================================

  function wireEvents() {
    if (peakSelect) {
      peakSelect.addEventListener("change", function () {
        // "all" is a valid, user-selectable region (every memory level).
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
    if (showAllRoofsBtn) {
      showAllRoofsBtn.addEventListener("click", function () {
        state.isolatedRoofs.clear();
        applyRoofIsolation();
        updateRoofPanel();
      });
    }
    if (runtimeSlider) {
      runtimeSlider.addEventListener("input", function () {
        var index = Number(runtimeSlider.value);
        state.runtimeThreshold = runtimeBreakpoints[index];
        updateRuntimeLabel();
        render();
      });
    }
    if (gd && typeof gd.on === "function") {
      gd.on("plotly_click", function (data) {
        if (!data || !data.points || !data.points.length) {
          return;
        }
        var traceIndex = data.points[0].curveNumber;
        // Clicking a roof slope isolates it, same as its panel row.
        if (memoryRoofIndices.indexOf(traceIndex) >= 0) {
          isolateRoof(traceIndex, isMultiSelectEvent(data.event));
          return;
        }
        var position = kernelTraceIndices.indexOf(traceIndex);
        if (position < 0 || !kernels[position]) {
          return;
        }
        toggleKernel(kernels[position].name, data.event);
      });
    }
  }

  function whenPlotReady(callback, attemptsLeft) {
    if (plotlyReady() && typeof gd.on === "function") {
      callback();
      return;
    }
    if (attemptsLeft <= 0) {
      callback();
      return;
    }
    setTimeout(function () {
      whenPlotReady(callback, attemptsLeft - 1);
    }, PLOT_READY_POLL_MS);
  }

  function resizePlot() {
    if (plotlyReady() && Plotly.Plots) {
      Plotly.Plots.resize(gd);
    }
  }

  // Show the true cumulative percent of runtime covered at the current stop.
  function updateRuntimeLabel() {
    if (runtimeValueEl) {
      runtimeValueEl.textContent = state.runtimeThreshold.toFixed(1) + "%";
    }
  }

  // Point the slider at the data-driven breakpoints: one stop per kernel
  // boundary, defaulting to the last (every kernel shown).
  function initRuntimeSlider() {
    if (!runtimeSlider || !runtimeBreakpoints.length) {
      return;
    }
    var lastIndex = runtimeBreakpoints.length - 1;
    runtimeSlider.min = "0";
    runtimeSlider.max = String(lastIndex);
    runtimeSlider.step = "1";
    runtimeSlider.value = String(lastIndex);
    state.runtimeThreshold = runtimeBreakpoints[lastIndex];
    updateRuntimeLabel();
  }

  function init() {
    buildPeakOptions();
    buildKernelPanel();
    buildRoofPanel();
    computeRuntimeBreakpoints();
    // The runtime filter is meaningless without per-kernel runtime data.
    if (runtimeFilterEl && !hasRuntimeData) {
      runtimeFilterEl.style.display = "none";
    }
    initRuntimeSlider();
    whenPlotReady(function () {
      wireEvents();
      resizePlot();
      render();
    }, PLOT_READY_MAX_ATTEMPTS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
