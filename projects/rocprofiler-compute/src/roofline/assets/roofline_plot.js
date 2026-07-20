// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline. It reads the embedded
// JSON model (#roofline-model), then keeps this state:
//   * peak         -- memory region shown (a level, or "all" = every level);
//                     independent of kernel selection (the dropdown never
//                     auto-changes when you pick a kernel)
//   * selected     -- a Set of isolated kernel names
//   * isolatedRoofs -- Set of memory-roof trace indices isolated via the panel
//                      row or by clicking the slope in the plot

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
  var roofList = document.getElementById("roofline-roof-list");
  var roofCountEl = document.getElementById("roofline-roof-count");
  var showAllRoofsBtn = document.getElementById("roofline-show-all-roofs");

  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var computeOverlayTraces = model.computeOverlayTraces || [];
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
    // The region dropdown alone decides which levels show: "All peaks" shows
    // every memory level, a specific region shows one dot. Kernel selection is
    // independent -- it only filters which kernels are drawn.
    if (state.peak === "all") {
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

  // The full-width compute ceilings stay put; the highlight overlays are shown
  // (and re-sampled) only while isolating, spanning from the leftmost isolated
  // slope's ridge rightward, so the cap continues left (dimmed base) but is
  // highlighted past the isolated slope.
  function updateCeilings() {
    if (!gd || typeof Plotly === "undefined" || !computeOverlayTraces.length) {
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
        var pts = logspace(left, Math.max(ceilingDenseHi, left), roofSamples);
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

  // Isolate the clicked memory roof(s) by dimming the others. The full-width
  // compute ceilings dim too, while their highlight overlays (updateCeilings)
  // carry the bright cap from the isolated slope's ridge rightward.
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
      opacities.push(isolating ? 0.15 : 1);
    });
    if (indices.length) {
      Plotly.restyle(gd, { opacity: opacities }, indices);
    }
    updateCeilings();
  }

  // Isolate a memory roof: plain click = only this one (click the sole isolated
  // roof again to clear), Ctrl/Cmd-click = add/remove from the set. Shared by
  // the roofline panel rows and by clicking a slope in the plot.
  function isolateRoof(traceIndex, multi) {
    if (memoryRoofIndices.indexOf(traceIndex) < 0) {
      return;
    }
    if (multi) {
      if (state.isolatedRoofs.has(traceIndex)) {
        state.isolatedRoofs.delete(traceIndex);
      } else {
        state.isolatedRoofs.add(traceIndex);
      }
    } else if (
      state.isolatedRoofs.size === 1 &&
      state.isolatedRoofs.has(traceIndex)
    ) {
      state.isolatedRoofs.clear();
    } else {
      state.isolatedRoofs.clear();
      state.isolatedRoofs.add(traceIndex);
    }
    applyRoofIsolation();
    updateRoofPanel();
  }

  // Reflect isolation state and the active "(AI axis)" region in the roofline
  // panel rows, plus the "(N) - M selected" count and the reset button.
  function updateRoofPanel() {
    var isolating = state.isolatedRoofs.size > 0;
    if (roofList) {
      Array.prototype.forEach.call(roofList.children, function (item) {
        var idx = Number(item.dataset.trace);
        var isolated = state.isolatedRoofs.has(idx);
        item.classList.toggle("selected", isolated);
        item.classList.toggle("dimmed", isolating && !isolated);
        var aiaxis = item.querySelector(".roofline-roof-aiaxis");
        if (aiaxis) {
          // Marks the region whose AI is on the x-axis; blank under "All peaks".
          aiaxis.textContent =
            item.dataset.level === state.peak ? "(AI axis)" : "";
        }
      });
    }
    if (roofCountEl) {
      var total = rooflineTraces.length;
      roofCountEl.textContent = isolating
        ? "(" + total + ") \u2014 " + state.isolatedRoofs.size + " selected"
        : "(" + total + ")";
    }
    if (showAllRoofsBtn) {
      showAllRoofsBtn.disabled = !isolating;
    }
  }

  function render() {
    if (!gd || typeof Plotly === "undefined" || !kernelTraceIndices.length) {
      updatePanel();
      updateRoofPanel();
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
      // With "All peaks" and exactly one kernel shown, color each dot by its
      // memory level (levels are distinguishable); otherwise color by kernel.
      var colorByLevel =
        state.peak === "all" &&
        state.selected.size === 1 &&
        state.selected.has(kernel.name);
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
    updateRoofPanel();
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
    // "All peaks" is a first-class, always-available option (view a kernel
    // across every memory level). The dropdown is independent of kernel
    // selection -- clicking a kernel never changes it.
    var allEl = document.createElement("option");
    allEl.value = "all";
    allEl.textContent = "All peaks";
    peakSelect.appendChild(allEl);
    peakSelect.value = state.peak;
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

  function buildRoofPanel() {
    if (!roofList) {
      return;
    }
    rooflineTraces.forEach(function (roof) {
      var item = document.createElement("li");
      item.className = "roofline-kernel-item";
      item.dataset.trace = String(roof.traceIndex);
      item.dataset.level = roof.level;
      item.title = roof.level;

      var swatch = document.createElement("span");
      swatch.className = "roofline-swatch roofline-roof-swatch";
      swatch.style.backgroundColor = peakColors[roof.level] || "#888888";

      var name = document.createElement("span");
      name.className = "roofline-kernel-name";
      name.textContent = roof.level;

      var aiaxis = document.createElement("span");
      aiaxis.className = "roofline-roof-aiaxis";

      item.appendChild(swatch);
      item.appendChild(name);
      item.appendChild(aiaxis);
      item.addEventListener("click", function (event) {
        isolateRoof(roof.traceIndex, event && (event.ctrlKey || event.metaKey));
      });
      roofList.appendChild(item);
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
        // Clicking a roof slope isolates it, same as its panel row.
        if (memoryRoofIndices.indexOf(traceIndex) >= 0) {
          isolateRoof(
            traceIndex,
            data.event && (data.event.ctrlKey || data.event.metaKey)
          );
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
    buildRoofPanel();
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
