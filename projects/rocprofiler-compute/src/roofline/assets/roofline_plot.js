// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline. It reads the embedded
// JSON model (#roofline-model), then keeps this state:
//   * peak         -- memory region for the aggregate view (default HBM); a
//                     single isolated kernel instead shows every level
//   * selected     -- a Set of isolated kernel names
//   * isolatedRoof -- trace index of the roof isolated in the legend, or null

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
  var autoZoomToggle = document.getElementById("roofline-auto-zoom");
  var runtimeSlider = document.getElementById("roofline-runtime-threshold");
  var runtimeValueEl = document.getElementById("roofline-runtime-value");
  var runtimeFilterEl = document.getElementById("roofline-runtime-filter");

  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var roofMaxAi = model.roofMaxAi || 1e150;
  var peakColors = model.peakColors || {};

  // Whether any kernel carries a percent-of-runtime, which gates the filter.
  var hasRuntimeData = kernels.some(function (kernel) {
    return kernel.pctRuntime != null && isFinite(kernel.pctRuntime);
  });
  // Names of the kernels within the current cumulative runtime threshold
  var thresholdSet = null;

  // Every non-kernel legend trace (memory roofs + compute ceilings). Clicking
  // one in the legend isolates it (dims the rest) rather than hiding it.
  var roofTraceIndices = rooflineTraces
    .map(function (roof) {
      return roof.traceIndex;
    })
    .concat(
      computeTraces.map(function (ceiling) {
        return ceiling.traceIndex;
      })
    );

  var state = {
    // The memory region shown in the aggregate (multi-kernel) view; defaults to
    // HBM. A single isolated kernel ignores this and shows every level.
    peak: model.defaultPeak || "HBM",
    selected: new Set(),
    // Trace index of the roof currently isolated in the legend, or null.
    isolatedRoof: null,
    // Cumulative percent of GPU resident time to display (100 = every kernel).
    runtimeThreshold: runtimeSlider ? Number(runtimeSlider.value) : 100,
    // Whether the view auto-recenters on filter changes; synced to the toggle.
    autoZoom: !autoZoomToggle || autoZoomToggle.checked,
  };

  // The first fit (page open) snaps instantly; later re-fits animate.
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

  // Isolate the clicked roof by dimming every other roof/ceiling; when nothing
  // is isolated all roofs render at full opacity.
  function applyRoofIsolation() {
    if (!gd || typeof Plotly === "undefined" || !roofTraceIndices.length) {
      return;
    }
    var opacities = roofTraceIndices.map(function (idx) {
      if (state.isolatedRoof === null) {
        return 1;
      }
      return idx === state.isolatedRoof ? 1 : 0.15;
    });
    Plotly.restyle(gd, { opacity: opacities }, roofTraceIndices);
  }

  function roofTraceShown(roof) {
    var trace = gd.data && gd.data[roof.traceIndex];
    return !!trace && trace.visible !== false && trace.visible !== "legendonly";
  }

  function snapCeilings(pendingToggleIndex) {
    if (!gd || typeof Plotly === "undefined" || !computeTraces.length) {
      return;
    }
    var visibleBw = rooflineTraces
      .filter(function (roof) {
        var shown = roofTraceShown(roof);
        if (roof.traceIndex === pendingToggleIndex) {
          return !shown;
        }
        return shown;
      })
      .map(function (roof) {
        return roof.bandwidth;
      })
      .filter(function (bw) {
        return bw > 0;
      });
    // Every diagonal hidden: fall back to the steepest overall so the ceilings
    // still get a sensible left endpoint.
    if (!visibleBw.length) {
      visibleBw = rooflineTraces
        .map(function (roof) {
          return roof.bandwidth;
        })
        .filter(function (bw) {
          return bw > 0;
        });
    }
    if (!visibleBw.length) {
      return;
    }
    var maxBw = Math.max.apply(null, visibleBw);
    var ceilingIndices = [];
    var ceilingX = [];
    computeTraces.forEach(function (ceiling) {
      ceilingIndices.push(ceiling.traceIndex);
      ceilingX.push([ceiling.peakPerf / maxBw, roofMaxAi]);
    });
    Plotly.restyle(gd, { x: ceilingX }, ceilingIndices);
  }

  function render() {
    snapCeilings();
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

  // Recenter the log axes on whatever is currently drawn
  function fitView() {
    if (!gd || typeof Plotly === "undefined") {
      return;
    }
    if (hasFitted && !state.autoZoom) {
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

    // Snap on the first framing; animate every re-fit after that.
    if (!hasFitted) {
      hasFitted = true;
      Plotly.relayout(gd, ranges);
      return;
    }
    Plotly.animate(
      gd,
      { layout: ranges },
      {
        transition: { duration: 350, easing: "cubic-in-out" },
        frame: { duration: 350, redraw: false },
      }
    );
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
    if (autoZoomToggle) {
      autoZoomToggle.addEventListener("change", function () {
        state.autoZoom = autoZoomToggle.checked;
        // Turning it on snaps the view to the current selection immediately.
        fitView();
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
      // Clicking a roof in the legend isolates it (dims the others) instead of
      // hiding it; clicking the isolated roof again clears the isolation.
      gd.on("plotly_legendclick", function (ev) {
        if (!ev || typeof ev.curveNumber !== "number") {
          return false;
        }
        if (roofTraceIndices.indexOf(ev.curveNumber) < 0) {
          return false;
        }
        state.isolatedRoof =
          state.isolatedRoof === ev.curveNumber ? null : ev.curveNumber;
        applyRoofIsolation();
        return false;
      });
      // Double-click is the "show everything" gesture: clear any isolation.
      gd.on("plotly_legenddoubleclick", function () {
        state.isolatedRoof = null;
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
