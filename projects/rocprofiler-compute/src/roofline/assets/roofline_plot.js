// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline. It reads the embedded
// JSON model (#roofline-model), then keeps two pieces of state:
//   * peak      -- which memory roof's points are shown
//   * selected  -- a Set of isolated kernel names

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
  var detailsEl = document.getElementById("roofline-kernel-details");
  var showAllBtn = document.getElementById("roofline-show-all");
  var kernelCountEl = document.getElementById("roofline-kernel-count");
  var autoZoomToggle = document.getElementById("roofline-auto-zoom");

  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var roofMaxAi = model.roofMaxAi || 1e150;
  var peakSymbols = model.peakSymbols || {};
  var aiUnit = model.aiUnit || "FLOPs/Byte";
  var perfUnit = model.perfUnit || "GFLOP/s";

  var hasTruncatedNames = kernels.some(function (kernel) {
    return kernel.hoverName && kernel.hoverName !== kernel.name;
  });

  // Unicode glyphs approximating the Plotly marker shapes, used in the details
  // table's symbol column so each row shows the same marker as its plot dot.
  var SYMBOL_GLYPHS = {
    circle: "\u25CF",
    square: "\u25A0",
    diamond: "\u25C6",
    cross: "\u271A",
    "triangle-up": "\u25B2",
  };

  var state = {
    peak: model.defaultPeak || "all",
    selected: new Set(),
    // Whether the view auto-recenters on filter changes; synced to the toggle.
    autoZoom: !autoZoomToggle || autoZoomToggle.checked,
  };

  // The first fit (page open) snaps instantly; later re-fits animate.
  var hasFitted = false;

  function peakGlyph(peak) {
    return SYMBOL_GLYPHS[peakSymbols[peak]] || SYMBOL_GLYPHS.circle;
  }

  function kernelIsVisible(kernel) {
    return state.selected.size === 0 || state.selected.has(kernel.name);
  }

  function pointsForCurrentPeak(kernel) {
    var points = kernel.points || [];
    if (state.peak === "all") {
      return points;
    }
    return points.filter(function (point) {
      return point.peak === state.peak;
    });
  }

  function roofIsVisible(roof) {
    return state.peak === "all" || roof.level === state.peak;
  }

  function applyPeakFilterToRoofs() {
    if (!gd || typeof Plotly === "undefined" || !rooflineTraces.length) {
      return;
    }
    var indices = [];
    var visibility = [];
    rooflineTraces.forEach(function (roof) {
      indices.push(roof.traceIndex);
      visibility.push(roofIsVisible(roof) ? true : "legendonly");
    });
    Plotly.restyle(gd, { visible: visibility }, indices);
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
    // Re-snap ceilings to whatever roofs are currently shown, but do NOT reset
    // roof visibility here (that would undo the user's legend toggles).
    snapCeilings();
    if (!gd || typeof Plotly === "undefined" || !kernelTraceIndices.length) {
      updatePanel();
      return;
    }

    var xs = [];
    var ys = [];
    var symbols = [];
    var customdata = [];
    var visibility = [];

    kernels.forEach(function (kernel) {
      var visible = kernelIsVisible(kernel);
      var points = visible ? pointsForCurrentPeak(kernel) : [];
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
      symbols.push(
        points.map(function (point) {
          return peakSymbols[point.peak] || "circle";
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
        "marker.symbol": symbols,
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
    var options = [{ value: "all", label: "All peaks" }];
    (model.peaks || []).forEach(function (peak) {
      options.push({ value: peak, label: peak });
    });
    options.forEach(function (option) {
      var el = document.createElement("option");
      el.value = option.value;
      el.textContent = option.label;
      peakSelect.appendChild(el);
    });
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
    updateDetails();
  }

  function buildKernelDetailBlock(kernel) {
    var block = document.createElement("div");
    block.className = "roofline-detail-block";

    var heading = document.createElement("div");
    heading.className = "roofline-details-name";
    heading.title = kernel.name;

    var swatch = document.createElement("span");
    swatch.className = "roofline-swatch";
    swatch.style.backgroundColor = kernel.color || "#888888";
    heading.appendChild(swatch);

    var nameText = document.createElement("span");
    nameText.className = "roofline-details-name-text";
    // Truncated heading
    nameText.textContent = kernel.hoverName || kernel.name;
    heading.appendChild(nameText);
    block.appendChild(heading);

    var points = pointsForCurrentPeak(kernel);
    if (!points.length) {
      var note = document.createElement("div");
      note.className = "roofline-detail-empty";
      note.textContent = "No points at the selected peak.";
      block.appendChild(note);
      return block;
    }

    var table = document.createElement("table");
    table.className = "roofline-details-table";
    var head = document.createElement("tr");
    // Leading empty header is the marker-symbol column; AI/Perf carry units.
    ["", "Peak", "AI (" + aiUnit + ")", "Perf (" + perfUnit + ")", "Bound"].forEach(
      function (label) {
        var th = document.createElement("th");
        th.textContent = label;
        head.appendChild(th);
      }
    );
    table.appendChild(head);

    points.forEach(function (point) {
      var row = document.createElement("tr");

      // Symbol cell: same marker shape as the plot dot, in the kernel color.
      var symbolCell = document.createElement("td");
      symbolCell.className = "roofline-details-symbol";
      symbolCell.textContent = peakGlyph(point.peak);
      symbolCell.style.color = kernel.color || "#444444";
      row.appendChild(symbolCell);

      [
        point.peak,
        Number(point.ai).toFixed(2),
        Number(point.perf).toFixed(2),
        point.status,
      ].forEach(function (value) {
        var td = document.createElement("td");
        td.textContent = value;
        td.title = value;
        row.appendChild(td);
      });
      table.appendChild(row);
    });
    block.appendChild(table);
    return block;
  }

  function updateDetails() {
    updateLongNameHint();
    if (!detailsEl) {
      return;
    }
    detailsEl.innerHTML = "";
    if (state.selected.size === 0) {
      return;
    }
    kernels.forEach(function (kernel) {
      if (state.selected.has(kernel.name)) {
        detailsEl.appendChild(buildKernelDetailBlock(kernel));
      }
    });
  }

  function updateLongNameHint() {
    var hintEl = document.getElementById("roofline-longname-hint");
    if (!hintEl) {
      return;
    }
    // The tip sits directly above the selected-kernel subtables, whose headings
    // are truncated, so only surface it when a subtable is shown and a name is
    // actually clipped.
    hintEl.hidden = !(hasTruncatedNames && state.selected.size > 0);
  }

  function buildShapeLegend() {
    var legendEl = document.getElementById("roofline-shape-legend");
    if (!legendEl) {
      return;
    }
    var peaks = model.peaks || [];
    if (!peaks.length) {
      legendEl.style.display = "none";
      return;
    }
    // Shape identifies the memory level and is the same for every kernel, so
    // this stays a compact, always-visible reference regardless of how many
    // kernels are selected.
    var caption = document.createElement("span");
    caption.className = "roofline-legend-caption";
    caption.textContent = "Shape = memory level:";
    legendEl.appendChild(caption);

    peaks.forEach(function (peak) {
      var item = document.createElement("span");
      item.className = "roofline-legend-item";

      var glyph = document.createElement("span");
      glyph.className = "roofline-legend-glyph";
      glyph.textContent = peakGlyph(peak);

      var label = document.createElement("span");
      label.textContent = peak;

      item.appendChild(glyph);
      item.appendChild(label);
      legendEl.appendChild(item);
    });
  }

  function wireEvents() {
    if (peakSelect) {
      peakSelect.addEventListener("change", function () {
        state.peak = peakSelect.value;
        // Changing the peak deliberately resets which roofs are shown.
        applyPeakFilterToRoofs();
        render();
      });
    }
    if (showAllBtn) {
      showAllBtn.addEventListener("click", function () {
        state.selected.clear();
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
      // Toggling a roof in the legend hides/shows a diagonal. The event fires
      // before Plotly flips the trace, so snap against the pending post-click
      // state (via curveNumber) instead of the stale live visibility.
      gd.on("plotly_legendclick", function (ev) {
        if (ev && typeof ev.curveNumber === "number") {
          snapCeilings(ev.curveNumber);
        } else {
          snapCeilings();
        }
        return true;
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
    buildShapeLegend();
    buildKernelPanel();
    whenPlotReady(function () {
      wireEvents();
      resizePlot();
      applyPeakFilterToRoofs();
      render();
    }, 40);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
