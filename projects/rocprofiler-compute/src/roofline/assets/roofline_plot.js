// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline. It reads the embedded
// JSON model (#roofline-model), then keeps two pieces of state:
//
//   * peak      -- which memory roof's points are shown ("all" or a level)
//   * selected  -- a Set of isolated kernel names (empty means "show all")

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

  var kernels = model.kernels || [];
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var peakSymbols = model.peakSymbols || {};

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
  };

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

  function render() {
    if (!gd || typeof Plotly === "undefined" || !kernelTraceIndices.length) {
      updatePanel();
      return;
    }

    var xs = [];
    var ys = [];
    var symbols = [];
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
      visibility.push(visible && points.length > 0);
    });

    Plotly.restyle(
      gd,
      { x: xs, y: ys, "marker.symbol": symbols, visible: visibility },
      kernelTraceIndices
    );
    updatePanel();
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
    kernels.forEach(function (kernel, index) {
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

  function updateDetails() {
    if (!detailsEl) {
      return;
    }
    detailsEl.innerHTML = "";
    if (state.selected.size !== 1) {
      return;
    }
    var name = state.selected.values().next().value;
    var kernel = kernels.find(function (candidate) {
      return candidate.name === name;
    });
    if (!kernel) {
      return;
    }

    var heading = document.createElement("div");
    heading.className = "roofline-details-name";
    heading.textContent = kernel.name;
    detailsEl.appendChild(heading);

    var points = pointsForCurrentPeak(kernel);
    if (!points.length) {
      return;
    }

    var table = document.createElement("table");
    table.className = "roofline-details-table";
    var head = document.createElement("tr");
    // Leading empty header is the marker-symbol column.
    ["", "Peak", "AI", "Perf", "Bound"].forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      head.appendChild(th);
    });
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
    detailsEl.appendChild(table);
  }

  function wireEvents() {
    if (peakSelect) {
      peakSelect.addEventListener("change", function () {
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

  function init() {
    buildPeakOptions();
    buildKernelPanel();
    whenPlotReady(function () {
      wireEvents();
      render();
    }, 40);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
