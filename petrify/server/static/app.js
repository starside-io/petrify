/* petrify run: local web page. Vanilla JS, no build step, no external resources. */
(function () {
  "use strict";

  var POLL_MS = 1000;
  var DEBOUNCE_MS = 400;
  var MAX_POLL_FAILURES = 3;

  var token = readToken();
  var pollTimer = null;
  var pollFailures = 0;
  var seenState = false;
  var ended = false;
  var streamsStarted = false;
  var debounceTimers = {};
  var editing = {};

  function $(id) {
    return document.getElementById(id);
  }

  function readToken() {
    var hash = window.location.hash.replace(/^#/, "");
    var value = null;
    hash.split("&").forEach(function (part) {
      var pair = part.split("=");
      if (pair[0] === "token" && pair.length > 1) {
        value = decodeURIComponent(pair.slice(1).join("="));
      }
    });
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    return value;
  }

  function api(method, path, body) {
    var options = {
      method: method,
      headers: { "Authorization": "Bearer " + token },
      cache: "no-store",
      credentials: "omit",
      referrerPolicy: "no-referrer"
    };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    return fetch(path, options).then(function (response) {
      return response.json().catch(function () {
        return {};
      }).then(function (data) {
        return { status: response.status, data: data };
      });
    });
  }

  function text(id, value) {
    var el = $(id);
    if (el) {
      el.textContent = value;
    }
  }

  function ms(value) {
    return value === null || value === undefined ? "n/a" : value.toFixed(1) + " ms";
  }

  function get(config, field) {
    if (field === "background_diff.enabled") {
      return config.background_diff !== null;
    }
    if (field === "roi") {
      return config.roi;
    }
    var parts = field.split(".");
    var value = config;
    for (var i = 0; i < parts.length; i += 1) {
      if (value === null || value === undefined) {
        return null;
      }
      value = value[parts[i]];
    }
    return value;
  }

  function describe(field, value) {
    if (field === "roi") {
      if (value.mode === "manual") {
        return "Applied: manual " + value.box.join(", ");
      }
      return "Applied: " + value.mode;
    }
    if (typeof value === "boolean") {
      return "Applied: " + (value ? "on" : "off");
    }
    if (value === null || value === undefined) {
      return field === "engine.frame_budget_ms" ? "Applied: none" : "Applied: n/a (differ off)";
    }
    return "Applied: " + value;
  }

  function controls() {
    return Array.prototype.slice.call(document.querySelectorAll("[data-field]"));
  }

  function showApplied(config, state) {
    controls().forEach(function (el) {
      var field = el.getAttribute("data-field");
      var value = get(config, field);
      text("applied-" + el.id, describe(field, value));
      if (editing[el.id]) {
        return;
      }
      if (field === "roi") {
        el.value = value.mode;
        var box = value.mode === "manual" ? value.box : (state.roi || null);
        document.querySelectorAll("[data-roi-part]").forEach(function (part) {
          var index = Number(part.getAttribute("data-roi-part"));
          if (document.activeElement !== part && box) {
            part.value = box[index];
          }
        });
      } else if (el.type === "checkbox") {
        el.checked = Boolean(value);
      } else if (document.activeElement !== el) {
        el.value = value === null || value === undefined ? "" : value;
      }
    });
    var differOff = config.background_diff === null;
    ["ctl-background-diff-pixel-threshold", "ctl-background-diff-area-threshold"].forEach(function (id) {
      $(id).disabled = differOff || ended;
    });
    showPersonDetector(state.person_detector === "dnn");
  }

  var PERSON_CONTROLS = ["ctl-person-min-confidence", "ctl-person-min-box-area", "ctl-person-scale"];

  function showPersonDetector(dnn) {
    var name = dnn ? "DNN" : "HOG";
    text("opt-detector-person", "Person (" + name + ")");
    text("person-legend", "Person detector (" + name + ")");
    $("person-dnn-note").hidden = !dnn;
    PERSON_CONTROLS.forEach(function (id) {
      $(id).disabled = dnn || ended;
    });
  }

  var shownWeakened = null;

  function showWeakened(list) {
    var items = Array.isArray(list) ? list : [];
    var key = items.join("\n");
    if (key === shownWeakened) {
      return; /* Unchanged: the alert is not announced again. */
    }
    shownWeakened = key;
    var ul = $("weakened-list");
    while (ul.firstChild) {
      ul.removeChild(ul.firstChild);
    }
    items.forEach(function (sentence) {
      var li = document.createElement("li");
      li.textContent = String(sentence);
      ul.appendChild(li);
    });
    $("weakened").hidden = items.length === 0;
  }

  /* The mode section. Every string in state.mode and state.modes comes from a
     plugin or from package metadata: it is only ever set with textContent or as
     a property value. Option names are data: controls are kept in an array in
     the order the server lists them, never looked up or keyed by name. */
  var modeShape = null;
  var modeEntries = [];
  var shownModeList = null;

  function clear(el) {
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function element(tag, className, content) {
    var el = document.createElement(tag);
    if (className) {
      el.className = className;
    }
    if (content !== undefined) {
      el.textContent = String(content);
    }
    return el;
  }

  function optionText(value) {
    if (typeof value === "boolean") {
      return value ? "on" : "off";
    }
    if (Array.isArray(value)) {
      return value.length ? value.map(String).join(", ") : "none";
    }
    if (value === null || value === undefined) {
      return "n/a";
    }
    return String(value);
  }

  function isNumber(value) {
    return typeof value === "number" && isFinite(value);
  }

  function rangeText(opt) {
    var low = isNumber(opt.minimum) ? opt.minimum : null;
    var high = isNumber(opt.maximum) ? opt.maximum : null;
    if (low !== null && high !== null) {
      return low + " to " + high;
    }
    if (low !== null) {
      return "at least " + low;
    }
    if (high !== null) {
      return "at most " + high;
    }
    return "";
  }

  function optionShape(options) {
    return JSON.stringify(options.map(function (opt) {
      return [opt.name, opt.kind, opt.runtime, opt.choices, opt.minimum, opt.maximum, opt.help];
    }));
  }

  function choiceList(opt) {
    return Array.isArray(opt.choices) ? opt.choices.map(String) : [];
  }

  function buildOption(opt, index) {
    var entry = { opt: opt, inputs: [], editing: false, timer: null };
    var id = "mode-opt-" + index;
    var kind = String(opt.kind);
    var runtime = opt.runtime === true;
    var labelText = String(opt.name) + (runtime ? "" : " (fixed for this run)");
    var range = rangeText(opt);
    if (range && (kind === "int" || kind === "float")) {
      labelText += " (" + range + ")";
    }
    var wrap;
    if (kind === "choices") {
      wrap = element("fieldset");
      wrap.appendChild(element("legend", "", labelText));
      choiceList(opt).forEach(function (choice, j) {
        var row = element("div", "field field-check");
        var box = element("input");
        box.type = "checkbox";
        box.id = id + "-" + j;
        box.value = choice;
        var label = element("label", "", choice);
        label.htmlFor = box.id;
        row.appendChild(box);
        row.appendChild(label);
        wrap.appendChild(row);
        entry.inputs.push(box);
      });
    } else {
      wrap = element("div", kind === "bool" ? "field field-check" : "field");
      var input;
      if (kind === "bool") {
        input = element("input");
        input.type = "checkbox";
      } else if (kind === "choice") {
        input = element("select");
        choiceList(opt).forEach(function (choice) {
          var choiceEl = element("option", "", choice);
          choiceEl.value = choice;
          input.appendChild(choiceEl);
        });
      } else {
        input = element("input");
        input.type = "number";
        input.step = kind === "int" ? "1" : "any";
        input.inputMode = kind === "int" ? "numeric" : "decimal";
        if (isNumber(opt.minimum)) {
          input.min = String(opt.minimum);
        }
        if (isNumber(opt.maximum)) {
          input.max = String(opt.maximum);
        }
      }
      input.id = id;
      var mainLabel = element("label", "", labelText);
      mainLabel.htmlFor = id;
      if (kind === "bool") {
        wrap.appendChild(input);
        wrap.appendChild(mainLabel);
      } else {
        wrap.appendChild(mainLabel);
        wrap.appendChild(input);
      }
      entry.inputs.push(input);
    }
    entry.help = element("span", "help", opt.help ? String(opt.help) : "");
    entry.help.id = id + "-help";
    entry.applied = element("span", "applied");
    entry.error = element("span", "error");
    entry.error.id = id + "-error";
    entry.error.setAttribute("aria-live", "polite");
    wrap.appendChild(entry.help);
    wrap.appendChild(entry.applied);
    wrap.appendChild(entry.error);
    entry.inputs.forEach(function (input) {
      input.setAttribute("data-mode-option", "");
      input.setAttribute("aria-describedby", entry.help.id + " " + entry.error.id);
      input.disabled = !runtime || ended;
      var eventName = input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input";
      input.addEventListener(eventName, function () {
        scheduleOption(entry);
      });
    });
    entry.root = wrap;
    return entry;
  }

  function setOption(entry, value) {
    var kind = entry.opt.kind;
    var first = entry.inputs[0];
    if (kind === "bool") {
      first.checked = value === true;
    } else if (kind === "choices") {
      var picked = Array.isArray(value) ? value.map(String) : [];
      entry.inputs.forEach(function (box) {
        box.checked = picked.indexOf(box.value) !== -1;
      });
    } else if (first && document.activeElement !== first) {
      first.value = value === null || value === undefined ? "" : String(value);
    }
  }

  function readOption(entry) {
    var kind = entry.opt.kind;
    var first = entry.inputs[0];
    if (kind === "bool") {
      return { ok: true, value: first.checked };
    }
    if (kind === "choice") {
      return { ok: true, value: first.value };
    }
    if (kind === "choices") {
      return {
        ok: true,
        value: entry.inputs.filter(function (box) {
          return box.checked;
        }).map(function (box) {
          return box.value;
        })
      };
    }
    return numberValue(first, kind === "int" ? "int" : "float");
  }

  function sendOption(entry) {
    var read = readOption(entry);
    if (!read.ok) {
      entry.error.textContent = "Not sent: " + read.message;
      entry.editing = false;
      return;
    }
    /* A prototype-less object with the name defined as an own property: a name
       such as __proto__ or constructor stays an ordinary key. */
    var values = Object.create(null);
    Object.defineProperty(values, String(entry.opt.name), {
      value: read.value,
      enumerable: true
    });
    entry.error.textContent = "Sending...";
    api("PATCH", "/api/config", { mode_options: values }).then(function (result) {
      entry.editing = false;
      if (result.status === 200) {
        entry.error.textContent = "";
      } else {
        entry.error.textContent = "Rejected: " + String(result.data.error || "the change was not accepted.");
      }
    }).catch(function () {
      entry.editing = false;
      entry.error.textContent = "Not sent: petrify could not be reached.";
    });
  }

  function scheduleOption(entry) {
    if (ended || entry.opt.runtime !== true) {
      return;
    }
    entry.editing = true;
    if (entry.timer) {
      window.clearTimeout(entry.timer);
    }
    entry.timer = window.setTimeout(function () {
      entry.timer = null;
      sendOption(entry);
    }, DEBOUNCE_MS);
  }

  function showModeOptions(options) {
    var shape = optionShape(options);
    if (shape !== modeShape) {
      modeShape = shape;
      modeEntries.forEach(function (entry) {
        if (entry.timer) {
          window.clearTimeout(entry.timer);
        }
      });
      var box = $("mode-options");
      clear(box);
      modeEntries = options.map(buildOption);
      modeEntries.forEach(function (entry) {
        box.appendChild(entry.root);
      });
    }
    $("mode-no-options").hidden = options.length !== 0;
    $("mode-fixed-note").hidden = !options.some(function (opt) {
      return opt.runtime !== true;
    });
    options.forEach(function (opt, i) {
      var entry = modeEntries[i];
      entry.applied.textContent = "Applied: " + optionText(opt.value) +
        " (default: " + optionText(opt.default) + ")";
      if (!entry.editing) {
        setOption(entry, opt.value);
      }
    });
  }

  function showModeList(modes) {
    var listed = modes && Array.isArray(modes.listed) ? modes.listed : [];
    var key = JSON.stringify(modes || null);
    if (key === shownModeList) {
      return;
    }
    shownModeList = key;
    var ul = $("mode-list");
    clear(ul);
    /* Status first, then the name and its origin in their own elements, so a
       name cannot pass for a status or for another entry. */
    listed.forEach(function (item) {
      var status;
      if (item.active === true) {
        status = "Active in this run:";
      } else if (item.selectable === true) {
        status = "Installed, selectable for a new run:";
      } else {
        status = "Cannot be selected:";
      }
      var li = element("li", "mode-entry");
      li.appendChild(element("span", "mode-status", status));
      li.appendChild(element("code", "mode-name", String(item.name)));
      li.appendChild(element("span", "mode-origin",
        item.builtin === true ? "built in" : "plugin " + String(item.origin)));
      if (item.active !== true && item.selectable !== true) {
        li.appendChild(element("span", "mode-problem", "Reason: " + String(item.problem)));
      }
      ul.appendChild(li);
    });
    var omitted = modes && isNumber(modes.omitted) ? modes.omitted : 0;
    var more = $("mode-list-omitted");
    more.textContent = omitted ? omitted + " more installed modes are not listed." : "";
    more.hidden = !omitted;
  }

  function showMode(mode, modes) {
    if (!mode || typeof mode !== "object") {
      return;
    }
    $("mode-section").hidden = false;
    var builtin = mode.builtin === true;
    text("mode-name", String(mode.name));
    text("mode-origin", builtin ? "Built in" : "Plugin " + String(mode.origin));
    text("mode-summary", mode.summary ? String(mode.summary) : "None given");
    $("mode-plugin-notice").hidden = builtin;
    showModeOptions(Array.isArray(mode.options) ? mode.options : []);
    showModeList(modes);
  }

  /* One line per detector and refiner, in run order. Lines are reused and only
     their text changes, so a poll does not rebuild them. A line no longer needed
     stays as a blank line, so a shorter list after a settings change does not
     move the rows below it. */
  function showDetectorTimes(entries) {
    var ul = $("st-detectors");
    if (!ul) {
      return;
    }
    var lines = [];
    if (Array.isArray(entries)) {
      entries.forEach(function (entry) {
        if (entry && typeof entry === "object") {
          lines.push(String(entry.name) + ": " + ms(entry.mean_ms) + " mean, " +
            ms(entry.last_ms) + " last");
        }
      });
    }
    if (!lines.length) {
      lines.push("None");
    }
    while (ul.children.length < lines.length) {
      ul.appendChild(element("li", "timing-line"));
    }
    for (var i = 0; i < ul.children.length; i += 1) {
      /* A no-break space keeps an unused line one line tall and shows nothing. */
      ul.children[i].textContent = i < lines.length ? lines[i] : "\u00a0";
    }
  }

  function render(state) {
    var stats = state.stats;
    var timings = state.timings;
    var source = state.source;
    var bg = state.background;
    var runText = state.ended ? "Ended" : (state.running ? "Running" : "Starting");
    text("run-state", "Run: " + runText);
    text("st-running", runText);
    text("st-source", (source.uri || "unknown") + ", " + source.width + "x" + source.height +
      " at " + source.fps.toFixed(2) + " fps" + (source.live ? ", live" : ", file or folder"));
    if (bg.in_use === false) {
      text("st-background", "Not used: this mode does not use a background");
    } else {
      text("st-background", bg.strategy + ": " + (bg.ready ? "ready" : "still learning") +
        (bg.rejected ? " (" + bg.rejected + " samples rejected)" : ""));
    }
    text("st-frames", String(stats.frames));
    text("st-frozen", String(stats.frozen));
    text("st-passed", String(stats.passed_through));
    text("st-failed", String(stats.failed_closed));
    text("st-over-budget", String(stats.over_budget));
    text("st-dropped", String(source.dropped));
    text("st-fps", timings.fps === null ? "n/a" : timings.fps.toFixed(1) + " fps");
    text("st-timings", ms(timings.detect_mean_ms) + " / " + ms(timings.render_mean_ms) +
      " / " + ms(timings.write_mean_ms) + " (mean)");
    showDetectorTimes(timings.detectors);
    text("st-roi", state.roi ? state.roi.join(", ") + " (x, y, w, h)" : "none");
    text("st-reconfigurations", String(stats.reconfigurations));
    text("st-error", state.last_error ? "Error: " + state.last_error : "None");
    $("raw-figure").hidden = !state.raw_preview;
    showWeakened(state.weakened);
    showApplied(state.config, state);
    showMode(state.mode, state.modes);
    if (!streamsStarted && !state.ended) {
      streamsStarted = true;
      $("stream-processed").src = "/stream.mjpg?token=" + encodeURIComponent(token);
      if (state.raw_preview) {
        $("stream-raw").src = "/raw.mjpg?token=" + encodeURIComponent(token);
      }
    }
    if (state.ended) {
      end("The run has ended. The feed and the settings are no longer live.");
    }
  }

  function end(message) {
    if (ended) {
      return;
    }
    ended = true;
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    ["stream-processed", "stream-raw"].forEach(function (id) {
      $(id).removeAttribute("src");
    });
    var banner = $("ended-banner");
    banner.textContent = message;
    banner.hidden = false;
    text("run-state", "Run: Ended");
    text("st-running", "Ended");
    controls().forEach(function (el) {
      el.disabled = true;
    });
    document.querySelectorAll("[data-roi-part]").forEach(function (el) {
      el.disabled = true;
    });
    document.querySelectorAll("[data-mode-option]").forEach(function (el) {
      el.disabled = true;
    });
    $("stop-button").disabled = true;
  }

  function poll() {
    api("GET", "/api/state").then(function (result) {
      if (result.status !== 200) {
        throw new Error("status " + result.status);
      }
      pollFailures = 0;
      seenState = true;
      render(result.data);
    }).catch(function () {
      pollFailures += 1;
      if (seenState && pollFailures >= MAX_POLL_FAILURES) {
        end("The run has ended (the page can no longer reach petrify).");
      } else if (!seenState) {
        text("run-state", "Cannot reach petrify. Check that the run is still going and that the link is complete.");
      }
    });
  }

  function numberValue(el, kind) {
    var raw = el.value.trim();
    if (kind === "float-or-null" && raw === "") {
      return { ok: true, value: null };
    }
    if (raw === "") {
      return { ok: false, message: "Enter a value." };
    }
    var value = Number(raw);
    if (!isFinite(value)) {
      return { ok: false, message: "Enter a number." };
    }
    if (kind === "int" && Math.floor(value) !== value) {
      return { ok: false, message: "Enter a whole number." };
    }
    return { ok: true, value: value };
  }

  function patchFor(el) {
    var field = el.getAttribute("data-field");
    var kind = el.getAttribute("data-type");
    var value;
    if (field === "detector") {
      value = el.value;
    } else if (field === "roi") {
      if (el.value === "disabled") {
        value = "disabled";
      } else if (el.value === "manual") {
        var box = [];
        var parts = document.querySelectorAll("[data-roi-part]");
        for (var i = 0; i < parts.length; i += 1) {
          var part = numberValue(parts[i], "int");
          if (!part.ok) {
            return { ok: false, message: "Box: " + part.message };
          }
          box.push(part.value);
        }
        value = box;
      } else {
        return { ok: false, message: "Choose disabled or a manual box." };
      }
    } else if (kind === "bool") {
      value = el.checked;
    } else {
      var parsed = numberValue(el, kind);
      if (!parsed.ok) {
        return parsed;
      }
      value = parsed.value;
    }
    if (field === "detector" || field === "roi") {
      var top = {};
      top[field] = value;
      return { ok: true, patch: top };
    }
    var path = field.split(".");
    var section = {};
    section[path[1]] = value;
    var patch = {};
    patch[path[0]] = section;
    return { ok: true, patch: patch };
  }

  function send(el) {
    var errorId = "err-" + el.id;
    var built = patchFor(el);
    if (!built.ok) {
      text(errorId, "Not sent: " + built.message);
      editing[el.id] = false;
      return;
    }
    text(errorId, "Sending...");
    api("PATCH", "/api/config", built.patch).then(function (result) {
      editing[el.id] = false;
      if (result.status === 200) {
        text(errorId, "");
      } else {
        text(errorId, "Rejected: " + (result.data.error || "the change was not accepted."));
      }
    }).catch(function () {
      editing[el.id] = false;
      text(errorId, "Not sent: petrify could not be reached.");
    });
  }

  function schedule(el) {
    if (ended) {
      return;
    }
    editing[el.id] = true;
    if (debounceTimers[el.id]) {
      window.clearTimeout(debounceTimers[el.id]);
    }
    debounceTimers[el.id] = window.setTimeout(function () {
      debounceTimers[el.id] = null;
      send(el);
    }, DEBOUNCE_MS);
  }

  function wire() {
    controls().forEach(function (el) {
      var eventName = el.tagName === "SELECT" || el.type === "checkbox" ? "change" : "input";
      el.addEventListener(eventName, function () {
        schedule(el);
      });
    });
    var roi = $("ctl-roi");
    document.querySelectorAll("[data-roi-part]").forEach(function (part) {
      part.addEventListener("input", function () {
        roi.value = "manual";
        schedule(roi);
      });
    });
    $("settings").addEventListener("submit", function (event) {
      event.preventDefault();
    });
    $("stop-button").addEventListener("click", function () {
      if (!window.confirm("Stop the run? The outputs are finalized and the run ends.")) {
        return;
      }
      api("POST", "/api/stop").then(function (result) {
        text("err-stop", result.status === 200 ? "Stopping..." : "Rejected: " + (result.data.error || "stop failed."));
      }).catch(function () {
        text("err-stop", "Not sent: petrify could not be reached.");
      });
    });
  }

  function start() {
    if (!token) {
      $("token-missing").hidden = false;
      text("run-state", "No access token");
      return;
    }
    wire();
    poll();
    pollTimer = window.setInterval(poll, POLL_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
