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
    $("bg-diff-warning").hidden = !(differOff && state.source.live);
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
    text("st-background", bg.strategy + ": " + (bg.ready ? "ready" : "still learning") +
      (bg.rejected ? " (" + bg.rejected + " samples rejected)" : ""));
    text("st-frames", String(stats.frames));
    text("st-frozen", String(stats.frozen));
    text("st-failed", String(stats.failed_closed));
    text("st-over-budget", String(stats.over_budget));
    text("st-dropped", String(source.dropped));
    text("st-fps", timings.fps === null ? "n/a" : timings.fps.toFixed(1) + " fps");
    text("st-timings", ms(timings.detect_mean_ms) + " / " + ms(timings.render_mean_ms) +
      " / " + ms(timings.write_mean_ms) + " (mean)");
    text("st-roi", state.roi ? state.roi.join(", ") + " (x, y, w, h)" : "none");
    text("st-reconfigurations", String(stats.reconfigurations));
    text("st-error", state.last_error ? "Error: " + state.last_error : "None");
    $("raw-figure").hidden = !state.raw_preview;
    showApplied(state.config, state);
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
