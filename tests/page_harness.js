/*
 * Test helper: runs petrify/server/static/app.js against a minimal stub DOM in
 * Node and prints what the page shows, as JSON, for tests/test_server.py.
 *
 * Usage: node tests/page_harness.js STATIC_DIR SCENARIO_JSON
 *
 * The scenario file holds {"state": <the /api/state payload>,
 * "patch_response": {"status": 422, "body": {...}}, "actions": [...]}. Each
 * action is {"input": N, "value": "7"} or {"input": N, "checked": true}, where N
 * indexes the mode option inputs in page order; the matching input or change
 * event is fired, then every pending timer runs (the debounce). Then, for each
 * entry of the optional "later_states" list, that state becomes the /api/state
 * payload and the page's polling interval runs once. Then the report is printed.
 *
 * The stub records every element id set by the script, every attribute name
 * passed to setAttribute, every href and dataset key written, and every
 * window.open call or location assignment, so a test can check that none of
 * them is built from data. For every static element (one with an id in
 * index.html) it reports whether it is hidden or disabled, and its text.
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const staticDir = process.argv[2];
const scenario = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

const record = { ids: [], attributes: [], hrefs: [], dataset: [], opened: [], locations: [] };

class Element {
  constructor(tag, id) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.attrs = {};
    this.listeners = {};
    this.ownText = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = "";
    this.className = "";
    this.htmlFor = "";
    this.type = "";
    this.min = "";
    this.max = "";
    this.step = "";
    this.inputMode = "";
    this.elementId = id || "";
    const datasetRecord = record.dataset;
    this.dataset = new Proxy({}, {
      set(target, key, value) {
        datasetRecord.push(String(key));
        target[key] = value;
        return true;
      }
    });
  }

  get id() { return this.elementId; }
  set id(value) { this.elementId = String(value); record.ids.push(this.elementId); }
  get href() { return this.attrs.href || ""; }
  set href(value) { record.hrefs.push(String(value)); this.attrs.href = String(value); }
  get firstChild() { return this.children[0] || null; }
  get textContent() { return this.ownText + this.children.map((c) => c.textContent).join(""); }
  set textContent(value) { this.ownText = String(value); this.children = []; }

  appendChild(child) { this.children.push(child); return child; }
  removeChild(child) { this.children = this.children.filter((c) => c !== child); return child; }
  setAttribute(name, value) { record.attributes.push(String(name)); this.attrs[name] = String(value); }
  getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); }
  fire(name) { (this.listeners[name] || []).forEach((fn) => fn({ preventDefault() {} })); }
}

/* Static elements: every element with an id in index.html. */
const staticById = {};
const html = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
for (const match of html.matchAll(/<(\w+)\b([^>]*)>/g)) {
  const idMatch = /\sid="([^"]+)"/.exec(match[2]);
  if (idMatch) {
    const el = new Element(match[1], idMatch[1]);
    el.hidden = /\shidden(\s|$|=)/.test(match[2]);
    staticById[idMatch[1]] = el;
  }
}
const created = [];

const document = {
  readyState: "complete",
  activeElement: null,
  getElementById(id) {
    return staticById[id] || created.find((el) => el.id === id) || null;
  },
  createElement(tag) {
    const el = new Element(tag);
    created.push(el);
    return el;
  },
  querySelectorAll(selector) {
    const match = /^\[([a-z-]+)\]$/.exec(selector);
    if (!match) {
      return [];
    }
    return created.filter((el) => match[1] in el.attrs);
  },
  addEventListener() {}
};

const timers = [];
const intervals = [];
let currentState = scenario.state;
const sent = [];
const location = {
  hash: "#token=harness-token-0123456789",
  pathname: "/",
  search: "",
  assign(url) { record.locations.push(String(url)); },
  replace(url) { record.locations.push(String(url)); }
};
const window = {
  history: { replaceState() {} },
  setTimeout(fn) { timers.push(fn); return timers.length; },
  clearTimeout() {},
  setInterval(fn) { intervals.push(fn); return intervals.length; },
  clearInterval() {},
  confirm() { return false; },
  open(url) { record.opened.push(String(url)); }
};
Object.defineProperty(window, "location", {
  get() { return location; },
  set(value) { record.locations.push(String(value)); }
});

function respond(status, body) {
  return Promise.resolve({ status: status, json: () => Promise.resolve(body) });
}

function fetch(url, options) {
  if (options && options.method === "PATCH") {
    sent.push(JSON.parse(options.body));
    const reply = scenario.patch_response || { status: 200, body: {} };
    return respond(reply.status, reply.body);
  }
  return respond(200, currentState);
}

const context = vm.createContext({
  window, document, fetch, console, JSON, Object, Array, Number, String, Boolean, Map,
  isFinite, Math, Promise, Error, encodeURIComponent, decodeURIComponent
});
vm.runInContext(fs.readFileSync(path.join(staticDir, "app.js"), "utf8"), context);

function flush() {
  return new Promise((resolve) => setImmediate(resolve));
}

function describe(el) {
  return {
    tag: el.tagName,
    className: el.className,
    text: el.textContent,
    children: el.children.map(describe)
  };
}

(async function main() {
  for (let i = 0; i < 5; i += 1) {
    await flush();
  }
  const inputs = () => created.filter((el) => "data-mode-option" in el.attrs);
  for (const action of scenario.actions || []) {
    const input = inputs()[action.input];
    if ("checked" in action) {
      input.checked = action.checked;
      input.fire("change");
    } else {
      input.value = action.value;
      input.fire(input.tagName === "SELECT" ? "change" : "input");
    }
  }
  timers.splice(0).forEach((fn) => fn());
  for (let i = 0; i < 5; i += 1) {
    await flush();
  }
  for (const state of scenario.later_states || []) {
    currentState = state;
    intervals.forEach((fn) => fn());
    for (let i = 0; i < 5; i += 1) {
      await flush();
    }
  }
  const hidden = {};
  const text = {};
  const disabled = {};
  for (const id of Object.keys(staticById)) {
    hidden[id] = staticById[id].hidden;
    text[id] = staticById[id].textContent;
    disabled[id] = staticById[id].disabled;
  }
  const report = {
    hidden: hidden,
    text: text,
    disabled: disabled,
    weakened: staticById["weakened-list"].children.map((li) => li.textContent),
    modeList: staticById["mode-list"].children.map(describe),
    detectorTimes: staticById["st-detectors"].children.map(describe),
    inputs: inputs().map((el) => ({
      tag: el.tagName, type: el.type, id: el.id, disabled: el.disabled, checked: el.checked,
      value: el.value, min: el.min, max: el.max, step: el.step
    })),
    labels: created.filter((el) => el.tagName === "LABEL" || el.tagName === "LEGEND")
      .map((el) => ({ tag: el.tagName, text: el.textContent, htmlFor: el.htmlFor })),
    errors: created.filter((el) => el.className === "error").map((el) => el.textContent),
    sent: sent,
    record: record
  };
  process.stdout.write(JSON.stringify(report));
})().catch((error) => {
  process.stderr.write(String(error && error.stack ? error.stack : error));
  process.exit(1);
});
