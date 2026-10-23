/* Controlled browser-event order: no decoder/GPU or timing assumptions. */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

test("a forward decode completing after a backward scrub is hidden before the next frame", () => {
  const nextFrames = [];
  const caption = { textContent: "" },
    message = { textContent: "", hidden: false };
  let video;
  const stage = {
    firstChild: null,
    querySelector(selector) {
      return selector === ".media-message" ? message : caption;
    },
    insertBefore(media) {
      video = media;
      media.isConnected = true;
    },
  };
  const card = { querySelector: () => stage };
  const document = {
    createElement(tag) {
      let time = 0;
      const classes = new Set(),
        listeners = new Map();
      return {
        tagName: tag.toUpperCase(),
        dataset: {},
        readyState: 0,
        seeking: false,
        isConnected: false,
        classList: {
          add: (name) => classes.add(name),
          remove: (name) => classes.delete(name),
          contains: (name) => classes.has(name),
        },
        get currentTime() {
          return time;
        },
        set currentTime(value) {
          time = value;
          this.seeking = true;
          this.readyState = 1;
        },
        addEventListener(name, handler) {
          listeners.set(name, handler);
        },
        remove() {
          this.isConnected = false;
        },
        dispatch(name) {
          listeners.get(name)?.();
        },
        completeSeek() {
          this.seeking = false;
          this.readyState = 2;
          this.dispatch("seeked");
        },
      };
    },
  };
  const context = {
    window: {},
    document,
    requestAnimationFrame: (fn) => nextFrames.push(fn),
  };
  vm.runInNewContext(
    fs.readFileSync(require.resolve("./replay-media.js"), "utf8"),
    context,
  );
  const asset = { media_id: "own", mime: "video/webm" };
  const frames = [
    { frame_id: "f0", media_time_s: 0, relative_s: 0 },
    { frame_id: "f4", media_time_s: 0.4, relative_s: 0.4 },
    { frame_id: "f30", media_time_s: 3, relative_s: 3 },
  ];
  const update = (target, cursor) =>
    context.window.ReplayMedia.update(
      card,
      asset,
      target,
      "",
      { frames, cursor, ended: false },
      1,
    );
  update(0, 0);
  video.readyState = 2;
  video.dispatch("loadeddata");
  assert.equal(video.dataset.presentedFrame, "f0");
  update(3.0001, 3.2); // In-flight decode; f0 is still the last visible frame.
  update(0.4001, 0.4); // Backward cursor while old forward decode is pending.
  assert.equal(video.classList.contains("pending"), false);
  video.completeSeek(); // Old 3 s frame becomes drawable before next RAF.
  assert.equal(
    video.classList.contains("pending"),
    true,
    "future decoded frame must be hidden immediately",
  );
  assert.equal(
    video.dataset.presentedFrame,
    "f0",
    "do not relabel a future decode as presented",
  );
  assert.match(message.textContent, /游标/);
  nextFrames.shift()();
  video.completeSeek();
  assert.equal(video.classList.contains("pending"), false);
  assert.equal(video.dataset.presentedFrame, "f4");
});

test("disposing a card releases decoder resources and late events cannot restart it", () => {
  const elements = [],
    callbacks = [];
  const caption = { textContent: "" },
    message = { textContent: "", hidden: false };
  const stage = {
    querySelector: (selector) =>
      selector === ".frame-meta" ? caption : message,
    insertBefore(media) {
      media.isConnected = true;
    },
  };
  const card = { querySelector: () => stage };
  const document = {
    createElement() {
      const listeners = new Map();
      const media = {
        tagName: "VIDEO",
        dataset: {},
        readyState: 0,
        currentTime: 0,
        classList: { add() {}, remove() {} },
        addEventListener: (name, fn) => listeners.set(name, fn),
        dispatch(name) {
          listeners.get(name)?.();
        },
        pause() {
          this.paused = true;
        },
        removeAttribute(name) {
          delete this[name];
        },
        load() {
          this.released = true;
        },
        remove() {
          this.isConnected = false;
        },
      };
      elements.push(media);
      return media;
    },
  };
  const context = {
    window: {},
    document,
    requestAnimationFrame: (fn) => callbacks.push(fn),
  };
  vm.runInNewContext(
    fs.readFileSync(require.resolve("./replay-media.js"), "utf8"),
    context,
  );
  const media = context.window.ReplayMedia;
  const asset = { media_id: "recorded", mime: "video/webm" };
  const frames = [{ frame_id: "f0", media_time_s: 0, relative_s: 0 }];
  media.update(card, asset, 0, "", { frames, cursor: 0 }, 1);
  const old = elements[0];
  old.readyState = 2;
  old.dispatch("loadeddata");
  media.dispose(card);
  assert.equal(old.paused, true);
  assert.equal(old.released, true);
  assert.equal(old.src, undefined);
  assert.equal(old.isConnected, false);
  const previousCaption = caption.textContent;
  old.dispatch("seeked");
  assert.equal(callbacks.length, 0);
  assert.equal(caption.textContent, previousCaption);
  media.update(card, asset, 0, "", { frames, cursor: 0 }, 1);
  assert.equal(elements.length, 2);
});
