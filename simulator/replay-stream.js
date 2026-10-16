/* Read-only replay evidence. Transport receipt never becomes a model timestamp. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ReplayStream = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const identifier = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}(?![\s\S])/;
  const eventType = /^[A-Za-z][A-Za-z0-9_.-]{0,127}(?![\s\S])/;
  const sourceKey = /^[a-z][a-z0-9_]*_id(?![\s\S])/;
  const domains = new Set([
    "host_monotonic",
    "cuda_event",
    "media",
    "simulation",
    "utc",
  ]);
  const units = new Set(["s", "ms", "ns"]);
  const collections = Object.freeze({
    observations: "observation_id",
    clean_requests: "id",
    profile_requests: "id",
    retained_outputs: "id",
    episodes: "id",
    kernels: null,
  });
  const own = (object, key) =>
    Object.prototype.hasOwnProperty.call(object, key);
  const object = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value);
  const terminal = (state) => state === "ended" || state === "failed";

  class StreamProtocolError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "StreamProtocolError";
      this.code = code;
    }
  }
  function requireValue(condition, code, message) {
    if (!condition) throw new StreamProtocolError(code, message);
  }
  function sequence(value, zero = false) {
    requireValue(
      Number.isSafeInteger(value) && value >= (zero ? 0 : 1),
      "invalid_sequence",
      "Invalid replay sequence",
    );
  }
  function json(value, parents = new Set()) {
    if (
      value === null ||
      typeof value === "string" ||
      typeof value === "boolean"
    )
      return;
    if (typeof value === "number") {
      requireValue(
        Number.isFinite(value),
        "invalid_json",
        "Replay metadata must be finite JSON",
      );
      return;
    }
    requireValue(
      typeof value === "object" && !parents.has(value),
      "invalid_json",
      "Replay metadata must be acyclic JSON",
    );
    requireValue(
      Array.isArray(value) ||
        Object.getPrototypeOf(value) === Object.prototype ||
        Object.getPrototypeOf(value) === null,
      "invalid_json",
      "Replay metadata must contain only JSON values",
    );
    parents.add(value);
    for (const item of Array.isArray(value) ? value : Object.values(value))
      json(item, parents);
    parents.delete(value);
  }
  function clone(value) {
    json(value);
    return JSON.parse(JSON.stringify(value));
  }
  function freeze(value) {
    if (
      value !== null &&
      typeof value === "object" &&
      !Object.isFrozen(value)
    ) {
      for (const item of Object.values(value)) freeze(item);
      Object.freeze(value);
    }
    return value;
  }
  function canonical(value) {
    if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
    if (object(value))
      return `{${Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
        .join(",")}}`;
    return JSON.stringify(value);
  }
  function validateRun(run, runId) {
    requireValue(
      object(run) &&
        run.schema === "pir2.replay.run" &&
        run.version === 1 &&
        run.id === runId &&
        ["model", "deployment"].includes(run.kind) &&
        ["bf16", "w8a8", "w4a4"].includes(run.precision),
      "invalid_run",
      "Normalized run identity/schema/kind/precision is invalid",
    );
  }
  function validateClocks(clocks) {
    requireValue(
      object(clocks),
      "invalid_clock",
      "Clock registry must be an object",
    );
    for (const [key, definition] of Object.entries(clocks)) {
      requireValue(
        identifier.test(key) &&
          object(definition) &&
          units.has(definition.unit) &&
          domains.has(definition.domain),
        "invalid_clock",
        "Invalid explicit clock definition",
      );
    }
  }
  function validateSnapshot(value, runId) {
    json(value);
    requireValue(
      object(value) &&
        value.schema === "pir2.replay.snapshot" &&
        value.version === 1 &&
        value.run_id === runId,
      "invalid_snapshot",
      "Snapshot schema/version/run ID is invalid",
    );
    sequence(value.last_sequence, true);
    validateRun(value.run, runId);
    validateClocks(value.clocks);
    requireValue(
      object(value.state) &&
        ["running", "ended", "failed", "historical"].includes(
          value.state.status,
        ),
      "invalid_state",
      "Invalid stream state",
    );
    for (const key of ["dropped_events", "recorded_events"])
      sequence(value.state[key], true);
    requireValue(
      typeof value.stream_available === "boolean",
      "invalid_snapshot",
      "Snapshot must declare stream availability",
    );
  }
  function validateEvent(value, runId, clocks) {
    json(value);
    requireValue(
      object(value) &&
        value.schema === "pir2.replay.event" &&
        value.version === 1 &&
        value.run_id === runId,
      "invalid_event",
      "Event schema/version/run ID is invalid",
    );
    sequence(value.sequence);
    requireValue(
      typeof value.type === "string" && eventType.test(value.type),
      "invalid_event",
      "Invalid replay event type",
    );
    const when = value.occurrence;
    requireValue(
      object(when) &&
        Object.keys(when).sort().join(",") === "clock_id,domain,unit,value",
      "invalid_clock",
      "Occurrence needs explicit value/unit/clock_id/domain",
    );
    const clock =
      typeof when.clock_id === "string" && own(clocks, when.clock_id)
        ? clocks[when.clock_id]
        : null;
    requireValue(
      clock && when.unit === clock.unit && when.domain === clock.domain,
      "invalid_clock",
      "Event clock does not match its registry",
    );
    requireValue(
      when.value === null ||
        (typeof when.value === "number" && Number.isFinite(when.value)),
      "invalid_clock",
      "Occurrence must be finite or explicitly unknown",
    );
    requireValue(
      object(value.sources),
      "invalid_sources",
      "Source IDs must be an object",
    );
    for (const [key, id] of Object.entries(value.sources)) {
      requireValue(
        sourceKey.test(key) &&
          (id === null ||
            (typeof id === "string" && id.length > 0 && id.length <= 1024)),
        "invalid_sources",
        "Source IDs must be named opaque strings or null",
      );
    }
    requireValue(
      object(value.payload),
      "invalid_event",
      "Payload must be a JSON object",
    );
  }
  function itemIdentity(collection, item) {
    const key = collections[collection];
    const id = key === null ? item : object(item) ? item[key] : null;
    requireValue(
      typeof id === "string" && id.length > 0,
      "invalid_projection",
      "Append item requires an immutable identity",
    );
    return id;
  }
  function project(run, projection) {
    if (projection == null) return run;
    requireValue(
      object(projection),
      "invalid_projection",
      "Projection must be an object",
    );
    if (projection.op === "replace") {
      validateRun(projection.run, run.id);
      requireValue(
        projection.run.kind === run.kind &&
          projection.run.precision === run.precision,
        "invalid_projection",
        "Replacement cannot change run kind or precision",
      );
      return freeze(clone(projection.run));
    }
    requireValue(
      projection.op === "append" &&
        typeof projection.collection === "string" &&
        own(collections, projection.collection) &&
        Array.isArray(projection.items),
      "invalid_projection",
      "Unknown append collection or projection operation",
    );
    const collection = projection.collection;
    const existing = own(run, collection) ? run[collection] : [];
    requireValue(
      Array.isArray(existing),
      "invalid_projection",
      "Append target must be an array",
    );
    const seen = new Set(
      existing.map((item) => itemIdentity(collection, item)),
    );
    for (const item of projection.items) {
      const id = itemIdentity(collection, item);
      requireValue(
        !seen.has(id),
        "identity_collision",
        "Append cannot replace an existing immutable identity",
      );
      seen.add(id);
    }
    return freeze({
      ...run,
      [collection]: [...existing, ...clone(projection.items)],
    });
  }

  class StreamStore {
    constructor(runId, maxEvents = 512) {
      requireValue(
        typeof runId === "string" && identifier.test(runId),
        "invalid_run",
        "Run ID must be an opaque identifier",
      );
      requireValue(
        Number.isSafeInteger(maxEvents) && maxEvents > 0,
        "invalid_bound",
        "Recent event bound must be a positive integer",
      );
      this.run_id = runId;
      this.maxEvents = maxEvents;
      this._snapshot = null;
      this._events = Object.freeze([]);
      this._known = new Map();
    }
    get run() {
      return this._snapshot?.run ?? null;
    }
    get last_sequence() {
      return this._snapshot?.last_sequence ?? 0;
    }
    get state() {
      return this._snapshot?.state ?? null;
    }
    get clocks() {
      return this._snapshot?.clocks ?? null;
    }
    get stream_available() {
      return this._snapshot?.stream_available ?? false;
    }
    get events() {
      return this._events;
    }
    snapshot(envelope) {
      validateSnapshot(envelope, this.run_id);
      this._snapshot = freeze(clone(envelope));
      this._events = Object.freeze([]);
      this._known.clear();
      return this;
    }
    event(envelope) {
      requireValue(
        this._snapshot !== null,
        "missing_snapshot",
        "Fetch a snapshot before applying events",
      );
      validateEvent(envelope, this.run_id, this.clocks);
      const signature = canonical(envelope);
      if (envelope.sequence <= this.last_sequence) {
        // Outside bounded history, only the cursor is known; no equality claim is made.
        requireValue(
          !this._known.has(envelope.sequence) ||
            this._known.get(envelope.sequence) === signature,
          "duplicate_conflict",
          "Known event sequence has conflicting evidence",
        );
        return false;
      }
      requireValue(
        envelope.sequence === this.last_sequence + 1,
        "sequence_gap",
        "Replay sequence has a gap; fetch a fresh snapshot",
      );
      requireValue(
        !terminal(this.state.status),
        "terminal_stream",
        "Cannot append after a terminal stream state",
      );
      const run = project(this.run, envelope.projection);
      const state = {
        ...this.state,
        recorded_events: this.state.recorded_events + 1,
      };
      if (envelope.type === "stream.dropped") {
        sequence(envelope.payload.count);
        state.dropped_events += envelope.payload.count;
      }
      if (envelope.type === "run.ended") state.status = "ended";
      if (["run.failed", "stream.failed"].includes(envelope.type)) {
        state.status = "failed";
        state.error = envelope.payload.error ?? "Run reported failure";
      }
      sequence(state.recorded_events, true);
      sequence(state.dropped_events, true);
      const saved = freeze(clone(envelope));
      this._snapshot = freeze({
        ...this._snapshot,
        run,
        state,
        last_sequence: envelope.sequence,
      });
      this._events = Object.freeze(
        [...this._events, saved].slice(-this.maxEvents),
      );
      this._known.set(envelope.sequence, signature);
      while (this._known.size > this.maxEvents)
        this._known.delete(this._known.keys().next().value);
      return true;
    }
  }

  class StreamConnection {
    constructor(options = {}) {
      this.fetchSnapshot =
        options.fetchSnapshot ??
        ((runId) =>
          fetch(`/api/replay/snapshot/${encodeURIComponent(runId)}`, {
            cache: "no-store",
          }));
      this.EventSource = options.EventSource ?? globalThis.EventSource;
      this.onSnapshot = options.onSnapshot ?? (() => {});
      this.onEvent = options.onEvent ?? (() => {});
      this.onStatus = options.onStatus ?? (() => {});
      this.maxEvents = options.maxEvents ?? 512;
      this.store = null;
      this._generation = 0;
      this._source = null;
    }
    _status(status, fields = {}) {
      this.onStatus({ status, run_id: this.store?.run_id ?? null, ...fields });
    }
    _disconnect() {
      const source = this._source;
      this._source = null;
      source?.close();
    }
    async start(runId) {
      const store = new StreamStore(runId, this.maxEvents);
      const generation = ++this._generation;
      this._disconnect();
      this.store = store;
      this._status("loading");
      return this._load(generation, runId);
    }
    close() {
      ++this._generation;
      this._disconnect();
      this._status("closed");
    }
    _failure(error) {
      this._disconnect();
      this._status("error", {
        code: error.code ?? "transport_error",
        error: String(error.message ?? error),
      });
    }
    async _load(generation, runId) {
      let envelope;
      try {
        const response = await this.fetchSnapshot(runId);
        if (generation !== this._generation) return null;
        if (response?.status === 404) {
          this._status("historical", { unavailable: true });
          return null;
        }
        if (typeof response?.json === "function") {
          if (response.ok === false)
            throw new Error(`Replay snapshot HTTP ${response.status}`);
          envelope = await response.json();
        } else envelope = response;
        if (generation !== this._generation) return null;
        this.store.snapshot(envelope);
      } catch (error) {
        if (generation !== this._generation) return null;
        if (error?.status === 404)
          this._status("historical", { unavailable: true });
        else this._failure(error);
        return null;
      }
      this.onSnapshot(this.store, envelope);
      if (generation !== this._generation) return null;
      if (terminal(this.store.state.status)) {
        this._status(this.store.state.status, {
          error: this.store.state.error,
        });
      } else if (
        !this.store.stream_available ||
        this.store.state.status === "historical"
      ) {
        this._status("historical");
      } else this._subscribe(generation, runId);
      return this.store;
    }
    _resnapshot(generation, runId, reason) {
      if (generation !== this._generation) return;
      const next = ++this._generation;
      this._disconnect();
      this._status("resnapshotting", { reason });
      void this._load(next, runId);
    }
    _subscribe(generation, runId) {
      let source;
      try {
        source = new this.EventSource(
          `/api/replay/events/${encodeURIComponent(runId)}?after=${this.store.last_sequence}`,
        );
      } catch (error) {
        this._failure(error);
        return;
      }
      this._source = source;
      const active = () =>
        generation === this._generation && this._source === source;
      source.onopen = () => {
        if (active()) this._status("live");
      };
      // Native EventSource owns reconnect and Last-Event-ID. EOF never ends a run.
      source.onerror = () => {
        if (!active()) return;
        if (source.readyState === 2) {
          this._failure(
            new StreamProtocolError(
              "transport_closed",
              "Replay transport closed without a terminal event",
            ),
          );
        } else this._status("reconnecting");
      };
      source.addEventListener("replay", (message) => {
        if (!active()) return;
        let value, accepted;
        try {
          value = JSON.parse(message.data);
          accepted = this.store.event(value);
        } catch (error) {
          if (error.code === "sequence_gap")
            this._resnapshot(generation, runId, error.code);
          else this._failure(error);
          return;
        }
        if (accepted) this.onEvent(value, this.store);
        if (!active()) return;
        if (terminal(this.store.state.status)) {
          this._disconnect();
          this._status(this.store.state.status, {
            error: this.store.state.error,
            scope: value.payload.scope ?? null,
            model_failure: value.payload.model_failure ?? null,
          });
        }
      });
      source.addEventListener("resnapshot", (message) => {
        if (!active()) return;
        try {
          const control = JSON.parse(message.data);
          requireValue(
            object(control) && control.run_id === runId,
            "invalid_control",
            "Resnapshot belongs to a different run",
          );
          // URL comes from the selected opaque ID, never untrusted control payload.
          this._resnapshot(generation, runId, control.reason ?? "resnapshot");
        } catch (error) {
          this._failure(error);
        }
      });
      source.addEventListener("end", (message) => {
        if (!active()) return;
        try {
          const control = JSON.parse(message.data);
          requireValue(
            object(control) &&
              control.run_id === runId &&
              terminal(control.status),
            "invalid_control",
            "Invalid terminal stream control",
          );
          sequence(control.last_sequence, true);
          if (
            control.last_sequence !== this.store.last_sequence ||
            control.status !== this.store.state.status
          ) {
            this._resnapshot(generation, runId, "terminal_cursor_mismatch");
            return;
          }
          this._disconnect();
          this._status(control.status);
        } catch (error) {
          this._failure(error);
        }
      });
      this._status("live");
    }
  }

  return { StreamStore, StreamConnection, StreamProtocolError };
});
