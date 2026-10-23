/* A small selector index and only the currently selected full recorded runs. */
(function (root, factory) {
  if (typeof module === "object" && module.exports)
    module.exports = factory(require("./replay-dynamic.js"));
  else root.DynamicReplayStore = factory(root.DynamicReplay);
})(typeof globalThis !== "undefined" ? globalThis : this, function (D) {
  "use strict";
  function validateEpisodes(episodes, descriptor) {
    if (
      !Array.isArray(episodes) ||
      (Number.isInteger(descriptor.episodes) &&
        episodes.length !== descriptor.episodes)
    )
      throw new Error(`Dynamic episode count mismatch: ${descriptor.id}`);
    const ids = new Set();
    for (const e of episodes) {
      if (!e || typeof e.id !== "string" || !e.id || ids.has(e.id))
        throw new Error(`Dynamic episode identity mismatch: ${descriptor.id}`);
      ids.add(e.id);
    }
  }
  function validateRun(run, descriptor) {
    if (
      !run ||
      run.schema !== "pir2.replay.run" ||
      run.version !== 1 ||
      ["id", "kind", "precision"].some((key) => run[key] !== descriptor[key]) ||
      (Object.hasOwn(descriptor, "protocol") &&
        (run.protocol ?? null) !== descriptor.protocol)
    )
      throw new Error(`Dynamic run identity mismatch: ${descriptor.id}`);
    validateEpisodes(run.episodes, descriptor);
    if (descriptor.episode_summaries) {
      for (const summary of descriptor.episode_summaries) {
        const episode = run.episodes.find((e) => e.id === summary.id);
        if (!episode || D.conditionKey(episode) !== D.conditionKey(summary))
          throw new Error(
            `Dynamic episode/condition identity mismatch: ${descriptor.id} / ${summary.id}`,
          );
        if (
          JSON.stringify(D.episodeSummary(episode)) !==
          JSON.stringify(D.episodeSummary(summary))
        )
          throw new Error(
            `Dynamic episode metadata mismatch; refresh catalog index: ${descriptor.id} / ${summary.id}`,
          );
      }
    }
    return run;
  }
  class Store {
    constructor({ fetch = globalThis.fetch.bind(globalThis) } = {}) {
      this.fetch = fetch;
      this.index = [];
      this.runs = new Map();
      this.pending = new Map();
      this.selected = new Set();
      this.epoch = 0;
    }
    async json(url, signal) {
      const response = await this.fetch(url, { signal });
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      return response.json();
    }
    async fetchRun(descriptor, signal) {
      const run = await this.json(
        `/api/replay/run/${encodeURIComponent(descriptor.id)}`,
        signal,
      );
      return validateRun(run, descriptor);
    }
    async initialize() {
      const catalog = await this.json("/api/replay/catalog");
      if (
        catalog.schema !== "pir2.replay.catalog" ||
        catalog.version !== 1 ||
        !Array.isArray(catalog.runs)
      )
        throw new Error("不支持的 catalog");
      const index = [],
        identifiers = new Set();
      for (const descriptor of catalog.runs.filter(
        (r) => r.kind === "dynamic",
      )) {
        if (
          typeof descriptor.id !== "string" ||
          !descriptor.id ||
          !descriptor.precision ||
          identifiers.has(descriptor.id)
        )
          throw new Error("Dynamic catalog run identity mismatch");
        identifiers.add(descriptor.id);
        let summaries = descriptor.episode_summaries,
          protocol = descriptor.protocol;
        if (summaries === undefined) {
          // Legacy fallback is sequential and never adds full data to the selected cache.
          const run = await this.fetchRun(descriptor);
          summaries = run.episodes.map(D.episodeSummary);
          protocol = run.protocol ?? null;
        }
        validateEpisodes(summaries, descriptor);
        if (protocol === undefined)
          throw new Error(`Dynamic catalog protocol missing: ${descriptor.id}`);
        index.push({
          ...descriptor,
          protocol,
          episode_summaries: summaries.map(D.episodeSummary),
        });
      }
      this.index = index;
      return catalog;
    }
    retain(ids) {
      this.selected = new Set(ids);
      for (const id of this.runs.keys())
        if (!this.selected.has(id)) this.runs.delete(id);
      for (const [id, entry] of this.pending) {
        if (this.selected.has(id)) continue;
        this.pending.delete(id);
        entry.controller.abort();
      }
    }
    get(descriptor) {
      const id = descriptor.id;
      if (this.runs.has(id)) return Promise.resolve();
      if (this.pending.has(id)) return this.pending.get(id).promise;
      const entry = { controller: new AbortController() };
      entry.promise = this.fetchRun(descriptor, entry.controller.signal)
        .then((run) => {
          if (
            entry.controller.signal.aborted ||
            this.pending.get(id) !== entry ||
            !this.selected.has(id)
          )
            return;
          this.runs.set(id, run);
        })
        .finally(() => {
          if (this.pending.get(id) === entry) this.pending.delete(id);
        });
      this.pending.set(id, entry);
      return entry.promise;
    }
    async select(rows) {
      const epoch = ++this.epoch;
      try {
        if (rows.length > D.MAX_ROWS) throw new Error("最多显示 4 列动态回放");
        const selected = rows.map((row) => {
          const descriptor = this.index.find((r) => r.id === row.run.id);
          const episode = descriptor?.episode_summaries.find(
            (e) => e.id === row.episode.id,
          );
          if (
            !episode ||
            D.conditionKey(episode) !== D.conditionKey(row.episode)
          )
            throw new Error(
              "Selected dynamic episode/condition identity mismatch",
            );
          return { row, descriptor };
        });
        this.retain(selected.map(({ descriptor }) => descriptor.id));
        // Pending selections hold only summaries; a superseded Promise.all must
        // not retain full runs while another transport ignores its abort signal.
        await Promise.all(
          selected.map(({ descriptor }) => this.get(descriptor)),
        );
        if (epoch !== this.epoch) return null;
        return selected.map(({ row, descriptor }) => {
          const run = this.runs.get(descriptor.id);
          return {
            ...row,
            run,
            episode: run.episodes.find((e) => e.id === row.episode.id),
          };
        });
      } catch (error) {
        if (epoch !== this.epoch) return null;
        this.retain([]);
        throw error;
      }
    }
  }
  return { Store };
});
