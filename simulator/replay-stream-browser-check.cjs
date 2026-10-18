/* Local protocol-fixture browser checks. Never starts inference or GPU work. */
"use strict";

const assert = require("node:assert/strict");
const { chromium } = require("../tools/ui-check/node_modules/playwright");

const external = process.argv
  .find((value) => value.startsWith("--url="))
  ?.slice("--url=".length);
const url = external ?? "http://127.0.0.1:8787/simulator/replay.html";
const fixtureNote = "Protocol fixture only; not measured model evidence";

// Extends the golden stream contract with explicitly synthetic display cursors.
// clean_wall=false excludes these numbers from measured clean timing summaries.
function request(runId, index) {
  return {
    id: `${runId}-request-${index}`,
    request_id: `${runId}-request-${index}`,
    observation_id: `${runId}-observation-${index}`,
    observation: index,
    round: 0,
    scope: "prepared_model",
    wall_ms: index * 1000,
    output: null,
    instrumented: null,
    clean_wall: false,
    seed: null,
    timing_status: "protocol_fixture_not_measured",
    test_fixture: true,
  };
}

function run(runId) {
  return {
    schema: "pir2.replay.run",
    version: 1,
    id: runId,
    kind: "model",
    precision: "w8a8",
    mode: runId,
    label: fixtureNote,
    status: "incomplete",
    observations: [],
    clean_requests: [request(runId, 1), request(runId, 2)],
    profile_requests: [],
    kernels: [],
    warnings: [fixtureNote],
  };
}

function profileRun(runId, revision = 0) {
  const value = run(runId);
  value.profile_requests = [
    {
      ...request(runId, 1),
      id: `${runId}-profile-1`,
      request_id: `${runId}-profile-1`,
      host_wall_ms: 1000,
      instrumented: true,
      fixture_revision: revision,
      timeline_status: "protocol_fixture_not_measured",
      timeline: revision
        ? ["vlm", "action"].map((stage, index) => ({
            stage,
            start_ms: 100 + index * 200,
            duration_ms: 20 + index * 10 + revision * 10,
            kernel_id: 0,
            category: "gemm",
            stream: 0,
            device: 0,
            test_fixture: true,
          }))
        : [],
    },
  ];
  value.kernels = revision ? [`protocol_fixture_kernel_v${revision}`] : [];
  return value;
}

function snapshot(runId, sequence = 7) {
  return {
    schema: "pir2.replay.snapshot",
    version: 1,
    run_id: runId,
    last_sequence: sequence,
    run: run(runId),
    stream_available: true,
    state: {
      status: "running",
      dropped_events: 0,
      recorded_events: sequence,
      error: null,
    },
    clocks: {
      "host-monotonic": { unit: "s", domain: "host_monotonic" },
      "recorder-monotonic": { unit: "s", domain: "host_monotonic" },
    },
  };
}

function event(runId, sequence, changes = {}) {
  return {
    schema: "pir2.replay.event",
    version: 1,
    run_id: runId,
    sequence,
    type: "model.stage.started",
    occurrence: {
      value: null,
      unit: "s",
      clock_id: "host-monotonic",
      domain: "host_monotonic",
    },
    sources: { request_id: null, observation_id: null },
    payload: {
      stage: "vlm",
      end: null,
      timing_boundary: "host_submission",
      test_fixture: true,
    },
    projection: null,
    ...changes,
  };
}

function appendRequest(runId, sequence, index) {
  return event(runId, sequence, {
    type: "model.request.completed",
    sources: { request_id: request(runId, index).request_id },
    projection: {
      op: "append",
      collection: "clean_requests",
      items: [request(runId, index)],
    },
  });
}

function catalog(live = true) {
  const result = {
    schema: "pir2.replay.catalog",
    version: 1,
    created_at: null,
    runs: ["contract-a", "contract-b"].map((id) => ({
      id,
      kind: "model",
      precision: "w8a8",
      mode: id,
      label: fixtureNote,
      path: `runs/${id}.json`,
    })),
  };
  if (live)
    result.streams = result.runs.map(({ id }) => ({
      run_id: id,
      snapshot_path: `streams/${id}/snapshot.json`,
      journal_path: `streams/${id}/events.jsonl`,
    }));
  return result;
}

async function fixturePage(browser, options = {}) {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  page.setDefaultTimeout(4000);
  const errors = [];
  const requests = [];
  const pendingSnapshots = [];
  let resolveHeldSnapshot;
  const snapshotHeld = new Promise((resolve) => {
    resolveHeldSnapshot = resolve;
  });
  const data = options.catalog ?? catalog();
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => {
    const harness = { sources: [], log: [], responses: [], frames: [] };
    window.__streamHarness = harness;
    const originalFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const response = await originalFetch(...args);
      const resource = String(args[0]);
      harness.responses.push(resource);
      if (resource.includes("/api/replay/snapshot/")) {
        const originalJson = response.json.bind(response);
        response.json = async () => {
          const value = await originalJson();
          harness.log.push({ kind: "snapshot", run: value.run_id });
          return value;
        };
      }
      return response;
    };
    const originalFrame = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (callback) =>
      callback.name === "animate"
        ? harness.frames.push(callback)
        : originalFrame(callback);
    harness.frame = (time) => {
      const callbacks = harness.frames.splice(0);
      for (const callback of callbacks) callback(time);
    };
    class FakeEventSource {
      constructor(sourceUrl) {
        this.url = sourceUrl;
        this.closed = false;
        this.readyState = 0;
        this.listeners = new Map();
        harness.sources.push(this);
        harness.log.push({ kind: "subscribe", url: sourceUrl });
        setTimeout(() => {
          if (this.closed) return;
          this.readyState = 1;
          this.onopen?.({});
        }, 0);
      }
      addEventListener(type, callback) {
        const callbacks = this.listeners.get(type) ?? [];
        callbacks.push(callback);
        this.listeners.set(type, callbacks);
      }
      close() {
        this.closed = true;
        this.readyState = 2;
      }
      emit(type, value) {
        // Deliberately deliver queued callbacks even after close for race checks.
        const message = { data: JSON.stringify(value) };
        for (const callback of this.listeners.get(type) ?? [])
          callback(message);
        if (type === "message") this.onmessage?.(message);
      }
    }
    window.EventSource = FakeEventSource;
  });
  await page.route("**/api/replay/**", async (route) => {
    const resource = new URL(route.request().url()).pathname;
    requests.push(resource);
    if (resource === "/api/replay/catalog")
      return route.fulfill({ json: data });
    const [, type, id] =
      resource.match(/\/api\/replay\/(run|snapshot)\/([^/]+)$/) ?? [];
    if (type === "run")
      return route.fulfill({ json: options.run?.(id) ?? run(id) });
    if (type === "snapshot") {
      if (options.holdSnapshot === id) {
        pendingSnapshots.push({ route, id });
        resolveHeldSnapshot();
        return;
      }
      return route.fulfill({ json: options.snapshot?.(id) ?? snapshot(id) });
    }
    return route.fulfill({
      status: 404,
      json: { error: "fixture has no asset" },
    });
  });
  await page.goto(url);
  try {
    if (options.holdSnapshot)
      await page.waitForFunction(
        () =>
          document.querySelector("#data-status").dataset.state === "loading",
      );
    else
      await page.waitForFunction(
        () => document.querySelector("#data-status").dataset.state === "ready",
      );
  } catch (error) {
    throw new Error(
      `${error.message}; page errors=${JSON.stringify(errors)}; banner=${await page.locator("#error-banner").textContent()}`,
    );
  }
  return {
    page,
    errors,
    requests,
    pendingSnapshots,
    snapshotHeld,
    catalog: data,
  };
}

async function sourceCount(page) {
  return page.evaluate(() => window.__streamHarness.sources.length);
}

async function waitForSource(page, runId) {
  await page.waitForFunction(
    (id) =>
      window.__streamHarness.sources.some(
        (source) =>
          source.url.includes(`/events/${id}?`) &&
          !source.closed &&
          source.readyState === 1,
      ),
    runId,
  );
  return page.evaluate(
    (id) =>
      window.__streamHarness.sources.findIndex(
        (source) =>
          source.url.includes(`/events/${id}?`) &&
          !source.closed &&
          source.readyState === 1,
      ),
    runId,
  );
}

async function emit(page, source, value, type = "replay") {
  await page.evaluate(
    async ({ source, value, type }) => {
      window.__streamHarness.sources[source].emit(type, value);
      // Projection rendering is coalesced on RAF; let that UI work settle.
      await new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(resolve)),
      );
    },
    { source, value, type },
  );
}

async function seek(page, value) {
  await page.locator("#seek").evaluate((element, cursor) => {
    element.value = String(cursor);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  }, value);
}

async function selection(page) {
  return page.evaluate(() => ({
    run: document.querySelector(".run-select")?.value,
    sample: document.querySelector("#sample").value,
    cursor: document.querySelector("#seek").value,
    play: document.querySelector("#play").textContent,
    inspector: document.querySelector("#inspector").textContent,
    selectedEvents: document.querySelectorAll("#event-list .selected").length,
  }));
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-gpu"],
  });
  const results = [];
  async function check(name, body, options) {
    const context = await fixturePage(browser, options);
    try {
      await body(context);
      assert.deepEqual(context.errors, [], "page JavaScript must not throw");
      results.push(name);
      console.log("PASS " + name);
    } finally {
      await context.page.close();
    }
  }
  try {
    await check(
      "empty catalog loads its first discovered run and starts its stream",
      async ({ page, catalog: data, requests }) => {
        assert.equal(await page.locator(".precision-card").count(), 0);
        assert.equal(await sourceCount(page), 0);
        assert.equal(await page.locator("#follow-latest").isDisabled(), true);
        Object.assign(data, catalog());
        await page.waitForSelector(".precision-card", { timeout: 6500 });
        await waitForSource(page, "contract-a");
        assert.equal(await page.locator(".precision-card").count(), 1);
        assert.equal(
          await page.locator(".run-select").inputValue(),
          "contract-a",
        );
        assert.equal(await page.locator("#sample option").count(), 2);
        assert.equal(await page.locator("#follow-latest").isDisabled(), false);
        assert.equal(await sourceCount(page), 1);
        assert.equal(
          requests.includes("/api/replay/snapshot/contract-b"),
          false,
        );
      },
      { catalog: { ...catalog(), runs: [], streams: [] } },
    );

    await check(
      "snapshot-first subscription and preserved user selection",
      async ({ page }) => {
        assert.equal(
          await page.locator("#stream-status").count(),
          1,
          "stream UI needs an explicit transport status",
        );
        assert.equal(await page.locator("#follow-latest").isChecked(), false);
        const source = await waitForSource(page, "contract-a");
        const log = await page.evaluate(() => window.__streamHarness.log);
        assert.deepEqual(log.slice(0, 2), [
          { kind: "snapshot", run: "contract-a" },
          { kind: "subscribe", url: "/api/replay/events/contract-a?after=7" },
        ]);
        assert.equal(
          await sourceCount(page),
          1,
          "unselected runs must not subscribe",
        );
        await page.selectOption("#sample", "contract-a-request-2");
        await page.selectOption("#window", "all");
        await page.locator("#event-list button").first().click();
        await seek(page, 250);
        await page.locator("#play").click();
        const before = await selection(page);
        assert.equal(before.play, "暂停");
        assert.equal(before.selectedEvents, 1);
        await emit(page, source, appendRequest("contract-a", 8, 3));
        assert.equal(await page.locator("#sample option").count(), 3);
        assert.deepEqual(
          await selection(page),
          before,
          "stream projection must preserve run, sample, cursor, PLAY and selected evidence",
        );
        await page.evaluate(() => {
          window.__streamHarness.frame(100);
          window.__streamHarness.frame(200);
        });
        assert.ok(
          Number(await page.locator("#seek").inputValue()) > 250,
          "PLAY must continue on its own clock after a stream update",
        );
        await page.locator("#play").click();
        await page.locator("#stream-details").evaluate((element) => {
          element.open = true;
        });
        await page.locator("#stream-event-list button").first().click();
        const inspected = JSON.parse(
          await page.locator("#inspector").innerText(),
        );
        const inspectedText = JSON.stringify(inspected);
        assert.match(inspectedText, /host-monotonic/);
        assert.match(inspectedText, /host_submission/);
        assert.match(inspectedText, /"value":null/);
        assert.match(inspectedText, /contract-a-request-3/);
        assert.match(inspectedText, /"end":null/);
      },
    );

    await check(
      "replacement refreshes cached profile tracks and same-ID kernel evidence",
      async ({ page }) => {
        const source = await waitForSource(page, "contract-a");
        await page.selectOption("#record-type", "profile");
        await page.selectOption("#window", "all");
        await seek(page, 250);
        await page.locator("#kernel-details").evaluate((element) => {
          element.open = true;
        });
        await page.waitForSelector("#kernel-timeline svg");
        assert.equal(await page.locator(".task-block").count(), 0);
        assert.equal(
          await page.locator('[data-event-kind="kernel"]').count(),
          0,
        );
        const before = await selection(page);
        await emit(
          page,
          source,
          event("contract-a", 8, {
            type: "model.request.completed",
            sources: { request_id: "contract-a-profile-1" },
            projection: { op: "replace", run: profileRun("contract-a", 1) },
          }),
        );
        assert.equal(
          await page.locator("#record-type").inputValue(),
          "profile",
        );
        assert.deepEqual(await selection(page), before);
        assert.deepEqual(
          {
            tasks: await page.locator(".task-block").count(),
            kernels: await page.locator('[data-event-kind="kernel"]').count(),
          },
          { tasks: 2, kernels: 2 },
          "new profile evidence must refresh both previously empty cached timelines",
        );
        await page
          .locator('.task-block[data-block="contract-a-profile-1/vlm"]')
          .click();
        assert.equal(
          JSON.parse(await page.locator("#inspector").innerText())
            .fixture_revision,
          1,
        );
        const selected = await selection(page);
        await emit(
          page,
          source,
          event("contract-a", 9, {
            type: "model.request.completed",
            sources: { request_id: "contract-a-profile-1" },
            projection: { op: "replace", run: profileRun("contract-a", 2) },
          }),
        );
        assert.deepEqual(
          await selection(page),
          selected,
          "replacing the same profile ID must preserve the inspected selection and cursor",
        );
        assert.equal(
          await page
            .locator(
              '.task-block.selected[data-block="contract-a-profile-1/vlm"]',
            )
            .count(),
          1,
        );
        await page
          .locator('.task-block[data-block="contract-a-profile-1/vlm"]')
          .click();
        assert.equal(
          JSON.parse(await page.locator("#inspector").innerText())
            .fixture_revision,
          2,
          "same-ID task click must inspect the updated record",
        );
        await page.locator('[data-event-kind="kernel"]').first().click();
        const inspected = JSON.parse(
          await page.locator("#inspector").innerText(),
        );
        assert.equal(inspected.kernel, "protocol_fixture_kernel_v2");
        assert.equal(inspected.duration_ms, 40);
      },
      { snapshot: (id) => ({ ...snapshot(id), run: profileRun(id) }) },
    );

    await check(
      "follow latest is opt-in and terminal stream state is explicit",
      async ({ page }) => {
        const source = await waitForSource(page, "contract-a");
        await page.selectOption("#sample", "contract-a-request-2");
        await seek(page, 250);
        await emit(page, source, appendRequest("contract-a", 8, 3));
        assert.equal(
          await page.locator("#sample").inputValue(),
          "contract-a-request-2",
        );
        assert.equal(await page.locator("#seek").inputValue(), "250");
        await page.locator("#follow-latest").check();
        await emit(page, source, appendRequest("contract-a", 9, 4));
        assert.equal(
          await page.locator("#seek").inputValue(),
          await page.locator("#seek").getAttribute("max"),
        );
        await page.locator("#follow-latest").uncheck();
        await seek(page, 125);
        const retained = await selection(page);
        await emit(
          page,
          source,
          event("contract-a", 10, {
            type: "run.ended",
            payload: { test_fixture: true },
          }),
        );
        assert.equal(
          await page.evaluate(
            (index) => window.__streamHarness.sources[index].closed,
            source,
          ),
          true,
        );
        assert.deepEqual(await selection(page), retained);
        assert.match(
          await page.locator("#stream-status").innerText(),
          /ended|结束/i,
        );
        assert.doesNotMatch(
          await page.locator("#precision-cards").innerText(),
          /回合最终结果：成功|模型成功/,
        );
      },
    );

    await check(
      "recording failure stays separate from model result",
      async ({ page }) => {
        const source = await waitForSource(page, "contract-a");
        await emit(
          page,
          source,
          event("contract-a", 8, {
            type: "stream.dropped",
            payload: { count: 3, test_fixture: true },
          }),
        );
        await emit(
          page,
          source,
          event("contract-a", 9, {
            type: "stream.failed",
            payload: {
              error: "fixture recorder unavailable",
              scope: "event_recording",
              model_failure: false,
              test_fixture: true,
            },
          }),
        );
        assert.equal(
          await page.evaluate(
            (index) => window.__streamHarness.sources[index].closed,
            source,
          ),
          true,
        );
        const status = await page.locator("#stream-status").innerText();
        assert.match(status, /failed|失败/i);
        assert.match(status, /fixture recorder unavailable/);
        assert.match(status, /3/);
        assert.match(
          await page.locator("#precision-cards").innerText(),
          /incomplete/,
        );
        assert.doesNotMatch(
          await page.locator("#precision-cards").innerText(),
          /回合最终结果：失败|模型失败/,
        );
      },
    );

    await check(
      "run and view changes close subscriptions and ignore late callbacks",
      async ({ page }) => {
        const old = await waitForSource(page, "contract-a");
        await page.selectOption(".run-select", "contract-b");
        const current = await waitForSource(page, "contract-b");
        assert.equal(
          await page.evaluate(
            (index) => window.__streamHarness.sources[index].closed,
            old,
          ),
          true,
        );
        const before = await selection(page);
        const statusBefore = await page.locator("#stream-status").innerText();
        await emit(page, old, appendRequest("contract-a", 8, 99));
        await page.evaluate((index) => {
          const stale = window.__streamHarness.sources[index];
          stale.onopen?.({});
          stale.onerror?.({});
        }, old);
        assert.deepEqual(await selection(page), before);
        assert.equal(
          await page.locator("#stream-status").innerText(),
          statusBefore,
        );
        assert.equal(
          await page
            .locator('#sample option[value="contract-a-request-99"]')
            .count(),
          0,
        );
        await page.locator("#mode-deployment").click();
        await page.waitForFunction(
          () =>
            document.querySelector("#data-status").dataset.state === "ready",
        );
        assert.equal(
          await page.evaluate(
            (index) => window.__streamHarness.sources[index].closed,
            current,
          ),
          true,
        );
        await emit(page, current, appendRequest("contract-b", 8, 99));
        assert.equal(await page.locator(".precision-card").count(), 0);
        assert.equal(await page.locator("#sample option").count(), 0);
        assert.equal(
          await page.locator("#mode-deployment").getAttribute("aria-pressed"),
          "true",
        );
        assert.equal(await sourceCount(page), 2);
      },
    );

    await check(
      "late snapshots cannot reopen a deselected run",
      async ({ page, pendingSnapshots, snapshotHeld }) => {
        await snapshotHeld;
        assert.equal(await sourceCount(page), 0);
        await page.locator("#mode-deployment").click();
        assert.equal(
          pendingSnapshots.length,
          1,
          "selected stream snapshot must be requested",
        );
        await pendingSnapshots[0].route.fulfill({
          json: snapshot("contract-a"),
        });
        await page.waitForFunction(() =>
          window.__streamHarness.responses.some((resource) =>
            resource.includes("/snapshot/contract-a"),
          ),
        );
        assert.equal(await sourceCount(page), 0);
        assert.equal(await page.locator(".precision-card").count(), 0);
        assert.equal(
          await page.locator("#mode-deployment").getAttribute("aria-pressed"),
          "true",
        );
      },
      { holdSnapshot: "contract-a" },
    );

    await check(
      "BFCache restore resnapshots and resumes subscriptions without resetting playback",
      async ({ page, requests }) => {
        const old = await waitForSource(page, "contract-a");
        await page.selectOption("#sample", "contract-a-request-2");
        await page.selectOption("#window", "all");
        await page.locator("#event-list button").first().click();
        await seek(page, 250);
        await page.locator("#play").click();
        const before = await selection(page);
        await page.evaluate(() => {
          window.dispatchEvent(
            new PageTransitionEvent("pagehide", { persisted: true }),
          );
          window.dispatchEvent(
            new PageTransitionEvent("pageshow", { persisted: true }),
          );
        });
        await page.waitForFunction(
          () => window.__streamHarness.sources.length === 2,
        );
        const current = await waitForSource(page, "contract-a");
        assert.notEqual(current, old);
        assert.equal(
          await page.evaluate(
            (index) => window.__streamHarness.sources[index].closed,
            old,
          ),
          true,
        );
        assert.equal(
          requests.filter(
            (resource) => resource === "/api/replay/snapshot/contract-a",
          ).length,
          2,
        );
        assert.deepEqual(await selection(page), before);
        assert.deepEqual(
          await page.evaluate(() => window.__streamHarness.log.slice(-2)),
          [
            { kind: "snapshot", run: "contract-a" },
            { kind: "subscribe", url: "/api/replay/events/contract-a?after=7" },
          ],
        );
        await emit(page, old, appendRequest("contract-a", 8, 99));
        assert.equal(
          await page
            .locator('#sample option[value="contract-a-request-99"]')
            .count(),
          0,
        );
        await emit(page, current, appendRequest("contract-a", 8, 3));
        assert.equal(await page.locator("#sample option").count(), 3);
        assert.deepEqual(await selection(page), before);
      },
    );

    await check(
      "catalog replacement retains loaded precision and execution-clock choices",
      async ({ page, catalog: data }) => {
        await page.selectOption("#precision", "w8a8");
        assert.equal(
          await page.locator("#execution-clock").inputValue(),
          "ready",
        );
        const before = await selection(page);
        data.runs = [];
        data.streams = [];
        data.created_at = "protocol-fixture-catalog-replaced";
        await page.waitForFunction(
          () =>
            document
              .querySelector("#catalog-meta")
              .textContent.includes("protocol-fixture-catalog-replaced"),
          null,
          { timeout: 6500 },
        );
        assert.deepEqual(
          {
            precision: await page.locator("#precision").inputValue(),
            cadence: await page.locator("#execution-clock").inputValue(),
          },
          { precision: "w8a8", cadence: "ready" },
        );
        assert.deepEqual(await selection(page), before);
      },
      {
        catalog: {
          ...catalog(false),
          default_view: "deployment",
          default_execution_clock: "ready",
          runs: [
            {
              id: "deployment-fixture",
              kind: "deployment",
              precision: "w8a8",
              execution_clock: "ready",
              episodes: 1,
              label: fixtureNote,
              path: "runs/deployment-fixture.json",
            },
          ],
        },
        run: (id) => ({
          ...run(id),
          kind: "deployment",
          execution_clock: "ready",
          clean_requests: [],
          episodes: [
            {
              id: `${id}-episode`,
              status: "incomplete",
              reason: fixtureNote,
              metadata: { task_id: 0, episode_id: 0, seed: null },
              queue: null,
              ticks: [],
              robot: { status: "unavailable", frames: [], media: {} },
            },
          ],
        }),
      },
    );

    await check(
      "catalog discovery adds choices without replacing the chosen run",
      async ({ page, catalog: data, requests }) => {
        await waitForSource(page, "contract-a");
        await page.selectOption(".run-select", "contract-b");
        await waitForSource(page, "contract-b");
        await page.selectOption("#sample", "contract-b-request-2");
        await seek(page, 250);
        const before = await selection(page);
        const beforeLabel = await page
          .locator(".run-select option:checked")
          .textContent();
        data.runs.push({
          ...data.runs[0],
          id: "contract-c",
          mode: "contract-c",
          path: "runs/contract-c.json",
        });
        data.streams.push({
          run_id: "contract-c",
          snapshot_path: "streams/contract-c/snapshot.json",
          journal_path: "streams/contract-c/events.jsonl",
        });
        await page.waitForSelector('.run-select option[value="contract-c"]', {
          state: "attached",
          timeout: 6500,
        });
        assert.deepEqual(await selection(page), before);
        assert.equal(
          await page.locator(".run-select option:checked").textContent(),
          beforeLabel,
          "catalog discovery must retain the human run label",
        );
        assert.equal(
          await sourceCount(page),
          2,
          "discovery must not subscribe to an unselected run",
        );
        assert.equal(requests.includes("/api/replay/run/contract-c"), false);
        assert.equal(
          requests.includes("/api/replay/snapshot/contract-c"),
          false,
        );
      },
    );

    const legacyDescriptors = ["bf16", "w8a8"].map((precision) => ({
      id: `legacy-${precision}`,
      kind: "deployment",
      precision,
      episodes: 0,
      execution_coupling: "synchronous",
      execution_clock: "ready",
      path: `runs/legacy-${precision}.json`,
      label: fixtureNote,
    }));
    let legacyPrefix = 8;
    const legacyRun = (id, k = legacyPrefix) => ({
      ...run(id),
      ...legacyDescriptors.find((r) => r.id === id),
      configuration: { synchronous_chunk_steps: k },
      episodes: [],
      clean_requests: [],
    });
    await check(
      "legacy SSE projections stay bound to the first resolved actual K/D cohort",
      async ({ page }) => {
        const bf16 = await waitForSource(page, "legacy-bf16");
        const w8 = await waitForSource(page, "legacy-w8a8");
        const grid = await page
          .locator("#precision-cards")
          .evaluate((element) => {
            const cards = [...element.querySelectorAll(".precision-card")];
            const style = getComputedStyle(element);
            return {
              count: cards.length,
              available:
                element.clientWidth -
                parseFloat(style.paddingLeft) -
                parseFloat(style.paddingRight),
              used:
                cards.at(-1).getBoundingClientRect().right -
                cards[0].getBoundingClientRect().left,
            };
          });
        assert.equal(grid.count, 2);
        assert.ok(
          Math.abs(grid.used - grid.available) < 2,
          "two precision cards must fill the desktop grid without a reserved third column",
        );
        const status = (precision) =>
          page.locator(
            `.precision-card[data-precision="${precision}"] .card-status`,
          );
        assert.doesNotMatch(await status("bf16").innerText(), /K\/D schedule/);
        assert.doesNotMatch(await status("w8a8").innerText(), /K\/D schedule/);
        await emit(
          page,
          bf16,
          event("legacy-bf16", 8, {
            projection: { op: "replace", run: legacyRun("legacy-bf16", 1) },
          }),
        );
        assert.match(
          await status("bf16").innerText(),
          /K\/D schedule/,
          "K1 cannot silently replace one card in a resolved K8 cohort",
        );
        assert.doesNotMatch(await status("w8a8").innerText(), /K\/D schedule/);
        assert.equal(
          await page
            .locator('.precision-card[data-precision="bf16"] video')
            .count(),
          0,
        );
        assert.equal(
          await page
            .locator('.precision-card[data-precision="bf16"] .metrics')
            .innerText(),
          "",
        );
        await emit(
          page,
          w8,
          event("legacy-w8a8", 8, {
            projection: { op: "replace", run: legacyRun("legacy-w8a8", 1) },
          }),
        );
        assert.match(
          await status("w8a8").innerText(),
          /K\/D schedule/,
          "the binding must survive rejection, even if both streams drift",
        );
        await emit(
          page,
          bf16,
          event("legacy-bf16", 9, {
            projection: { op: "replace", run: legacyRun("legacy-bf16", 8) },
          }),
        );
        await emit(
          page,
          w8,
          event("legacy-w8a8", 9, {
            projection: { op: "replace", run: legacyRun("legacy-w8a8", 8) },
          }),
        );
        assert.doesNotMatch(await status("bf16").innerText(), /K\/D schedule/);
        assert.doesNotMatch(await status("w8a8").innerText(), /K\/D schedule/);
        legacyPrefix = 1;
        await page.selectOption("#execution-coupling", "synchronous");
        await page.waitForFunction(
          () =>
            document.querySelector("#data-status").dataset.state === "ready",
        );
        assert.doesNotMatch(
          await status("bf16").innerText(),
          /K\/D schedule/,
          "explicit cohort reload resolves a new matching schedule",
        );
        assert.doesNotMatch(await status("w8a8").innerText(), /K\/D schedule/);
      },
      {
        catalog: {
          ...catalog(false),
          default_view: "deployment",
          default_execution_coupling: "synchronous",
          default_execution_clock: "ready",
          runs: legacyDescriptors,
          streams: legacyDescriptors.map((r) => ({
            run_id: r.id,
            snapshot_path: `streams/${r.id}/snapshot.json`,
            journal_path: `streams/${r.id}/events.jsonl`,
          })),
        },
        run: legacyRun,
        snapshot: (id) => ({ ...snapshot(id), run: legacyRun(id) }),
      },
    );

    const policyDescriptors = ["bf16", "w8a8"].map((precision) => ({
      id: `policy-${precision}`,
      kind: "deployment",
      precision,
      execution_coupling: "asynchronous",
      execution_clock: "paced",
      path: `runs/policy-${precision}.json`,
      label: fixtureNote,
    }));
    const policyRun = (id, variant = "pir2") => ({
      ...run(id),
      ...policyDescriptors.find((d) => d.id === id),
      policy_variant: variant,
      episodes: [],
      clean_requests: [],
    });
    await check(
      "legacy SSE descriptors bind the policy variant and reject Flow4 or unknown replacements",
      async ({ page }) => {
        const bf16 = await waitForSource(page, "policy-bf16");
        const w8 = await waitForSource(page, "policy-w8a8");
        const status = (precision) =>
          page.locator(
            `.precision-card[data-precision="${precision}"] .card-status`,
          );
        assert.doesNotMatch(
          await status("bf16").innerText(),
          /model or controller policy/,
        );
        assert.equal(
          await page.locator("#execution-prefix-label").isVisible(),
          false,
        );
        assert.equal(
          await page.locator("#async-overlap-label").isVisible(),
          false,
        );
        await emit(
          page,
          bf16,
          event("policy-bf16", 8, {
            projection: {
              op: "replace",
              run: policyRun("policy-bf16", "flow"),
            },
          }),
        );
        assert.match(
          await status("bf16").innerText(),
          /model or controller policy/,
        );
        await emit(
          page,
          w8,
          event("policy-w8a8", 8, {
            projection: { op: "replace", run: policyRun("policy-w8a8", null) },
          }),
        );
        assert.match(
          await status("w8a8").innerText(),
          /model or controller policy/,
        );
        await emit(
          page,
          bf16,
          event("policy-bf16", 9, {
            projection: { op: "replace", run: policyRun("policy-bf16") },
          }),
        );
        assert.doesNotMatch(
          await status("bf16").innerText(),
          /model or controller policy/,
        );
        assert.match(
          await status("w8a8").innerText(),
          /model or controller policy/,
        );
      },
      {
        catalog: {
          ...catalog(false),
          default_view: "deployment",
          default_execution_coupling: "asynchronous",
          default_execution_clock: "paced",
          runs: policyDescriptors,
          streams: policyDescriptors.map((r) => ({
            run_id: r.id,
            snapshot_path: `streams/${r.id}/snapshot.json`,
            journal_path: `streams/${r.id}/events.jsonl`,
          })),
        },
        run: policyRun,
        snapshot: (id) => ({ ...snapshot(id), run: policyRun(id) }),
      },
    );

    const matrix = [
      [1, 1],
      [4, 1],
      [4, 2],
      [8, 1],
      [8, 4],
    ].flatMap(([k, d]) =>
      ["bf16", "w8a8", "w4a4"].map((precision) => ({
        id: `${precision}-k${k}-d${d}`,
        kind: "deployment",
        precision,
        execution_clock: "ready",
        execution_coupling: "chunked_async",
        execution_prefix_steps: k,
        async_overlap_steps: d,
        path: `runs/${precision}-k${k}-d${d}.json`,
        episodes: 0,
        label: fixtureNote,
      })),
    );
    await check(
      "matrix selectors load one cohort and keep K/D through populated and empty catalog refresh",
      async ({ page, catalog: data, requests }) => {
        const ready = () =>
          page.waitForFunction(
            () =>
              document.querySelector("#data-status").dataset.state === "ready",
          );
        const ids = () =>
          page
            .locator(".run-select")
            .evaluateAll((xs) => xs.map((x) => x.value));
        assert.deepEqual(await ids(), [
          "bf16-k4-d2",
          "w8a8-k4-d2",
          "w4a4-k4-d2",
        ]);
        assert.equal(
          requests.filter((p) => p.includes("/api/replay/run/")).length,
          3,
          "only the selected schedule is fetched",
        );
        await page.selectOption("#execution-prefix-steps", "8");
        await ready();
        assert.equal(
          await page.locator("#async-overlap-steps").inputValue(),
          "1",
          "invalid D2 falls back to recorded D1",
        );
        await page.selectOption("#async-overlap-steps", "4");
        await ready();
        assert.deepEqual(await ids(), [
          "bf16-k8-d4",
          "w8a8-k8-d4",
          "w4a4-k8-d4",
        ]);
        assert.equal(
          requests.filter((p) => p.includes("/api/replay/run/")).length,
          9,
        );
        const before = await selection(page);
        data.runs.push(
          ...matrix
            .filter(
              (r) =>
                r.execution_prefix_steps === 8 && r.async_overlap_steps === 4,
            )
            .map((r) => ({
              ...r,
              id: r.id + "-repeat",
              path: `runs/${r.id}-repeat.json`,
            })),
        );
        data.created_at = "matrix-repetition-added";
        await page.waitForSelector(
          '.run-select option[value="bf16-k8-d4-repeat"]',
          { state: "attached", timeout: 6500 },
        );
        assert.deepEqual(await selection(page), before);
        assert.equal(
          requests.filter((p) => p.includes("/api/replay/run/")).length,
          9,
          "refresh does not fetch unselected repetitions",
        );
        data.runs = [];
        data.created_at = "matrix-empty-refresh";
        await page.waitForFunction(
          () =>
            document
              .querySelector("#catalog-meta")
              .textContent.includes("matrix-empty-refresh"),
          null,
          { timeout: 6500 },
        );
        assert.equal(
          await page.locator("#execution-prefix-steps").inputValue(),
          "8",
        );
        assert.equal(
          await page.locator("#async-overlap-steps").inputValue(),
          "4",
        );
        assert.deepEqual(await selection(page), before);
        data.runs = matrix.filter((r) => r.execution_prefix_steps === 4);
        data.created_at = "matrix-selected-schedule-removed";
        await page.waitForFunction(
          () =>
            document.querySelector("#execution-prefix-steps").value === "4" &&
            document.querySelector("#data-status").dataset.state === "ready",
          null,
          { timeout: 6500 },
        );
        assert.equal(
          await page.locator("#async-overlap-steps").inputValue(),
          "1",
        );
        assert.deepEqual(await ids(), [
          "bf16-k4-d1",
          "w8a8-k4-d1",
          "w4a4-k4-d1",
        ]);
      },
      {
        catalog: {
          ...catalog(false),
          default_view: "deployment",
          default_execution_coupling: "chunked_async",
          default_execution_clock: "ready",
          default_execution_prefix_steps: 4,
          default_async_overlap_steps: 2,
          runs: matrix,
        },
        run: (id) => ({
          ...run(id),
          ...matrix.find((r) => r.id === id),
          episodes: [],
          clean_requests: [],
        }),
      },
    );

    await check(
      "legacy catalogs keep static replay and never probe SSE or snapshots",
      async ({ page, requests }) => {
        assert.equal(await page.locator(".precision-card").count(), 1);
        await page.selectOption("#sample", "contract-a-request-2");
        await seek(page, 500);
        await page.locator("#play").click();
        assert.equal(await page.locator("#play").innerText(), "暂停");
        assert.equal(await sourceCount(page), 0);
        assert.deepEqual(
          requests.filter((resource) => /\/(snapshot|events)\//.test(resource)),
          [],
        );
      },
      { catalog: catalog(false) },
    );

    console.log(
      JSON.stringify(
        {
          status: "passed",
          checks: results.length,
          fixture: fixtureNote,
          gpu: false,
          screenshots: false,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
