const { chromium } = require("../tools/ui-check/node_modules/playwright");
const fs = require("node:fs");
const path = require("node:path");

(async () => {
  const directory = path.resolve("artifacts/simulator");
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--disable-gpu"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1050 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("file://" + path.resolve("simulator/index.html"));
  await page.waitForFunction(() => !!window.lastQueueSimulation);
  await page.selectOption("#queue-executionPolicy", "libero");
  await page.waitForFunction(
    () => window.lastQueueSimulation?.config.executionPolicy === "libero",
  );
  const queueCases = [];
  for (const [preset, delay] of [
    ["1", 1],
    ["2", 1],
    ["3", 2],
    ["4", 2],
  ]) {
    await page.click(`[data-queue-preset="${preset}"]`);
    const state = await page.evaluate(() => ({
      d: window.lastQueueSimulation.result.initialDelay,
      expired: window.lastQueueSimulation.result.metrics.expiredOutputs,
      fallback: window.lastQueueSimulation.result.metrics.fallbackTicks,
    }));
    if (state.d !== delay || state.expired !== 0 || state.fallback !== 0)
      throw new Error(
        "Incorrect calibrated queue preset: " +
          JSON.stringify({ preset, state }),
      );
    queueCases.push({ preset, ...state });
  }
  await page.locator("#queue-time").fill("90");
  await page.locator("#queue-time").dispatchEvent("input");
  const ready = await page.evaluate(
    () => window.lastQueueSimulation.snapshot.action.ready,
  );
  if (!ready || ready.outputs.length !== 2)
    throw new Error(
      "80ms request must retain two ready actions before the 100ms tick",
    );
  await page.locator("#queue-time").fill("100");
  await page.locator("#queue-time").dispatchEvent("input");
  const executed = await page.evaluate(
    () => window.lastQueueSimulation.snapshot.execution.lastExecuted,
  );
  if (executed.origin.label !== "I-1a[0]" || executed.fallback)
    throw new Error(
      "Controller did not execute the first adopted action at 100ms",
    );
  await page.locator("#queue-vlmMs").fill("75.5");
  await page.locator("#queue-actionMs").fill("61.3");
  await page.waitForFunction(
    () =>
      window.lastQueueSimulation?.config.vlmMs === 75.5 &&
      window.lastQueueSimulation.config.actionMs === 61.3,
  );
  if (await page.locator("#queue-error").isVisible())
    throw new Error("Custom inference times failed");
  for (const value of ["61.3", "0.5"]) {
    const input = page.locator("#queue-actionMs");
    await input.click();
    await input.press("Control+A");
    await input.pressSequentially(value, { delay: 200 });
    await page.waitForFunction(
      (expected) => window.lastQueueSimulation?.config.actionMs === expected,
      Number(value),
    );
    if ((await input.inputValue()) !== value)
      throw new Error("Live normalization corrupted typed decimal latency");
    await input.press("Tab");
  }
  await page.click('[data-queue-preset="4"]');
  await page.selectOption("#queue-delayMode", "fixed");
  await page.waitForFunction(
    () => window.lastQueueSimulation?.config.delayMode === "fixed",
  );
  for (const [time, rejected] of [
    [90, false],
    [100, true],
  ]) {
    await page.locator("#queue-time").fill(String(time));
    await page.locator("#queue-time").dispatchEvent("input");
    const visible = await page.evaluate(() => {
      const { result, timeMs } = window.lastQueueSimulation;
      const output = result.actionRequests[0].outputs[0];
      return output.unused && output.unusedAtMs <= timeMs;
    });
    if (visible !== rejected)
      throw new Error("Output rejection leaked before controller publication");
  }
  await page.click('[data-queue-preset="4"]');
  const outputBatch = (requestId) =>
    page
      .locator(
        `#queue-timeline .queue-output-slot[data-request-id="${requestId}"]`,
      )
      .first();
  const moveTime = async (time) => {
    await page.locator("#queue-time").fill(String(time));
    await page.locator("#queue-time").dispatchEvent("input");
  };
  await moveTime(79);
  if (
    (await outputBatch(0).getAttribute("data-output-status")) !== "待生成" ||
    (await outputBatch(0).getAttribute("data-age-ms")) !== ""
  )
    throw new Error(
      "Unfinished output was presented as a fresh generated action",
    );
  await moveTime(90);
  if (
    (await outputBatch(0).getAttribute("data-age-ms")) !== "90" ||
    (await outputBatch(0).getAttribute("data-latest-completed")) !== "true" ||
    (await outputBatch(0).getAttribute("data-latest-batch")) !== "false"
  )
    throw new Error("Ready result freshness or adoption badge is incorrect");
  await moveTime(100);
  const latestSlot = page
    .locator('#queue-execution .queue-freshness-slot[data-request-id="0"]')
    .first();
  if (
    (await latestSlot.getAttribute("data-age-ms")) !== "100" ||
    (await latestSlot.getAttribute("data-latest-batch")) !== "true"
  )
    throw new Error("Newest adopted batch is not visibly distinguished");
  await moveTime(150);
  if (
    (await outputBatch(0).getAttribute("data-age-ms")) !== "150" ||
    (await outputBatch(0).locator(".unused-mark").count()) !== 0
  )
    throw new Error("Old state age was mistaken for rejected output");
  await moveTime(300);
  if (
    (await outputBatch(2).getAttribute("data-age-ms")) !== "100" ||
    (await outputBatch(2).getAttribute("data-vision-age-ms")) !== "300" ||
    (await outputBatch(2).getAttribute("data-latest-batch")) !== "true" ||
    (await outputBatch(0).getAttribute("data-latest-batch")) !== "false"
  )
    throw new Error(
      "Slot freshness did not retain the feature actually read by its request",
    );
  if (!process.argv.includes("--no-screenshots"))
    await page.screenshot({
      path: path.join(directory, "queues-freshness.png"),
      fullPage: true,
    });
  await page.click("#queue-shared-preset");
  await page.locator("details.queue-advanced > summary").click();
  for (const [id, value] of [
    ["queue-periodMs", "1000"],
    ["queue-cameraHz", "1"],
    ["queue-seconds", "0.5"],
  ])
    await page.locator(`#${id}`).fill(value);
  await page.waitForFunction(
    () =>
      window.lastQueueSimulation?.config.computeMode === "shared" &&
      window.lastQueueSimulation.config.periodMs === 1000 &&
      window.lastQueueSimulation.config.seconds === 0.5 &&
      window.lastQueueSimulation.config.cameraHz === 1,
  );
  const shared = await page.evaluate(() => {
    const r = window.lastQueueSimulation.result;
    return {
      a: r.actionRequests[0].finishMs,
      v: r.visualJobs[0].finishMs,
      overlap: r.metrics.gpuOverlapMs,
    };
  });
  if (shared.a !== 40 || shared.v !== 120 || shared.overlap !== 40)
    throw new Error(
      "Shared GPU did not resume solo speed after overlap: " +
        JSON.stringify(shared),
    );
  await page.selectOption("#queue-computeMode", "serial");
  await page.waitForFunction(
    () => window.lastQueueSimulation?.config.computeMode === "serial",
  );
  const serial = await page.evaluate(() => {
    const r = window.lastQueueSimulation.result;
    return {
      a: r.actionRequests[0].finishMs,
      v: r.visualJobs[0].finishMs,
      wait: r.visualJobs[0].gpuWaitMs,
      overlap: r.metrics.gpuOverlapMs,
    };
  });
  if (
    serial.a !== 20 ||
    serial.v !== 120 ||
    serial.wait !== 20 ||
    serial.overlap !== 0
  )
    throw new Error(
      "Serial GPU boundary scheduling mismatch: " + JSON.stringify(serial),
    );
  await page.selectOption("#queue-computeMode", "shared");
  await page.locator("#queue-gpuShareA").fill("0");
  await page.waitForFunction(
    () =>
      window.lastQueueSimulation?.config.gpuShareA === 0 &&
      window.lastQueueSimulation.config.computeMode === "shared",
  );
  const hostOnly = await page.evaluate(() => ({
    a: window.lastQueueSimulation.result.actionRequests[0].finishMs,
    v: window.lastQueueSimulation.result.visualJobs[0].finishMs,
  }));
  if (hostOnly.a !== 20 || hostOnly.v !== 100)
    throw new Error("Non-GPU work was incorrectly slowed by GPU contention");
  await page.click("#queue-shared-preset");
  await page.locator("#queue-time").fill("30");
  await page.locator("#queue-time").dispatchEvent("input");
  if (
    !(await page.locator("#queue-timeline").textContent()).includes("GPU 活跃")
  )
    throw new Error("GPU activity lane missing");
  if (!process.argv.includes("--no-screenshots"))
    await page.screenshot({
      path: path.join(directory, "queues-desktop.png"),
      fullPage: true,
    });
  const queueDownloadPromise = page.waitForEvent("download");
  await page.click("#queue-export");
  const queueDownload = await queueDownloadPromise;
  await queueDownload.saveAs(
    path.join(directory, "queue-scenario-export.json"),
  );
  const queueExport = JSON.parse(
    fs.readFileSync(path.join(directory, "queue-scenario-export.json")),
  );
  if (
    queueExport.kind !== "pir2_policy_queue_simulation" ||
    !queueExport.result.executions.length ||
    queueExport.config.computeMode !== "shared" ||
    !queueExport.result.resourceSegments.length
  )
    throw new Error("Queue export is missing execution provenance");
  await page.click("#lab-example-feedback");
  const feedback = await page.evaluate(() => {
    const s = window.lastQueueSimulation;
    return {
      baseD: s.reference.finalDelay,
      d: s.result.finalDelay,
      baseHz: s.reference.metrics.admittedVlmHz,
      hz: s.result.metrics.admittedVlmHz,
      camera: s.config.cameraHz,
    };
  });
  if (
    feedback.baseD !== 1 ||
    feedback.d !== 2 ||
    feedback.baseHz !== 5 ||
    feedback.hz !== 8 ||
    feedback.camera !== 20
  )
    throw new Error(
      "Admission/host feedback example regressed: " + JSON.stringify(feedback),
    );
  await page.click("#lab-compare-cap");
  const capped = await page.evaluate(() => window.lastQueueSimulation);
  if (
    capped.result.finalDelay !== 1 ||
    capped.result.metrics.admittedVlmHz !== 5 ||
    capped.config.cameraHz !== 20
  )
    throw new Error(
      "Admission cap changed capture rate or failed to bound work",
    );
  await page.click("#lab-example-latch");
  const latch = await page.evaluate(() => ({
    r: window.lastQueueSimulation.result,
    source: window.lastQueueSimulation.calibrationSource,
  }));
  if (
    latch.r.initialDelay !== 1 ||
    latch.r.finalDelay !== 2 ||
    latch.r.actionRequests[0].responseMs !== 51 ||
    latch.source !== null
  )
    throw new Error(
      "First dispatch overhead did not trigger persistent d transition",
    );
  await page.click('[data-queue-preset="1"]');
  await page.locator("#scenario-visionPrecision").selectOption("INT4 / W4A16");
  await page.waitForFunction(
    () =>
      window.lastQueueSimulation.scenario.visionPrecision === "INT4 / W4A16",
  );
  if (
    await page.evaluate(
      () =>
        window.lastQueueSimulation.effectiveConfig.vlmMs !==
        window.lastQueueSimulation.config.vlmMs,
    )
  )
    throw new Error("Precision label silently scaled latency");
  await page.locator("#scenario-visionSpeedup").fill("2");
  await page.waitForFunction(
    () => window.lastQueueSimulation.effectiveConfig.vlmMs === 15,
  );
  const dependent = page.locator('#queue-timeline [data-action-id="5"]');
  await dependent.dispatchEvent("click");
  const provenance = await page.evaluate(() => {
    const r = window.lastQueueSimulation.result,
      child = r.actionRequests[5];
    const highlighted = document.querySelector(
      "#queue-timeline .lab-selected-job[data-vision-id]",
    );
    const arrow = document.querySelector("#queue-timeline .lab-dependency");
    return {
      parent: child.featureVersion,
      highlight: highlighted ? Number(highlighted.dataset.visionId) : null,
      arrow: arrow ? Number(arrow.dataset.parentFeature) : null,
    };
  });
  if (
    provenance.parent !== provenance.highlight ||
    provenance.parent !== provenance.arrow
  )
    throw new Error(
      "Dependency highlight does not use actual consumed feature",
    );
  await page.locator("#queue-vlmMs").fill("140");
  await page.waitForFunction(
    () => window.lastQueueSimulation.config.vlmMs === 140,
  );
  if (await page.locator("#queue-timeline .lab-selected-job").count())
    throw new Error(
      "Selection retained an obsolete feature dependency after recompute",
    );
  if ((await page.locator("#lab-aoi-chart .lab-aoi-series").count()) !== 3)
    throw new Error(
      "Cache / executed image / executed state AoI series missing",
    );
  await page.locator("#scenario-visionSpeedup").fill("0");
  await page.waitForFunction(() => window.lastQueueSimulation === null);
  await page.selectOption("#lab-timeline-span", "600");
  await page.locator("#queue-timeline").dispatchEvent("click");
  await page.locator("#scenario-visionSpeedup").fill("2");
  await page.waitForFunction(() => window.lastQueueSimulation !== null);
  await page.click("#lab-example-repeat");
  const repeated = await page.evaluate(() => {
    const s = window.lastQueueSimulation,
      repeats = s.result.executions.filter((e) => e.repeated);
    const e = repeats.find(
      (e) =>
        e.origin.kind === "request" &&
        s.result.actionRequests[e.origin.requestId].featureVersion >= 0,
    );
    if (!e) return null;
    const r = s.result.actionRequests[e.origin.requestId];
    return {
      tick: e.tick,
      time: e.timeMs,
      id: r.id,
      offset: e.origin.offset,
      state: r.stateCaptureMs,
      image: r.featureCaptureMs,
      mode: s.config.executionPolicy,
      d: s.result.finalDelay,
    };
  });
  if (!repeated || repeated.mode !== "paper" || repeated.d !== 2)
    throw new Error("Old-segment repeat scenario missing provenance");
  await page.locator("#queue-time").fill(String(repeated.time + 10));
  await page.locator("#queue-time").dispatchEvent("input");
  const repeatState = await page.evaluate(() => {
    const s = window.lastQueueSimulation;
    return {
      e: s.snapshot.execution.lastExecuted,
      image: PiR2Scenarios.ageAt(s.aoi.executionImage, s.timeMs),
      state: PiR2Scenarios.ageAt(s.aoi.executionState, s.timeMs),
      live: document.querySelector("#lab-policy-live").textContent,
    };
  });
  if (
    !repeatState.e.repeated ||
    repeatState.e.origin.requestId !== repeated.id ||
    repeatState.image !== repeated.time + 10 - repeated.image ||
    repeatState.state !== repeated.time + 10 - repeated.state ||
    !repeatState.live.includes("重复末动作")
  )
    throw new Error("Repeated action lost source or reset AoI");
  if (
    (await page
      .locator('#queue-timeline [data-execution-mode="repeat-last"]')
      .count()) === 0 ||
    (await page.locator("#lab-aoi-chart .lab-aoi-repeat").count()) === 0
  )
    throw new Error("Repeat time ranges not exposed in charts");
  await page
    .locator(`#queue-timeline [data-execution-tick="${repeated.tick}"]`)
    .dispatchEvent("click");
  if (
    (await page
      .locator(
        `#queue-timeline .lab-selected-job[data-action-id="${repeated.id}"]`,
      )
      .count()) !== 1
  )
    throw new Error(
      "Clicking a repeated action did not highlight its original producer",
    );
  await page.click("#lab-example-recover-d");
  const recovery = await page.evaluate(() =>
    window.lastQueueSimulation.result.actionRequests
      .slice(0, 4)
      .map((r) => r.d),
  );
  if (JSON.stringify(recovery) !== "[1,3,2,1]")
    throw new Error(
      "Rolling d did not return after initial stall: " +
        JSON.stringify(recovery),
    );
  if (
    (await page.locator('#lab-delay-chart [data-series="delay"]').count()) !== 1
  )
    throw new Error("d timeline missing");
  await page.selectOption("#queue-executionPolicy", "libero");
  await page.waitForFunction(
    () => window.lastQueueSimulation?.config.executionPolicy === "libero",
  );
  if (
    (await page.evaluate(() => window.lastQueueSimulation.result.finalDelay)) <=
    1
  )
    throw new Error("LIBERO contrast lost monotone d");
  await page.click("#lab-example-repeat");
  const integratedUrl = process.argv
    .find((arg) => arg.startsWith("--url="))
    ?.slice(6);
  let integrated = null;
  if (integratedUrl) {
    const live = await browser.newPage({
      viewport: { width: 1440, height: 1050 },
    });
    live.on("pageerror", (error) => errors.push(error.message));
    await live.goto(integratedUrl);
    const iframe = live.frameLocator("#lab-replay-frame");
    await iframe.locator("#load-calibration").waitFor();
    await iframe.locator("#layout").selectOption("single");
    await iframe.locator("#condition").selectOption("core1200");
    const before = await live.evaluate(
      () => window.lastQueueSimulation.config.actionMs,
    );
    await live.evaluate(() =>
      window.postMessage(
        {
          type: "pir2:load-solo-calibration",
          version: 1,
          layout: "single",
          solo: { actionMs: 999, vlmMs: 999 },
          periodMs: 50,
          cameraHz: 20,
        },
        location.origin,
      ),
    );
    await live.waitForTimeout(30);
    if (
      (await live.evaluate(
        () => window.lastQueueSimulation.config.actionMs,
      )) !== before
    )
      throw new Error(
        "Parent accepted calibration from outside the report iframe",
      );
    await iframe.locator("#load-calibration").click();
    await live.waitForFunction(
      () =>
        window.lastQueueSimulation.calibrationSource?.conditionId ===
        "single/core1200",
    );
    if (!(await live.locator("#queue-mode").isVisible()))
      throw new Error("Calibration did not switch to scenario");
    integrated = await live.evaluate(() => ({
      source: window.lastQueueSimulation.calibrationSource,
      mode: window.lastQueueSimulation.config.computeMode,
      effective: window.lastQueueSimulation.effectiveConfig,
    }));
    if (
      integrated.mode !== "shared" ||
      integrated.effective.actionMs >= 100 ||
      integrated.effective.vlmMs <= 0
    )
      throw new Error("Solo calibration import failed");
    await live.click("#lab-tab-replay");
    await iframe.locator("#counterpart").click();
    if (
      (await iframe.locator("#layout").inputValue()) !== "dual" ||
      (await iframe.locator("#condition").inputValue()) !== "core1200"
    )
      throw new Error("Counterpart switch did not preserve frequency");
    await iframe.locator("#load-calibration").click();
    await live.waitForFunction(
      () => window.lastQueueSimulation.calibrationSource?.layout === "dual",
    );
    if (
      (await live.evaluate(
        () => window.lastQueueSimulation.config.computeMode,
      )) !== "independent"
    )
      throw new Error("Dual calibration not assigned independent GPUs");
    await live.close();
  }
  await page.setViewportSize({ width: 390, height: 844 });
  if (!process.argv.includes("--no-screenshots"))
    await page.screenshot({
      path: path.join(directory, "queues-mobile.png"),
      fullPage: true,
    });
  const queueLayout = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  if (queueLayout.scroll > queueLayout.width + 1)
    throw new Error("Queue layout overflows on mobile");
  if (errors.length) throw new Error(errors.join("\n"));
  fs.writeFileSync(
    path.join(directory, "browser-check.json"),
    JSON.stringify(
      {
        queueCases,
        shared,
        serial,
        hostOnly,
        queueLayout,
        feedback,
        repeated,
        recovery,
        integrated,
        errors,
      },
      null,
      2,
    ),
  );
  console.log(
    "PASS queue timing, resource modes, freshness, decimal inputs, exports and desktop/mobile layout",
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
