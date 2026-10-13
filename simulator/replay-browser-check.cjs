/* Explicit local browser check: prebuilt artifacts only, never model inference. */
const { chromium } = require("../tools/ui-check/node_modules/playwright");
const { spawn } = require("node:child_process");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const arg = (name) =>
  process.argv.find((v) => v.startsWith(name + "="))?.slice(name.length + 1);
const catalog = arg("--catalog");
const external = arg("--url");
if (!catalog && !external)
  throw new Error(
    "Use --catalog=/absolute/catalog.json or --url=http://127.0.0.1:port/simulator/replay.html",
  );
(async () => {
  let server, browser;
  try {
    let url = external;
    if (!url) {
      server = spawn(
        process.env.PYTHON || "python3",
        [
          "-u",
          "-c",
          "import importlib.util; s=importlib.util.spec_from_file_location('preview','simulator/serve.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); server=m.make_server(0,replay=" +
            JSON.stringify(catalog) +
            "); print(server.server_port,flush=True); server.serve_forever()",
        ],
        {
          cwd: path.resolve(__dirname, ".."),
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
      const port = await new Promise((resolve, reject) => {
        server.stdout.once("data", (d) => resolve(Number(d.toString().trim())));
        server.once("exit", (c) => reject(new Error("preview exited " + c)));
      });
      url = `http://127.0.0.1:${port}/simulator/replay.html`;
    }
    const output = path.resolve(
      __dirname,
      "../artifacts/simulator/replay-browser",
    );
    fs.mkdirSync(output, { recursive: true });
    browser = await chromium.launch({ headless: true, args: ["--no-sandbox"] });
    const page = await browser.newPage({
        viewport: { width: 1440, height: 1100 },
      }),
      errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(url);
    await page.waitForSelector(".precision-card");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    assert.equal(await page.locator(".precision-card").count(), 3);
    assert.equal(
      await page.locator("#evidence-mode").inputValue(),
      "deployment",
      "real robot view should be the visible default",
    );
    await page.selectOption("#evidence-mode", "model");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.locator("#seek").evaluate((e) => {
      e.value = e.max;
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForSelector("#curve svg");
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /动作已记录/,
    );
    await page.selectOption("#precision", "w8a8");
    assert.equal(await page.locator(".precision-card").count(), 1);
    await page.selectOption("#precision", "all");
    assert.equal(await page.locator(".precision-card").count(), 3);
    await page.selectOption("#record-type", "profile");
    await page.waitForSelector('[data-event-kind="kernel"]');
    await page.locator('[data-event-kind="kernel"]').first().click();
    assert.match(await page.locator("#inspector").innerText(), /kernel|Kernel/);
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /did not retain outputs|尚未产生/,
    );
    await page.selectOption("#evidence-mode", "deployment");
    await page.waitForSelector(".precision-card video");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".precision-card video")].every(
        (v) => v.readyState >= 2,
      ),
    );
    await page.locator("#play").click();
    const presented = [];
    for (let i = 0; i < 12; i++) {
      await page.waitForTimeout(200);
      presented.push(
        await page.locator(".precision-card").evaluateAll((cards) =>
          cards.map((card) => {
            const v = card.querySelector("video"),
              canvas = document.createElement("canvas");
            canvas.width = 32;
            canvas.height = 32;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(v, 0, 0, 32, 32);
            const pixels = ctx.getImageData(0, 0, 32, 32).data;
            let hash = 0;
            for (const x of pixels) hash = (hash * 31 + x) >>> 0;
            return {
              visible: getComputedStyle(v).visibility !== "hidden",
              hash,
              time: v.currentTime,
              presentedWall: Number(v.dataset.presentedWall),
              cursor: Number(document.querySelector("#seek").value),
            };
          }),
        ),
      );
    }
    await page.locator("#play").click();
    for (let p = 0; p < 3; p++) {
      assert.ok(
        new Set(presented.map((sample) => sample[p].hash)).size >= 5,
        "actual decoded pixels must move during PLAY",
      );
      assert.ok(
        presented.filter((sample) => sample[p].visible).length >= 9,
        "decoded video must stay visible during forward playback",
      );
      assert.ok(
        presented.every(
          (sample) => sample[p].presentedWall <= sample[p].cursor,
        ),
        "presented frames cannot come from a future cursor",
      );
    }
    fs.writeFileSync(
      path.join(output, "desktop-playback-check.json"),
      JSON.stringify(
        {
          samples: presented,
          summary: [0, 1, 2].map((p) => ({
            precision: ["bf16", "w8a8", "w4a4"][p],
            unique_pixel_hashes: new Set(
              presented.map((sample) => sample[p].hash),
            ).size,
            visible_samples: presented.filter((sample) => sample[p].visible)
              .length,
            total_samples: presented.length,
          })),
        },
        null,
        2,
      ),
    );
    const stopped = Number(await page.locator("#seek").inputValue());
    await page.waitForTimeout(250);
    assert.equal(Number(await page.locator("#seek").inputValue()), stopped);
    await page.locator("#play").click();
    await page.waitForTimeout(300);
    await page.locator("#play").click();
    assert.ok(Number(await page.locator("#seek").inputValue()) > stopped);
    await page.locator("#seek").evaluate((e) => {
      e.value = "2";
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".precision-card video")].every(
        (v) => v.readyState >= 2 && !v.seeking && v.currentTime > 0,
      ),
    );
    assert.equal(
      new Set(
        await page
          .locator(".precision-card video")
          .evaluateAll((vs) => vs.map((v) => v.src)),
      ).size,
      3,
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /回合最终结果/,
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /正在回放/,
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /当前动作的状态年龄/,
    );
    assert.equal(await page.locator(".lineage-row").count(), 3);
    assert.ok(
      (await page
        .locator('.lineage-arrow[data-from="cache"][data-to="action"]')
        .count()) >= 3,
    );
    await page
      .locator(".lineage-row")
      .first()
      .locator('[data-kind="action"]')
      .click();
    assert.match(
      await page.locator("#inspector").innerText(),
      /feature_sequence|feature_id/,
    );
    assert.equal(await page.locator(".lineage-row.selected").count(), 1);
    if (!process.argv.includes("--no-screenshots"))
      await page.screenshot({
        path: path.join(output, "desktop-lineage-2s.png"),
        fullPage: true,
      });
    await page.locator("#lineage-follow").click();
    // Rapid backward/forward seek keeps the shared clock authoritative.
    await page.locator("#seek").evaluate((e) => {
      for (const t of [3, 0.4, 2.4, 0.2]) {
        e.value = String(t);
        e.dispatchEvent(new Event("input", { bubbles: true }));
      }
    });
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".precision-card video")].every(
        (v) =>
          !v.seeking &&
          !v.classList.contains("pending") &&
          Number(v.dataset.presentedWall) <= 0.2,
      ),
    );
    await page.route("**/api/replay/media/*", async (route) => {
      await new Promise((r) => setTimeout(r, 100));
      await route.continue();
    });
    const oldSources = await page
      .locator(".precision-card video")
      .evaluateAll((vs) => vs.map((v) => v.src));
    await page.selectOption("#camera", "wrist_image");
    await page.waitForFunction(
      (old) =>
        [...document.querySelectorAll(".precision-card video")].every(
          (v, i) => v.src !== old[i] && v.readyState >= 2 && !v.seeking,
        ),
      oldSources,
    );
    await page.locator("#play").click();
    await page.waitForTimeout(650);
    await page.locator("#play").click();
    assert.ok(Number(await page.locator("#seek").inputValue()) > 0.6);
    await page.waitForFunction(() =>
      [...document.querySelectorAll(".precision-card video")].every(
        (v) =>
          Number(v.dataset.presentedWall) > 0.4 &&
          !v.classList.contains("pending"),
      ),
    );
    await page.unroute("**/api/replay/media/*");
    assert.match(
      await page.locator("#evidence-note").innerText(),
      /轨迹|observation/i,
    );
    await page.locator("#seek").evaluate((e) => {
      e.value = e.max;
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /失败|failed/,
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /时序无效|invalid/,
    );
    if (!process.argv.includes("--no-screenshots"))
      await page.screenshot({
        path: path.join(output, "desktop-robot.png"),
        fullPage: true,
      });
    // A delayed old model response must never overwrite a later robot selection.
    await page.route("**/api/replay/run/model-*", async (route) => {
      await new Promise((r) => setTimeout(r, 200));
      await route.continue();
    });
    await page.selectOption("#evidence-mode", "model");
    await page.selectOption("#evidence-mode", "deployment");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.waitForTimeout(300);
    assert.equal(await page.locator(".precision-card video").count(), 3);
    await page.unroute("**/api/replay/run/model-*");
    await page.selectOption("#evidence-mode", "model");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.route("**/api/replay/run/deployment-*", async (route) => {
      await new Promise((r) => setTimeout(r, 450));
      await route.continue();
    });
    await page.selectOption("#evidence-mode", "deployment");
    assert.equal(await page.locator("#precision").isDisabled(), true);
    await page.locator("#precision").evaluate((e) => {
      e.value = "w8a8";
      e.dispatchEvent(new Event("change", { bubbles: true }));
    });
    assert.equal(
      await page.locator(".precision-card").count(),
      0,
      "secondary change during loading cannot reuse old model runs",
    );
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    assert.ok(
      (
        await page
          .locator(".run-select")
          .evaluateAll((xs) => xs.map((x) => x.value))
      ).every((id) => id.startsWith("deployment-")),
    );
    await page.selectOption("#precision", "all");
    await page.unroute("**/api/replay/run/deployment-*");
    await page.route("**/api/replay/run/deployment-*", async (route) => {
      const response = await route.fetch(),
        run = await response.json();
      run.episodes = [
        {
          id: run.id + "/incomplete",
          status: "incomplete",
          reason: "Missing controller end window",
          metadata: { task_id: 4, episode_id: 2, seed: 1000, success: true },
          queue: null,
          ticks: [],
          robot: { status: "unavailable", frames: [], media: {} },
        },
      ];
      await route.fulfill({ json: run });
    });
    await page.selectOption("#evidence-mode", "model");
    await page.selectOption("#evidence-mode", "deployment");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    assert.match(await page.locator("#sample").innerText(), /Task 4/);
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /incomplete.*Missing controller end window/s,
    );
    assert.doesNotMatch(
      await page.locator("#precision-cards").innerText(),
      /正在回放|回合最终结果：成功|undefined/,
    );
    assert.equal(await page.locator("#play").isDisabled(), true);
    await page.unroute("**/api/replay/run/deployment-*");

    await page.route("**/api/replay/run/model-*", (route) =>
      route.fulfill({ status: 404, body: "missing" }),
    );
    await page.selectOption("#evidence-mode", "model");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /404|无法加载/,
    );
    assert.ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    );
    await page.unroute("**/api/replay/run/model-*");
    let withoutPairing = false;
    await page.route("**/api/replay/run/model-*", async (route) => {
      const response = await route.fetch(),
        run = await response.json(),
        source = run.clean_requests[0];
      run.clock_cohort = withoutPairing ? null : "scope-browser-fixture";
      run.clean_requests = [
        ["policy", 38],
        ["preparation_transfer", 8],
        ["prepared_model", 27],
        ["custom_unknown", 11],
      ].map(([scope, wall_ms]) => ({
        ...source,
        id: run.id + "/scope/" + scope,
        scope,
        wall_ms,
        comparison_key: withoutPairing
          ? null
          : "same-intentional-key-for-scope-regression",
        output:
          scope === "preparation_transfer"
            ? {
                status: "unavailable",
                reason: "Input preparation does not execute the model",
              }
            : source.output,
      }));
      await route.fulfill({ json: run });
    });
    await page.selectOption("#evidence-mode", "deployment");
    await page.selectOption("#evidence-mode", "model");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.selectOption("#record-type", "clean");
    for (const [scope, expected] of [
      ["policy", "38.00"],
      ["preparation_transfer", "8.00"],
      ["prepared_model", "27.00"],
    ]) {
      await page.selectOption("#scope", scope);
      assert.equal(
        await page
          .locator(".precision-card .metric strong")
          .first()
          .innerText(),
        expected,
      );
      assert.match(
        await page.locator("#sample").innerText(),
        new RegExp(scope),
      );
      if (scope === "preparation_transfer") {
        assert.equal(
          await page
            .locator(
              '.lineage-arrow[data-to="vlm"], .lineage-arrow[data-to="action"]',
            )
            .count(),
          0,
        );
        assert.match(
          await page.locator("#evidence-note").innerText(),
          /不执行 VLM\/DiT/,
        );
      }
    }
    await page.selectOption("#scope", "custom_unknown");
    assert.equal(
      await page
        .locator(
          '.lineage-arrow[data-to="vlm"], .lineage-arrow[data-to="action"]',
        )
        .count(),
      0,
    );
    assert.match(await page.locator("#evidence-note").innerText(), /边界未知/);
    withoutPairing = true;
    await page.selectOption("#evidence-mode", "deployment");
    await page.selectOption("#evidence-mode", "model");
    await page.waitForFunction(
      () => document.querySelector("#data-status").dataset.state === "ready",
    );
    await page.selectOption("#scope", "prepared_model");
    assert.match(
      await page.locator(".precision-card").first().innerText(),
      /未配对 \/ 比较条件未知/,
    );
    assert.equal(
      await page.locator(".precision-card .metric strong").first().innerText(),
      "27.00",
    );
    assert.match(
      await page.locator(".precision-card").nth(1).innerText(),
      /缺少同输入/,
    );
    await page.selectOption("#precision", "w8a8");
    await page.locator("#seek").evaluate((e) => {
      e.value = e.max;
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /动作已记录/,
    );
    assert.equal(
      await page.locator(".precision-card .metric strong").first().innerText(),
      "27.00",
    );
    assert.match(
      await page.locator("#precision-cards").innerText(),
      /未配对 \/ 比较条件未知/,
    );
    assert.deepEqual(errors, []);
    console.log(
      "Replay desktop, real WebM, selection, cursor, stale/error contracts passed.",
    );
  } finally {
    if (browser) await browser.close();
    if (server) server.kill("SIGTERM");
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
