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
  await page.waitForFunction(() => !!window.lastSimulation);
  await page.screenshot({
    path: path.join(directory, "desktop.png"),
    fullPage: true,
  });
  const initial = await page.evaluate(() => ({
    miss: window.lastSimulation.result.missRate,
    service: window.lastSimulation.result.service.S2.total,
  }));
  await page.selectOption("#qV", "int8");
  await page.waitForFunction(() => window.lastSimulation.config.qV === "int8");
  const quantized = await page.evaluate(
    () => window.lastSimulation.result.service.S2.total,
  );
  if (!(quantized < initial.service))
    throw new Error("Quantization did not update the service chart");
  await page.click('[data-preset="stress"]');
  await page.waitForFunction(() => window.lastSimulation.config.period === 20);
  const stressed = await page.evaluate(
    () => window.lastSimulation.result.missRate,
  );
  if (!(stressed > initial.miss))
    throw new Error("Pressure preset did not increase deadline misses");
  await page.click('[data-preset="aoi"]');
  const aoi = await page.evaluate(() => ({
    actual: window.lastSimulation.result.aoiMean,
    expected: window.lastSimulation.result.theoreticalAoi,
  }));
  if (Math.abs(aoi.actual - aoi.expected) > 0.1)
    throw new Error("AoI demonstration does not match deterministic formula");
  await page.selectOption("#hardware", "edge");
  await page.waitForFunction(
    () => window.lastSimulation.config.compute === 0.25,
  );
  if (await page.locator("#error").isVisible())
    throw new Error("Simulation error on edge scenario");
  await page.click("#reset");
  const downloadPromise = page.waitForEvent("download");
  await page.click("#export");
  const download = await downloadPromise;
  await download.saveAs(path.join(directory, "scenario-export.json"));
  const exported = JSON.parse(
    fs.readFileSync(path.join(directory, "scenario-export.json")),
  );
  if (exported.kind !== "mechanism_simulation_not_hardware_prediction")
    throw new Error("Missing assumption label in export");
  await page.click('[data-qa="w4"][data-qv="int8"]');
  if ((await page.evaluate(() => window.lastSimulation.config.qA)) !== "w4")
    throw new Error("Sweep selection failed");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.click("#reset");
  await page.screenshot({
    path: path.join(directory, "mobile.png"),
    fullPage: true,
  });
  const layout = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  if (layout.scroll > layout.width + 1)
    throw new Error("Mobile horizontal overflow: " + JSON.stringify(layout));
  if (errors.length) throw new Error(errors.join("\n"));
  fs.writeFileSync(
    path.join(directory, "browser-check.json"),
    JSON.stringify(
      { initial, quantized, stressed, aoi, layout, errors },
      null,
      2,
    ),
  );
  console.log(
    "PASS desktop/mobile, precision controls, pressure/AoI/hardware presets, heatmap selection and JSON export",
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
