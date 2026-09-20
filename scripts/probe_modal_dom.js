const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    userAgent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
  });
  const email = "paul.faverjon@student-cs.fr";
  const message =
    "Hello Ms. He, we are China Access sourcing agent for a US client. Please quote FOB for one cloud-style 3-seat fabric sofa ~220cm, MOQ 1, lead time, fabric options, CBM for USA export. Thank you - China Access agent.";

  await page.goto(
    "https://yefeifurniture.en.made-in-china.com/contact-info.html",
    { waitUntil: "domcontentloaded", timeout: 60000 },
  );
  await page.waitForTimeout(2000);
  await page.fill("#J-quick-inquiry-input", email);
  await page.fill("#inquiryContent", message);
  await page.click("#inquirySend");
  await page.waitForTimeout(4000);

  const frames = page.frames().map((f) => f.url());
  console.log("frames", frames);

  for (const frame of page.frames()) {
    const inputs = await frame
      .$$eval("input,textarea,select,button", (els) =>
        els.map((el) => ({
          name: el.name,
          id: el.id,
          ph: el.placeholder,
          type: el.type,
          text: (el.innerText || el.value || "").slice(0, 40),
          visible: !!(el.offsetWidth || el.offsetHeight),
        })),
      )
      .catch(() => []);
    if (inputs.length) {
      console.log("FRAME", frame.url(), JSON.stringify(inputs, null, 2));
    }
  }

  // Deep search including shadow DOM
  const deep = await page.evaluate(() => {
    function walk(root, acc = []) {
      const els = root.querySelectorAll
        ? root.querySelectorAll("input,textarea,select,button")
        : [];
      for (const el of els) {
        acc.push({
          name: el.name,
          id: el.id,
          ph: el.placeholder,
          type: el.type,
          text: (el.innerText || "").slice(0, 30),
        });
      }
      const all = root.querySelectorAll ? root.querySelectorAll("*") : [];
      for (const el of all) {
        if (el.shadowRoot) walk(el.shadowRoot, acc);
      }
      return acc;
    }
    return walk(document);
  });
  console.log("deep", JSON.stringify(deep, null, 2));

  // HTML of dialog container
  const dlg = await page.evaluate(() => {
    const el = [...document.querySelectorAll("div")].find((d) =>
      /build trust/i.test(d.innerText || "") &&
      (d.innerText || "").length < 1500,
    );
    return el
      ? { cls: el.className.toString().slice(0, 100), html: el.innerHTML.slice(0, 4000) }
      : null;
  });
  console.log("dlg", JSON.stringify(dlg, null, 2));

  await page.screenshot({
    path: "/workspace/scripts/outreach_modal2.png",
    fullPage: true,
  });
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
