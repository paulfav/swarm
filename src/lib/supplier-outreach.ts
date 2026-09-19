import { chromium, type Browser, type Page } from "playwright";

export interface AgentIdentity {
  email: string;
  fullName: string;
  company: string;
  mobile: string;
  mobileCountryCode?: string;
}

export interface OutreachInput {
  /** Made-in-China supplier showroom / product / contact URL */
  listingUrl: string;
  supplierName?: string;
  productTitle?: string;
  message: string;
  identity?: Partial<AgentIdentity>;
}

export interface OutreachResult {
  ok: boolean;
  channel: "made-in-china-inquiry";
  supplierName?: string;
  contactPerson?: string;
  listingUrl: string;
  message: string;
  identityEmail: string;
  successUrl?: string;
  inquiryId?: string;
  steps: string[];
  error?: string;
  sentAt: string;
}

function defaultIdentity(): AgentIdentity {
  return {
    email:
      process.env.CHINA_ACCESS_AGENT_EMAIL ||
      "paul.faverjon@student-cs.fr",
    fullName: process.env.CHINA_ACCESS_AGENT_NAME || "China Access Agent",
    company: process.env.CHINA_ACCESS_AGENT_COMPANY || "China Access",
    mobile: process.env.CHINA_ACCESS_AGENT_MOBILE || "5550100123",
    mobileCountryCode: process.env.CHINA_ACCESS_AGENT_MOBILE_CC || "1",
  };
}

function contactInfoUrl(listingUrl: string): string {
  try {
    const u = new URL(listingUrl);
    if (/\.en\.made-in-china\.com$/i.test(u.hostname)) {
      return `${u.protocol}//${u.hostname}/contact-info.html`;
    }
  } catch {
    /* ignore */
  }
  return listingUrl;
}

function extractInquiryId(successUrl?: string): string | undefined {
  if (!successUrl) return undefined;
  const m = successUrl.match(/success_([A-Za-z0-9]+)/);
  return m?.[1];
}

async function withBrowser<T>(fn: (page: Page) => Promise<T>): Promise<T> {
  let browser: Browser | null = null;
  try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
      locale: "en-US",
    });
    page.setDefaultTimeout(45000);
    return await fn(page);
  } finally {
    await browser?.close().catch(() => undefined);
  }
}

/**
 * Send a real Made-in-China inquiry to the supplier contact (e.g. Ms. He).
 * Uses the showroom contact form → quick-form iframe → success page.
 */
export async function contactMadeInChinaSupplier(
  input: OutreachInput,
): Promise<OutreachResult> {
  const identity = { ...defaultIdentity(), ...input.identity };
  const listingUrl = input.listingUrl;
  const contactUrl = contactInfoUrl(listingUrl);
  const steps: string[] = [];
  const sentAt = new Date().toISOString();

  try {
    return await withBrowser(async (page) => {
      await page.goto(contactUrl, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(2000);
      steps.push(`opened ${contactUrl}`);

      const contactPerson = await page
        .locator(".sr-sendMsg-name, .sr-sendMsg-name-ltr")
        .first()
        .innerText()
        .catch(() => undefined);
      const supplierName =
        input.supplierName ||
        (await page.locator("h1, .company-name, .sr-company-name").first().innerText().catch(() => undefined)) ||
        undefined;

      const emailInput = page.locator("#J-quick-inquiry-input");
      const contentInput = page.locator("#inquiryContent");
      if ((await emailInput.count()) === 0 || (await contentInput.count()) === 0) {
        throw new Error("Made-in-China contact form not found on page");
      }

      await emailInput.fill(identity.email);
      await contentInput.fill(input.message);
      steps.push("filled inquiry email + message");

      await page.click("#inquirySend");
      steps.push("clicked Send");

      const frameHandle = await page.waitForSelector(
        'iframe[src*="showQuickForm"]',
        { timeout: 20000 },
      );
      const frame = await frameHandle.contentFrame();
      if (!frame) throw new Error("Quick-form iframe failed to load");
      await frame.waitForSelector("#senderMail", { timeout: 15000 });
      steps.push("trust modal iframe ready");

      await frame.fill("#senderMail", identity.email);
      await frame.locator('input[name="senderName"]').fill(identity.fullName);
      await frame.locator('input[name="senderCom"]').fill(identity.company);
      await frame.fill("#senderMobile", identity.mobile);
      if (identity.mobileCountryCode) {
        await frame
          .locator("#senderMobileCountryCode")
          .fill(identity.mobileCountryCode)
          .catch(() => undefined);
      }
      steps.push("filled buyer identity in trust modal");

      let successUrl: string | undefined;
      page.on("response", (r) => {
        if (/sendInquiry\/success_/i.test(r.url())) {
          successUrl = r.url();
        }
      });

      await Promise.all([
        frame.waitForNavigation({ timeout: 25000 }).catch(() => null),
        frame.click("#submitSend"),
      ]);
      await page.waitForTimeout(4000);
      steps.push("submitted inquiry");

      // Success may render inside iframe
      let successText = "";
      for (const f of page.frames()) {
        const t = await f
          .locator("body")
          .innerText()
          .catch(() => "");
        if (/sent successfully|thank you|has been sent/i.test(t)) {
          successText = t.slice(0, 500);
          if (!successUrl) successUrl = f.url();
        }
      }
      const parentText = await page.locator("body").innerText();
      if (/sent successfully/i.test(parentText)) {
        successText = successText || parentText.slice(0, 500);
      }

      const ok = Boolean(
        successUrl || /sent successfully|has been sent/i.test(successText),
      );
      if (!ok) {
        throw new Error(
          "Inquiry submit finished without Made-in-China success confirmation",
        );
      }
      steps.push("Made-in-China confirmed Sent Successfully");

      return {
        ok: true,
        channel: "made-in-china-inquiry" as const,
        supplierName,
        contactPerson: contactPerson?.trim(),
        listingUrl,
        message: input.message,
        identityEmail: identity.email,
        successUrl,
        inquiryId: extractInquiryId(successUrl),
        steps,
        sentAt,
      };
    });
  } catch (e) {
    return {
      ok: false,
      channel: "made-in-china-inquiry",
      listingUrl,
      message: input.message,
      identityEmail: identity.email,
      steps,
      error: e instanceof Error ? e.message : "Outreach failed",
      sentAt,
    };
  }
}

export function buildQuoteRequestMessage(input: {
  productTitle: string;
  description: string;
  quantity: number;
  destinationCountry: string;
  contactName?: string;
}): string {
  const who = input.contactName ? `Hello ${input.contactName},` : "Hello,";
  return [
    who,
    "",
    "We are China Access, a sourcing agent helping a hard-goods client.",
    `Product: ${input.productTitle}`,
    `Brief: ${input.description.slice(0, 400)}`,
    `Quantity: ${input.quantity}`,
    `Destination: ${input.destinationCountry}`,
    "",
    "Please reply with:",
    "- FOB USD price",
    "- MOQ (ideally 1 unit for this order)",
    "- Lead time (days after deposit)",
    "- Material / finish options",
    "- Export packing CBM",
    "",
    "Thank you,",
    "China Access agent",
  ].join("\n");
}
