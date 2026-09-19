import { chromium, type Page } from "playwright";

export interface SupplierContacts {
  whatsapp?: string; // E.164-ish digits
  whatsappRaw?: string;
  wechat?: string;
  email?: string;
  phone?: string;
  contactPerson?: string;
  pageUrl: string;
}

function normalizeWhatsApp(raw: string): string | undefined {
  const digits = raw.replace(/[^\d]/g, "");
  if (digits.length < 8 || digits.length > 15) return undefined;
  return digits;
}

export function extractContactsFromText(
  text: string,
  pageUrl: string,
): SupplierContacts {
  const out: SupplierContacts = { pageUrl };

  const wa =
    text.match(/wa\.me\/(\+?\d{8,15})/i) ||
    text.match(/api\.whatsapp\.com\/send\?phone=(\d{8,15})/i) ||
    text.match(/WhatsApp[:\s]*([+\d][\d\s-]{7,20}\d)/i);
  if (wa) {
    out.whatsappRaw = wa[1];
    out.whatsapp = normalizeWhatsApp(wa[1]!);
  }

  const wechat =
    text.match(/WeChat[:\sID]*[:\s]*([A-Za-z][A-Za-z0-9_-]{4,30})/i) ||
    text.match(/微信[:\s]*([A-Za-z0-9_-]{5,30})/);
  if (wechat) out.wechat = wechat[1];

  const emails = [
    ...text.matchAll(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi),
  ].map((m) => m[0]);
  const email = emails.find(
    (e) => !/made-in-china|example\.com|sentry/i.test(e),
  );
  if (email) out.email = email;

  const phone = text.match(
    /(?:Tel|Phone|Mobile|电话|手机)[:\s]*([+\d][\d\s()-]{8,20}\d)/i,
  );
  if (phone) out.phone = phone[1]!.replace(/\s+/g, " ").trim();

  const person = text.match(/\b(Ms\.|Mr\.|Mrs\.|Miss)\s+([A-Z][a-z]+)/);
  if (person) out.contactPerson = `${person[1]} ${person[2]}`;

  return out;
}

async function withPage<T>(fn: (page: Page) => Promise<T>): Promise<T> {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({
      userAgent:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    });
    page.setDefaultTimeout(40000);
    return await fn(page);
  } finally {
    await browser.close().catch(() => undefined);
  }
}

export async function scrapeSupplierContacts(
  listingOrContactUrl: string,
): Promise<SupplierContacts> {
  let contactUrl = listingOrContactUrl;
  try {
    const u = new URL(listingOrContactUrl);
    if (/\.en\.made-in-china\.com$/i.test(u.hostname)) {
      contactUrl = `${u.protocol}//${u.hostname}/contact-info.html`;
    }
  } catch {
    /* keep */
  }

  return withPage(async (page) => {
    await page.goto(contactUrl, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(2500);
    const text = await page.locator("body").innerText();
    const hrefs = await page.$$eval("a[href]", (as) =>
      as.map((a) => (a as HTMLAnchorElement).href).slice(0, 80),
    );
    const merged = `${text}\n${hrefs.join("\n")}`;
    const contacts = extractContactsFromText(merged, contactUrl);

    const person = await page
      .locator(".sr-sendMsg-name, .sr-sendMsg-name-ltr")
      .first()
      .innerText()
      .catch(() => undefined);
    if (person) contacts.contactPerson = person.trim();

    return contacts;
  });
}
