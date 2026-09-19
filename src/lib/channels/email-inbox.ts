export interface InboundEmail {
  from: string;
  to: string;
  subject: string;
  text: string;
  receivedAt: string;
  messageId?: string;
}

/**
 * Poll IMAP inbox for supplier replies when CHINA_ACCESS_IMAP_* is set.
 * Returns [] when not configured (callers should use paste/webhook instead).
 */
export async function pollAgentInbox(options?: {
  sinceMinutes?: number;
  limit?: number;
}): Promise<{ configured: boolean; emails: InboundEmail[]; error?: string }> {
  const user = process.env.CHINA_ACCESS_IMAP_USER;
  const pass = process.env.CHINA_ACCESS_IMAP_PASS;
  const host = process.env.CHINA_ACCESS_IMAP_HOST || "imap.gmail.com";
  const port = Number(process.env.CHINA_ACCESS_IMAP_PORT || 993);

  if (!user || !pass) {
    return { configured: false, emails: [] };
  }

  try {
    const { ImapFlow } = await import("imapflow");
    const client = new ImapFlow({
      host,
      port,
      secure: true,
      auth: { user, pass },
      logger: false,
    });
    await client.connect();
    const emails: InboundEmail[] = [];
    const since = new Date(
      Date.now() - (options?.sinceMinutes ?? 60 * 24 * 7) * 60_000,
    );
    const lock = await client.getMailboxLock("INBOX");
    try {
      const limit = options?.limit ?? 20;
      let count = 0;
      for await (const msg of client.fetch(
        { since },
        { envelope: true, source: true },
      )) {
        if (count >= limit) break;
        const env = msg.envelope;
        const from = env?.from?.[0]?.address || "";
        const to = env?.to?.[0]?.address || user;
        const subject = env?.subject || "";
        const source = msg.source?.toString("utf8") || "";
        const text = stripMime(source);
        emails.push({
          from,
          to,
          subject,
          text,
          receivedAt: new Date(env?.date || Date.now()).toISOString(),
          messageId: env?.messageId,
        });
        count += 1;
      }
    } finally {
      lock.release();
    }
    await client.logout();
    return { configured: true, emails };
  } catch (e) {
    return {
      configured: true,
      emails: [],
      error: e instanceof Error ? e.message : "IMAP poll failed",
    };
  }
}

function stripMime(raw: string): string {
  // Prefer text/plain body if present
  const plain = raw.match(
    /Content-Type:\s*text\/plain[\s\S]*?\r?\n\r?\n([\s\S]*?)(?:\r?\n--|\r?\nContent-Type:)/i,
  );
  if (plain?.[1]) {
    return plain[1].replace(/=\r?\n/g, "").replace(/=([0-9A-F]{2})/gi, (_, h) =>
      String.fromCharCode(parseInt(h, 16)),
    ).trim();
  }
  const body = raw.split(/\r?\n\r?\n/).slice(1).join("\n\n");
  return body.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim().slice(0, 8000);
}

/** Match an inbound email to a deal via inquiry id, supplier name, or subject. */
export function scoreEmailForDeal(
  email: InboundEmail,
  deal: {
    id: string;
    title: string;
    providerName: string;
    outreach?: { inquiryId?: string; supplierName?: string; identityEmail?: string }[];
  },
): number {
  const blob = `${email.subject}\n${email.text}\n${email.from}`.toLowerCase();
  let score = 0;
  for (const o of deal.outreach || []) {
    if (o.inquiryId && blob.includes(o.inquiryId.toLowerCase())) score += 50;
    if (o.supplierName) {
      const token = o.supplierName.toLowerCase().split(/[\s(]/)[0];
      if (token && token.length > 3 && blob.includes(token)) score += 20;
    }
  }
  if (deal.providerName) {
    const token = deal.providerName.toLowerCase().split(/[\s(]/)[0];
    if (token && token.length > 3 && blob.includes(token)) score += 15;
  }
  const titleTok = deal.title.toLowerCase().split(/\s+/).filter((t) => t.length > 4);
  for (const t of titleTok.slice(0, 4)) {
    if (blob.includes(t)) score += 5;
  }
  if (/made-in-china|inquiry|quote|fob|moq/i.test(blob)) score += 5;
  return score;
}
