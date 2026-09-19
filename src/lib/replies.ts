import {
  pollAgentInbox,
  scoreEmailForDeal,
  type InboundEmail,
} from "./channels/email-inbox";
import {
  applyParsedQuoteToDealQuote,
  parseSupplierReply,
  retranscribeReply,
} from "./parse-quote";
import type { Deal, ThreadMessage } from "./types";

function id(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function ingestSupplierReply(
  deal: Deal,
  input: {
    text: string;
    channel?: ThreadMessage["channel"];
    from?: string;
    subject?: string;
    receivedAt?: string;
  },
): Deal {
  const now = input.receivedAt || new Date().toISOString();
  const parsed = parseSupplierReply(input.text);
  const retranscription = retranscribeReply(input.text, parsed);
  const newQuote = applyParsedQuoteToDealQuote(
    deal.quote,
    parsed,
    deal.destinationCountry,
  );

  const msg: ThreadMessage = {
    id: id("msg"),
    at: now,
    direction: "inbound",
    channel: input.channel || "email",
    author: "supplier",
    body: input.text,
    retranscription,
    meta: {
      ...(input.from ? { from: input.from } : {}),
      ...(input.subject ? { subject: input.subject } : {}),
      ...(parsed.factoryPriceUsd != null
        ? { fob_usd: String(parsed.factoryPriceUsd) }
        : {}),
    },
  };

  const timeline = [
    ...deal.timeline,
    {
      id: id("evt"),
      at: now,
      actor: "factory" as const,
      kind: "answered" as const,
      title: "Supplier reply received",
      body: retranscription,
      facts: {
        channel: msg.channel,
        ...(parsed.factoryPriceUsd != null
          ? { fob_usd: `$${parsed.factoryPriceUsd}` }
          : {}),
        ...(parsed.moq != null ? { moq: String(parsed.moq) } : {}),
        ...(parsed.leadTimeDays != null
          ? { lead_days: String(parsed.leadTimeDays) }
          : {}),
      },
    },
  ];

  // Refresh freight module cbm in blueprint if parsed
  let blueprint = deal.blueprint;
  if (parsed.cbm != null) {
    blueprint = {
      ...deal.blueprint,
      modules: deal.blueprint.modules.map((m) =>
        m.type === "freight_volume" ? { ...m, cbm: parsed.cbm! } : m,
      ),
    };
  }

  return {
    ...deal,
    updatedAt: now,
    status: newQuote ? "quoted" : "awaiting_reply",
    quote: newQuote ?? deal.quote,
    blueprint,
    thread: [...(deal.thread ?? []), msg],
    timeline,
  };
}

export function appendOutboundThread(
  deal: Deal,
  msg: Omit<ThreadMessage, "id" | "direction" | "author"> & {
    author?: ThreadMessage["author"];
  },
): Deal {
  const full: ThreadMessage = {
    id: id("msg"),
    direction: "outbound",
    author: msg.author || "agent",
    at: msg.at,
    channel: msg.channel,
    body: msg.body,
    retranscription: msg.retranscription,
    meta: msg.meta,
  };
  return {
    ...deal,
    thread: [...(deal.thread ?? []), full],
  };
}

export async function pollAndIngestReplies(deal: Deal): Promise<{
  deal: Deal;
  matched: number;
  inboxConfigured: boolean;
  error?: string;
}> {
  const inbox = await pollAgentInbox({ sinceMinutes: 60 * 24 * 14, limit: 30 });
  if (!inbox.configured) {
    return {
      deal,
      matched: 0,
      inboxConfigured: false,
      error:
        inbox.error ||
        "No inbox configured. Set CHINA_ACCESS_MAILTM_* or CHINA_ACCESS_IMAP_*.",
    };
  }
  if (inbox.error) {
    return {
      deal,
      matched: 0,
      inboxConfigured: true,
      error: inbox.error,
    };
  }

  const existingIds = new Set(
    (deal.thread || [])
      .map((m) => m.meta?.messageId)
      .filter(Boolean) as string[],
  );

  let updated = deal;
  let matched = 0;
  const ranked = inbox.emails
    .map((e) => ({ email: e, score: scoreEmailForDeal(e, deal) }))
    .filter((x) => x.score >= 15)
    .sort((a, b) => b.score - a.score);

  for (const { email } of ranked) {
    if (email.messageId && existingIds.has(email.messageId)) continue;
    // Skip our own outbound echoes roughly
    if (/china access agent/i.test(email.text) && /please reply with/i.test(email.text)) {
      continue;
    }
    updated = ingestSupplierReply(updated, {
      text: `Subject: ${email.subject}\n\n${email.text}`,
      channel: "email",
      from: email.from,
      subject: email.subject,
      receivedAt: email.receivedAt,
    });
    const last = updated.thread?.[updated.thread.length - 1];
    if (last && email.messageId) {
      last.meta = { ...last.meta, messageId: email.messageId };
    }
    matched += 1;
    if (email.messageId) existingIds.add(email.messageId);
  }

  return { deal: updated, matched, inboxConfigured: true };
}

export function ingestWebhookEmail(
  deal: Deal,
  email: InboundEmail,
): Deal | null {
  const score = scoreEmailForDeal(email, deal);
  if (score < 10) return null;
  return ingestSupplierReply(deal, {
    text: `Subject: ${email.subject}\n\n${email.text}`,
    channel: "email",
    from: email.from,
    subject: email.subject,
    receivedAt: email.receivedAt,
  });
}
