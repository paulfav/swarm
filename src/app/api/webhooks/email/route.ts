import { NextResponse } from "next/server";
import { ingestWebhookEmail } from "@/lib/replies";
import { listDealsAsync, saveDeal } from "@/lib/store";
import type { InboundEmail } from "@/lib/channels/email-inbox";

export const maxDuration = 60;

/**
 * Inbound email webhook (SendGrid/Resend/Mailgun-compatible JSON).
 * Matches the best deal and ingests the reply.
 */
export async function POST(request: Request) {
  const secret = process.env.CHINA_ACCESS_WEBHOOK_SECRET;
  if (secret) {
    const hdr = request.headers.get("x-china-access-secret");
    if (hdr !== secret) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
  }

  const body = (await request.json()) as Partial<InboundEmail> & {
    html?: string;
  };
  const email: InboundEmail = {
    from: body.from || "",
    to: body.to || "",
    subject: body.subject || "",
    text: body.text || body.html?.replace(/<[^>]+>/g, " ") || "",
    receivedAt: body.receivedAt || new Date().toISOString(),
    messageId: body.messageId,
  };

  if (!email.text.trim()) {
    return NextResponse.json({ error: "empty email" }, { status: 400 });
  }

  const deals = await listDealsAsync();
  let best: { id: string; score: number } | null = null;
  const { scoreEmailForDeal } = await import("@/lib/channels/email-inbox");
  for (const d of deals) {
    const score = scoreEmailForDeal(email, d);
    if (!best || score > best.score) best = { id: d.id, score };
  }
  if (!best || best.score < 10) {
    return NextResponse.json({
      matched: false,
      reason: "No deal scored high enough",
    });
  }

  const deal = deals.find((d) => d.id === best!.id)!;
  const updated = ingestWebhookEmail(deal, email);
  if (!updated) {
    return NextResponse.json({ matched: false, reason: "ingest skipped" });
  }
  saveDeal(updated);
  return NextResponse.json({ matched: true, dealId: updated.id, deal: updated });
}
