import { NextResponse } from "next/server";
import { ingestSupplierReply } from "@/lib/replies";
import { listDealsAsync, saveDeal } from "@/lib/store";

export const maxDuration = 60;

/**
 * Twilio WhatsApp inbound webhook (form-encoded).
 * Matches by WhatsApp From number to deal.contacts.whatsapp.
 */
export async function POST(request: Request) {
  const form = await request.formData();
  const from = String(form.get("From") || "").replace(/^whatsapp:/i, "");
  const body = String(form.get("Body") || "");
  const digits = from.replace(/[^\d]/g, "");

  if (!body.trim()) {
    return new NextResponse("<Response></Response>", {
      headers: { "Content-Type": "text/xml" },
    });
  }

  const deals = await listDealsAsync();
  const deal = deals.find((d) => {
    const wa = d.contacts?.whatsapp?.replace(/[^\d]/g, "");
    return wa && (wa === digits || digits.endsWith(wa) || wa.endsWith(digits));
  });

  if (deal) {
    const updated = ingestSupplierReply(deal, {
      text: body,
      channel: "whatsapp",
      from: digits,
    });
    saveDeal(updated);
  }

  return new NextResponse("<Response></Response>", {
    headers: { "Content-Type": "text/xml" },
  });
}
