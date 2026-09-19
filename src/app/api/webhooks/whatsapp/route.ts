import { NextResponse } from "next/server";
import { ingestSupplierReply } from "@/lib/replies";
import { listDealsAsync, saveDeal } from "@/lib/store";

export const maxDuration = 60;

function digitsOf(value: string): string {
  return value.replace(/[^\d]/g, "");
}

function findDealByWhatsApp(digits: string) {
  return listDealsAsync().then((deals) =>
    deals.find((d) => {
      const wa = d.contacts?.whatsapp?.replace(/[^\d]/g, "");
      return (
        wa && (wa === digits || digits.endsWith(wa) || wa.endsWith(digits))
      );
    }),
  );
}

/**
 * Inbound WhatsApp webhook.
 * - Twilio: form-encoded From/Body
 * - Green-API: JSON incomingMessageReceived
 */
export async function POST(request: Request) {
  const contentType = request.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const payload = (await request.json()) as {
      typeWebhook?: string;
      senderData?: { sender?: string; chatId?: string };
      messageData?: {
        textMessageData?: { textMessage?: string };
        extendedTextMessageData?: { text?: string };
      };
      idMessage?: string;
      timestamp?: number;
    };

    if (payload.typeWebhook === "incomingMessageReceived") {
      const text =
        payload.messageData?.textMessageData?.textMessage ||
        payload.messageData?.extendedTextMessageData?.text ||
        "";
      const sender =
        payload.senderData?.sender || payload.senderData?.chatId || "";
      const digits = digitsOf(sender);
      if (text.trim() && digits) {
        const deal = await findDealByWhatsApp(digits);
        if (deal) {
          const updated = ingestSupplierReply(deal, {
            text,
            channel: "whatsapp",
            from: digits,
            receivedAt: payload.timestamp
              ? new Date(payload.timestamp * 1000).toISOString()
              : undefined,
          });
          const last = updated.thread?.[updated.thread.length - 1];
          if (last && payload.idMessage) {
            last.meta = { ...last.meta, messageId: payload.idMessage };
          }
          saveDeal(updated);
        }
      }
    }
    return NextResponse.json({ ok: true });
  }

  const form = await request.formData();
  const from = String(form.get("From") || "").replace(/^whatsapp:/i, "");
  const body = String(form.get("Body") || "");
  const digits = digitsOf(from);

  if (!body.trim()) {
    return new NextResponse("<Response></Response>", {
      headers: { "Content-Type": "text/xml" },
    });
  }

  const deal = await findDealByWhatsApp(digits);
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
