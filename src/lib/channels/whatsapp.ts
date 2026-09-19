export interface WhatsAppSendInput {
  to: string; // digits
  body: string;
}

export interface WhatsAppSendResult {
  ok: boolean;
  channel: "whatsapp";
  to: string;
  body: string;
  provider: "twilio" | "deep_link" | "none";
  messageSid?: string;
  deepLink?: string;
  error?: string;
  sentAt: string;
}

function e164(to: string): string {
  const d = to.replace(/[^\d]/g, "");
  return d.startsWith("+") ? d : `+${d}`;
}

/**
 * Send WhatsApp via Twilio if configured; otherwise return a wa.me deep link
 * the agent/ops can open. Never fakes a successful Twilio send.
 */
export async function sendWhatsApp(
  input: WhatsAppSendInput,
): Promise<WhatsAppSendResult> {
  const sentAt = new Date().toISOString();
  const to = input.to.replace(/[^\d]/g, "");
  const deepLink = `https://wa.me/${to}?text=${encodeURIComponent(input.body)}`;

  const sid = process.env.TWILIO_ACCOUNT_SID;
  const token = process.env.TWILIO_AUTH_TOKEN;
  const from = process.env.TWILIO_WHATSAPP_FROM; // e.g. whatsapp:+14155238886

  if (!sid || !token || !from) {
    return {
      ok: false,
      channel: "whatsapp",
      to,
      body: input.body,
      provider: "deep_link",
      deepLink,
      error:
        "Twilio WhatsApp not configured (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_WHATSAPP_FROM). Deep link prepared for agent.",
      sentAt,
    };
  }

  try {
    const auth = Buffer.from(`${sid}:${token}`).toString("base64");
    const body = new URLSearchParams({
      From: from.startsWith("whatsapp:") ? from : `whatsapp:${from}`,
      To: `whatsapp:${e164(to)}`,
      Body: input.body,
    });
    const res = await fetch(
      `https://api.twilio.com/2010-04-01/Accounts/${sid}/Messages.json`,
      {
        method: "POST",
        headers: {
          Authorization: `Basic ${auth}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body,
      },
    );
    const data = (await res.json()) as { sid?: string; message?: string };
    if (!res.ok) {
      return {
        ok: false,
        channel: "whatsapp",
        to,
        body: input.body,
        provider: "twilio",
        deepLink,
        error: data.message || `Twilio HTTP ${res.status}`,
        sentAt,
      };
    }
    return {
      ok: true,
      channel: "whatsapp",
      to,
      body: input.body,
      provider: "twilio",
      messageSid: data.sid,
      deepLink,
      sentAt,
    };
  } catch (e) {
    return {
      ok: false,
      channel: "whatsapp",
      to,
      body: input.body,
      provider: "twilio",
      deepLink,
      error: e instanceof Error ? e.message : "WhatsApp send failed",
      sentAt,
    };
  }
}
