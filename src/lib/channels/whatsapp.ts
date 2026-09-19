import {
  getGreenApiConfig,
  greenApiGetState,
  greenApiSendMessage,
} from "./green-api";

export interface WhatsAppSendInput {
  to: string; // digits
  body: string;
}

export interface WhatsAppSendResult {
  ok: boolean;
  channel: "whatsapp";
  to: string;
  body: string;
  provider: "green-api" | "twilio" | "deep_link" | "none";
  messageSid?: string;
  deepLink?: string;
  error?: string;
  sentAt: string;
}

function e164(to: string): string {
  const d = to.replace(/[^\d]/g, "");
  return d.startsWith("+") ? d : `+${d}`;
}

async function sendViaTwilio(
  to: string,
  body: string,
  deepLink: string,
  sentAt: string,
): Promise<WhatsAppSendResult> {
  const sid = process.env.TWILIO_ACCOUNT_SID;
  const token = process.env.TWILIO_AUTH_TOKEN;
  const from = process.env.TWILIO_WHATSAPP_FROM;
  if (!sid || !token || !from) {
    return {
      ok: false,
      channel: "whatsapp",
      to,
      body,
      provider: "deep_link",
      deepLink,
      error:
        "No WhatsApp sender ready (Green-API unauthorized or Twilio missing). Deep link prepared.",
      sentAt,
    };
  }

  try {
    const auth = Buffer.from(`${sid}:${token}`).toString("base64");
    const form = new URLSearchParams({
      From: from.startsWith("whatsapp:") ? from : `whatsapp:${from}`,
      To: `whatsapp:${e164(to)}`,
      Body: body,
    });
    const res = await fetch(
      `https://api.twilio.com/2010-04-01/Accounts/${sid}/Messages.json`,
      {
        method: "POST",
        headers: {
          Authorization: `Basic ${auth}`,
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: form,
      },
    );
    const data = (await res.json()) as { sid?: string; message?: string };
    if (!res.ok) {
      return {
        ok: false,
        channel: "whatsapp",
        to,
        body,
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
      body,
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
      body,
      provider: "twilio",
      deepLink,
      error: e instanceof Error ? e.message : "WhatsApp send failed",
      sentAt,
    };
  }
}

/**
 * Send WhatsApp via self-provisioned Green-API (preferred), Twilio, or wa.me deep link.
 */
export async function sendWhatsApp(
  input: WhatsAppSendInput,
): Promise<WhatsAppSendResult> {
  const sentAt = new Date().toISOString();
  const to = input.to.replace(/[^\d]/g, "");
  const deepLink = `https://wa.me/${to}?text=${encodeURIComponent(input.body)}`;

  const green = getGreenApiConfig();
  if (green) {
    const state = await greenApiGetState();
    if (state.stateInstance === "authorized") {
      const sent = await greenApiSendMessage({
        phoneDigits: to,
        message: input.body,
      });
      if (sent.ok) {
        return {
          ok: true,
          channel: "whatsapp",
          to,
          body: input.body,
          provider: "green-api",
          messageSid: sent.idMessage,
          deepLink,
          sentAt,
        };
      }
      return {
        ok: false,
        channel: "whatsapp",
        to,
        body: input.body,
        provider: "green-api",
        deepLink,
        error: sent.error,
        sentAt,
      };
    }
    // Fall through to Twilio / deep link if not yet QR-linked
  }

  return sendViaTwilio(to, input.body, deepLink, sentAt);
}
