export interface GreenApiConfig {
  idInstance: string;
  apiTokenInstance: string;
  apiUrl: string;
}

export function getGreenApiConfig(): GreenApiConfig | null {
  const idInstance = process.env.GREEN_API_ID_INSTANCE;
  const apiTokenInstance = process.env.GREEN_API_TOKEN_INSTANCE;
  const apiUrl = (
    process.env.GREEN_API_API_URL || "https://api.green-api.com"
  ).replace(/\/$/, "");
  if (!idInstance || !apiTokenInstance) return null;
  return { idInstance, apiTokenInstance, apiUrl };
}

function instanceUrl(cfg: GreenApiConfig, method: string, extra = ""): string {
  return `${cfg.apiUrl}/waInstance${cfg.idInstance}/${method}/${cfg.apiTokenInstance}${extra}`;
}

export async function greenApiGetState(): Promise<{
  configured: boolean;
  stateInstance?: string;
  error?: string;
}> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { configured: false };
  try {
    const res = await fetch(instanceUrl(cfg, "getStateInstance"), {
      cache: "no-store",
    });
    const data = (await res.json()) as {
      stateInstance?: string;
      message?: string;
    };
    if (!res.ok) {
      return {
        configured: true,
        error: data.message || `HTTP ${res.status}`,
      };
    }
    return { configured: true, stateInstance: data.stateInstance };
  } catch (e) {
    return {
      configured: true,
      error: e instanceof Error ? e.message : "state check failed",
    };
  }
}

export async function greenApiGetQr(): Promise<{
  ok: boolean;
  type?: string;
  /** base64 PNG when type=qrCode */
  message?: string;
  error?: string;
}> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { ok: false, error: "Green-API not configured" };
  try {
    const res = await fetch(instanceUrl(cfg, "qr"), { cache: "no-store" });
    const data = (await res.json()) as {
      type?: string;
      message?: string;
    };
    if (!res.ok) {
      return { ok: false, error: data.message || `HTTP ${res.status}` };
    }
    return { ok: true, type: data.type, message: data.message };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "QR fetch failed",
    };
  }
}

export async function greenApiSendMessage(input: {
  phoneDigits: string;
  message: string;
}): Promise<{ ok: boolean; idMessage?: string; error?: string }> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { ok: false, error: "Green-API not configured" };
  const chatId = `${input.phoneDigits.replace(/[^\d]/g, "")}@c.us`;
  try {
    const res = await fetch(instanceUrl(cfg, "sendMessage"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chatId, message: input.message }),
    });
    const data = (await res.json()) as {
      idMessage?: string;
      message?: string;
    };
    if (!res.ok) {
      return { ok: false, error: data.message || `HTTP ${res.status}` };
    }
    return { ok: true, idMessage: data.idMessage };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "send failed",
    };
  }
}

export interface GreenApiInbound {
  receiptId: number;
  fromDigits: string;
  text: string;
  idMessage?: string;
  receivedAt: string;
}

type NotificationBody = {
  typeWebhook?: string;
  timestamp?: number;
  idMessage?: string;
  senderData?: { sender?: string; chatId?: string };
  messageData?: {
    typeMessage?: string;
    textMessageData?: { textMessage?: string };
    extendedTextMessageData?: { text?: string };
  };
};

/**
 * Drain Green-API notification queue (incoming text only).
 */
export async function greenApiPollIncoming(limit = 20): Promise<{
  configured: boolean;
  messages: GreenApiInbound[];
  error?: string;
}> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { configured: false, messages: [] };

  const messages: GreenApiInbound[] = [];
  try {
    for (let i = 0; i < limit; i++) {
      const res = await fetch(instanceUrl(cfg, "receiveNotification"), {
        cache: "no-store",
      });
      if (!res.ok) {
        const err = await res.text();
        return {
          configured: true,
          messages,
          error: err || `HTTP ${res.status}`,
        };
      }
      const data = (await res.json()) as {
        receiptId?: number;
        body?: NotificationBody;
      } | null;
      if (!data || data.receiptId == null) break;

      const body = data.body || {};
      if (body.typeWebhook === "incomingMessageReceived") {
        const text =
          body.messageData?.textMessageData?.textMessage ||
          body.messageData?.extendedTextMessageData?.text ||
          "";
        const sender =
          body.senderData?.sender || body.senderData?.chatId || "";
        const fromDigits = sender.replace(/[^\d]/g, "");
        if (text.trim() && fromDigits) {
          messages.push({
            receiptId: data.receiptId,
            fromDigits,
            text,
            idMessage: body.idMessage,
            receivedAt: body.timestamp
              ? new Date(body.timestamp * 1000).toISOString()
              : new Date().toISOString(),
          });
        }
      }

      await fetch(
        instanceUrl(cfg, "deleteNotification", `/${data.receiptId}`),
        { method: "DELETE" },
      ).catch(() => undefined);
    }
    return { configured: true, messages };
  } catch (e) {
    return {
      configured: true,
      messages,
      error: e instanceof Error ? e.message : "poll failed",
    };
  }
}
