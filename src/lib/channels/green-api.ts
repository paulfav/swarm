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
    const raw = await res.text();
    if (!raw.trim()) {
      return {
        configured: true,
        error: `Empty state response (HTTP ${res.status})`,
      };
    }
    let data: { stateInstance?: string; message?: string };
    try {
      data = JSON.parse(raw) as { stateInstance?: string; message?: string };
    } catch {
      return {
        configured: true,
        error: `Bad state JSON (HTTP ${res.status})`,
      };
    }
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
    if (data.type === "error" || data.type === "timeout") {
      return {
        ok: false,
        type: data.type,
        message: data.message,
        error: data.message || data.type,
      };
    }
    if (data.type === "alreadyLogged") {
      return { ok: true, type: data.type, message: data.message };
    }
    return { ok: true, type: data.type, message: data.message };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "QR fetch failed",
    };
  }
}

export async function greenApiReboot(): Promise<{
  ok: boolean;
  error?: string;
}> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { ok: false, error: "Green-API not configured" };
  try {
    const res = await fetch(instanceUrl(cfg, "reboot"), { cache: "no-store" });
    const data = (await res.json()) as {
      isReboot?: boolean;
      message?: string;
    };
    if (!res.ok || !data.isReboot) {
      return { ok: false, error: data.message || `HTTP ${res.status}` };
    }
    return { ok: true };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "reboot failed",
    };
  }
}

/**
 * Alternative to QR: WhatsApp → Linked devices → Link with phone number instead.
 * Returns a short code the user types into WhatsApp (valid ~2.5 min).
 */
export async function greenApiGetAuthorizationCode(
  phoneDigits: string,
): Promise<{ ok: boolean; code?: string; error?: string }> {
  const cfg = getGreenApiConfig();
  if (!cfg) return { ok: false, error: "Green-API not configured" };
  const phoneNumber = Number(phoneDigits.replace(/[^\d]/g, ""));
  if (!Number.isFinite(phoneNumber) || String(phoneNumber).length < 8) {
    return { ok: false, error: "Enter a full international phone number" };
  }
  try {
    const res = await fetch(instanceUrl(cfg, "getAuthorizationCode"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ phoneNumber }),
    });
    const data = (await res.json()) as {
      status?: boolean;
      code?: string;
      message?: string;
    };
    if (!res.ok) {
      return { ok: false, error: data.message || `HTTP ${res.status}` };
    }
    if (!data.status || !data.code) {
      return {
        ok: false,
        error:
          data.message ||
          "Could not get link code — reboot the instance and try again",
      };
    }
    return { ok: true, code: data.code };
  } catch (e) {
    return {
      ok: false,
      error: e instanceof Error ? e.message : "auth code failed",
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
