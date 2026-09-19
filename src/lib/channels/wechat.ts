export interface WeChatSendInput {
  wechatId: string;
  body: string;
}

export interface WeChatSendResult {
  ok: boolean;
  channel: "wechat";
  wechatId: string;
  body: string;
  provider: "wecom" | "queued";
  error?: string;
  sentAt: string;
}

/**
 * WeChat personal accounts aren't API-reachable. Enterprise WeChat (WeCom)
 * can send if credentials exist; otherwise we queue for bilingual ops.
 */
export async function sendWeChat(
  input: WeChatSendInput,
): Promise<WeChatSendResult> {
  const sentAt = new Date().toISOString();
  const corpId = process.env.WECOM_CORP_ID;
  const secret = process.env.WECOM_SECRET;
  const agentId = process.env.WECOM_AGENT_ID;

  if (!corpId || !secret || !agentId) {
    return {
      ok: false,
      channel: "wechat",
      wechatId: input.wechatId,
      body: input.body,
      provider: "queued",
      error:
        "WeCom not configured. WeChat ID captured — ops/agent must message manually; result will be pasted into the deal thread.",
      sentAt,
    };
  }

  try {
    const tokenRes = await fetch(
      `https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=${corpId}&corpsecret=${secret}`,
    );
    const tokenJson = (await tokenRes.json()) as {
      access_token?: string;
      errmsg?: string;
    };
    if (!tokenJson.access_token) {
      throw new Error(tokenJson.errmsg || "WeCom token failed");
    }

    // WeCom external-contact send is org-specific; store as queued message via app chat fallback
    const sendRes = await fetch(
      `https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=${tokenJson.access_token}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          touser: input.wechatId,
          msgtype: "text",
          agentid: Number(agentId),
          text: { content: input.body },
        }),
      },
    );
    const sendJson = (await sendRes.json()) as {
      errcode?: number;
      errmsg?: string;
    };
    if (sendJson.errcode && sendJson.errcode !== 0) {
      return {
        ok: false,
        channel: "wechat",
        wechatId: input.wechatId,
        body: input.body,
        provider: "wecom",
        error: sendJson.errmsg || `WeCom error ${sendJson.errcode}`,
        sentAt,
      };
    }
    return {
      ok: true,
      channel: "wechat",
      wechatId: input.wechatId,
      body: input.body,
      provider: "wecom",
      sentAt,
    };
  } catch (e) {
    return {
      ok: false,
      channel: "wechat",
      wechatId: input.wechatId,
      body: input.body,
      provider: "wecom",
      error: e instanceof Error ? e.message : "WeChat send failed",
      sentAt,
    };
  }
}
