import { NextResponse } from "next/server";
import { pollAgentInbox } from "@/lib/channels/email-inbox";
import {
  getGreenApiConfig,
  greenApiGetState,
} from "@/lib/channels/green-api";

export const dynamic = "force-dynamic";

/**
 * Channel status without fetching a QR code.
 * QR must only be requested from /api/whatsapp/link so it stays fresh
 * and isn't invalidated by concurrent polls.
 */
export async function GET() {
  const address =
    process.env.CHINA_ACCESS_AGENT_EMAIL ||
    process.env.CHINA_ACCESS_MAILTM_ADDRESS ||
    null;

  const inbox = await pollAgentInbox({ limit: 5 });
  const greenCfg = getGreenApiConfig();
  const waState = greenCfg ? await greenApiGetState() : { configured: false };

  return NextResponse.json({
    agentEmail: address,
    inboxConfigured: inbox.configured,
    provider: inbox.provider || null,
    messageCount: inbox.emails.length,
    error: inbox.error || null,
    whatsapp: {
      greenApiConfigured: Boolean(greenCfg),
      idInstance: greenCfg?.idInstance || null,
      state: waState.stateInstance || null,
      authorized: waState.stateInstance === "authorized",
      linkPath: "/api/whatsapp/link",
      error: waState.error || null,
      twilioConfigured: Boolean(
        process.env.TWILIO_ACCOUNT_SID &&
          process.env.TWILIO_AUTH_TOKEN &&
          process.env.TWILIO_WHATSAPP_FROM,
      ),
    },
    wechat: {
      wecomConfigured: Boolean(
        process.env.WECOM_CORP_ID &&
          process.env.WECOM_SECRET &&
          process.env.WECOM_AGENT_ID,
      ),
      note: "WeChat personal accounts have no public send API; WeCom needs a verified Chinese company corp id.",
    },
  });
}
