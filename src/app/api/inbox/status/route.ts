import { NextResponse } from "next/server";
import { pollAgentInbox } from "@/lib/channels/email-inbox";

export async function GET() {
  const address =
    process.env.CHINA_ACCESS_AGENT_EMAIL ||
    process.env.CHINA_ACCESS_MAILTM_ADDRESS ||
    null;

  const inbox = await pollAgentInbox({ limit: 5 });

  return NextResponse.json({
    agentEmail: address,
    inboxConfigured: inbox.configured,
    provider: inbox.provider || null,
    messageCount: inbox.emails.length,
    error: inbox.error || null,
    whatsapp: {
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
    },
  });
}
