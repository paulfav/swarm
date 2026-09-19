import { writeFileSync, readFileSync, existsSync } from "fs";

const ENV_PATH = new URL("../.env.local", import.meta.url);

async function main() {
  if (existsSync(ENV_PATH)) {
    const cur = readFileSync(ENV_PATH, "utf8");
    if (/CHINA_ACCESS_MAILTM_ADDRESS=/.test(cur) && /CHINA_ACCESS_MAILTM_PASSWORD=/.test(cur)) {
      const addr = cur.match(/CHINA_ACCESS_MAILTM_ADDRESS=(.*)/)?.[1]?.trim();
      console.log(`[inbox] using existing mail.tm inbox: ${addr}`);
      return;
    }
  }

  const domains = await fetch("https://api.mail.tm/domains").then((r) => r.json());
  const domain = domains["hydra:member"]?.[0]?.domain;
  if (!domain) throw new Error("No mail.tm domain");
  const user = `chinaaccess${Date.now().toString().slice(-6)}`;
  const address = `${user}@${domain}`;
  const password = `Ca${Math.random().toString(16).slice(2)}!aA1`;
  const created = await fetch("https://api.mail.tm/accounts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ address, password }),
  });
  if (!created.ok) {
    throw new Error(`mail.tm create failed: ${await created.text()}`);
  }
  const secret = Math.random().toString(16).slice(2) + Math.random().toString(16).slice(2);
  const body = [
    `CHINA_ACCESS_AGENT_EMAIL=${address}`,
    `CHINA_ACCESS_AGENT_NAME="China Access Agent"`,
    `CHINA_ACCESS_AGENT_COMPANY="China Access"`,
    `CHINA_ACCESS_AGENT_MOBILE=5550100123`,
    `CHINA_ACCESS_MAILTM_ADDRESS=${address}`,
    `CHINA_ACCESS_MAILTM_PASSWORD=${password}`,
    `CHINA_ACCESS_WEBHOOK_SECRET=${secret}`,
    "",
  ].join("\n");
  writeFileSync(ENV_PATH, body);
  console.log(`[inbox] provisioned mail.tm inbox: ${address}`);
}

main().catch((e) => {
  console.error("[inbox]", e);
  process.exit(0); // don't block dev
});
