#!/usr/bin/env node
/**
 * Self-provision a free Green-API WhatsApp developer instance.
 * Reuses existing .env.local credentials when present.
 * Instance still needs one WhatsApp Linked-Devices QR scan to authorize.
 */
import { writeFileSync, readFileSync, existsSync } from "fs";

const ENV_PATH = new URL("../.env.local", import.meta.url);
const CONSOLE = "https://console.green-api.com/api/v1/";
const CONSOLE_AUTH = "gac.cb546085ecfd42f1a135480c82c9279e";

function readEnv() {
  if (!existsSync(ENV_PATH)) return {};
  const out = {};
  for (const line of readFileSync(ENV_PATH, "utf8").split("\n")) {
    const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
    if (m) out[m[1]] = m[2].replace(/^"|"$/g, "");
  }
  return out;
}

function upsertEnv(vars) {
  const cur = existsSync(ENV_PATH) ? readFileSync(ENV_PATH, "utf8") : "";
  let next = cur;
  for (const [k, v] of Object.entries(vars)) {
    const line = `${k}=${v}`;
    if (new RegExp(`^${k}=`, "m").test(next)) {
      next = next.replace(new RegExp(`^${k}=.*$`, "m"), line);
    } else {
      next = `${next.replace(/\s*$/, "")}\n${line}\n`;
    }
  }
  writeFileSync(ENV_PATH, next.endsWith("\n") ? next : `${next}\n`);
}

async function mailtmToken(address, password) {
  const res = await fetch("https://api.mail.tm/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ address, password }),
  });
  const json = await res.json();
  if (!json.token) throw new Error(json.message || "mail.tm token failed");
  return json.token;
}

async function waitConfirmCode(address, password) {
  const token = await mailtmToken(address, password);
  for (let i = 0; i < 30; i++) {
    const list = await fetch("https://api.mail.tm/messages", {
      headers: { Authorization: `Bearer ${token}` },
    }).then((r) => r.json());
    for (const m of list["hydra:member"] || []) {
      const detail = await fetch(`https://api.mail.tm/messages/${m.id}`, {
        headers: { Authorization: `Bearer ${token}` },
      }).then((r) => r.json());
      const blob = `${detail.subject || ""}\n${detail.text || ""}\n${
        Array.isArray(detail.html) ? detail.html.join("\n") : detail.html || ""
      }`;
      if (!/confirm|green|code/i.test(blob) && i < 2) continue;
      const codes = blob.match(/\b(\d{6})\b/g);
      if (codes?.length) return codes[0];
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
  throw new Error("No Green-API confirmation code in inbox");
}

async function consoleApi(method, body, user = {}) {
  const res = await fetch(CONSOLE, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${CONSOLE_AUTH}`,
      "x-ga-method": method,
      "x-ga-user-id": user.idUser || "null",
      "x-ga-user-token": user.apiTokenUser || "null",
      "x-ga-project-id": user.projectId || "null",
      Origin: "https://console.green-api.com",
      Referer: "https://console.green-api.com/",
    },
    body: JSON.stringify(body ?? {}),
  });
  const json = await res.json();
  if (!json.result) {
    throw new Error(
      json.error?.description || json.error?.code || `console ${method} failed`,
    );
  }
  return json.data;
}

async function instanceState(apiUrl, idInstance, token) {
  const url = `${apiUrl.replace(/\/$/, "")}/waInstance${idInstance}/getStateInstance/${token}`;
  const res = await fetch(url);
  if (!res.ok) return { error: await res.text() };
  return res.json();
}

async function main() {
  const env = readEnv();
  if (
    env.GREEN_API_ID_INSTANCE &&
    env.GREEN_API_TOKEN_INSTANCE &&
    env.GREEN_API_API_URL
  ) {
    const state = await instanceState(
      env.GREEN_API_API_URL,
      env.GREEN_API_ID_INSTANCE,
      env.GREEN_API_TOKEN_INSTANCE,
    );
    console.log(
      `[whatsapp] green-api instance ${env.GREEN_API_ID_INSTANCE} state=${state.stateInstance || state.error || "unknown"}`,
    );
    return;
  }

  const email = env.CHINA_ACCESS_MAILTM_ADDRESS || env.CHINA_ACCESS_AGENT_EMAIL;
  const password = env.CHINA_ACCESS_MAILTM_PASSWORD;
  if (!email || !password) {
    console.log("[whatsapp] skip — provision mail.tm inbox first");
    return;
  }

  let user = {
    idUser: env.GREEN_API_USER_ID,
    apiTokenUser: env.GREEN_API_USER_TOKEN,
    projectId: env.GREEN_API_PROJECT_ID,
  };

  if (!user.idUser || !user.apiTokenUser || !user.projectId) {
    console.log(`[whatsapp] registering Green-API account for ${email}`);
    try {
      await consoleApi("registerUser", {
        login: email,
        country: "US",
        language: "EN",
      });
      const code = await waitConfirmCode(email, password);
      const verified = await consoleApi("verifyUser", {
        login: email,
        code: Number(code),
      });
      user = {
        idUser: verified.idUser,
        apiTokenUser: verified.apiTokenUser,
        projectId: verified.project.projectId,
      };
    } catch (e) {
      // Already registered — try login with email code flow is not password-based.
      // Fall back: require existing user tokens.
      console.log(`[whatsapp] register note: ${e.message}`);
      if (!user.idUser) throw e;
    }
  }

  const order = await consoleApi(
    "user.instance.createByOrder",
    {
      tariff: "DEVELOPER",
      period: "infinitely",
      payment: "tinkoff_rub",
      quantity: 1,
      idCompany: "",
    },
    user,
  );
  console.log(`[whatsapp] order ${order.orderNumber} confirmed=${order.isConfirmed}`);

  let instance;
  for (let i = 0; i < 15; i++) {
    const list = await consoleApi("user.instances.list", {}, user);
    instance = (list || []).find((x) => !x.deleted && x.apiTokenInstance);
    if (instance && instance.tariff === "DEVELOPER" && !instance.isExpired) break;
    await new Promise((r) => setTimeout(r, 2000));
  }
  if (!instance) throw new Error("No Green-API instance after order");

  upsertEnv({
    GREEN_API_ID_INSTANCE: String(instance.idInstance),
    GREEN_API_TOKEN_INSTANCE: instance.apiTokenInstance,
    GREEN_API_API_URL: String(instance.apiUrl || "").replace(/\/$/, ""),
    GREEN_API_USER_ID: user.idUser,
    GREEN_API_USER_TOKEN: user.apiTokenUser,
    GREEN_API_PROJECT_ID: user.projectId,
    GREEN_API_CONSOLE_AUTH: CONSOLE_AUTH,
  });

  const state = await instanceState(
    instance.apiUrl,
    instance.idInstance,
    instance.apiTokenInstance,
  );
  console.log(
    `[whatsapp] provisioned instance ${instance.idInstance} state=${state.stateInstance || state.error || "pending"} — scan QR via GET /api/inbox/status`,
  );
}

main().catch((e) => {
  console.error("[whatsapp]", e.message || e);
  process.exit(0);
});
