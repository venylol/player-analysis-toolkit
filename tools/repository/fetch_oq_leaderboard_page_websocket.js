#!/usr/bin/env node

const fs = require("node:fs");

const page = process.argv[2];
const output = process.argv[3];
if (page !== "24" || !output) {
  throw new Error("this recovery tool is restricted to leaderboard page 24 and requires an output path");
}

const base = "http://questgames.net:3002/socket.io/1";
const eventName = "25764636";
const timeoutMs = 30_000;

async function main() {
  const handshakeResponse = await fetch(`${base}/?t=${Date.now()}`);
  if (!handshakeResponse.ok) {
    throw new Error(`handshake HTTP ${handshakeResponse.status}`);
  }
  const handshake = await handshakeResponse.text();
  const sessionId = handshake.split(":", 1)[0].trim();
  if (!sessionId) {
    throw new Error(`empty session id: ${handshake.slice(0, 120)}`);
  }
  const socketUrl = `ws://questgames.net:3002/socket.io/1/websocket/${sessionId}`;
  const socket = new WebSocket(socketUrl);
  const frames = [];
  let sent = false;
  const timer = setTimeout(() => {
    socket.close();
    process.stderr.write("page 24 websocket recovery timed out\n");
    process.exitCode = 1;
  }, timeoutMs);

  socket.addEventListener("message", (message) => {
    const frame = String(message.data);
    frames.push(frame);
    if (!sent && frame.startsWith("1::")) {
      sent = true;
      const packet = {
        name: eventName,
        args: [{ gtype: "reversi", page }],
      };
      socket.send(`5:::${JSON.stringify(packet)}`);
      return;
    }
    if (frame.startsWith("5:::")) {
      const decoded = JSON.parse(frame.slice(4));
      if (decoded.name !== eventName || !decoded.args?.[0]) {
        throw new Error("unexpected leaderboard websocket event");
      }
      const record = {
        page: Number(page),
        fetched_at: new Date().toISOString(),
        transport: "websocket-single-failed-page-recovery",
        handshake,
        frames,
        data: decoded.args[0],
      };
      fs.writeFileSync(output, `${JSON.stringify(record)}\n`, { encoding: "utf8", flag: "wx" });
      clearTimeout(timer);
      socket.close();
      process.stdout.write(`${JSON.stringify({ ok: true, page: 24, users: record.data.users?.length ?? 0 })}\n`);
    }
  });
  socket.addEventListener("error", () => {
    clearTimeout(timer);
    process.stderr.write("page 24 websocket recovery failed\n");
    process.exitCode = 1;
  });
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
