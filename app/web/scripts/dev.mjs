import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import http from "node:http";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(webDir, "..", "..");

const apiHost = process.env.MODEL_TESTER_API_HOST || "127.0.0.1";
const apiPort = Number(process.env.MODEL_TESTER_API_PORT || 8000);
const viteHost = process.env.MODEL_TESTER_VITE_HOST || "0.0.0.0";
const vitePort = process.env.MODEL_TESTER_VITE_PORT;

const children = new Set();

function requestHealth() {
  return new Promise((resolve) => {
    const request = http.get(
      {
        host: apiHost,
        port: apiPort,
        path: "/health",
        timeout: 1000,
      },
      (response) => {
        response.resume();
        resolve(response.statusCode === 200);
      },
    );

    request.on("timeout", () => {
      request.destroy();
      resolve(false);
    });
    request.on("error", () => resolve(false));
  });
}

function canConnect() {
  return new Promise((resolve) => {
    const socket = net.createConnection({ host: apiHost, port: apiPort });
    socket.setTimeout(1000);
    socket.on("connect", () => {
      socket.destroy();
      resolve(true);
    });
    socket.on("timeout", () => {
      socket.destroy();
      resolve(false);
    });
    socket.on("error", () => resolve(false));
  });
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function findPython() {
  const candidates = [
    process.env.MODEL_TESTER_PYTHON,
    path.join(repoRoot, ".venv", "Scripts", "python.exe"),
    path.join(repoRoot, ".venv", "bin", "python"),
    "python",
  ].filter(Boolean);

  return candidates.find((candidate) => candidate === "python" || existsSync(candidate));
}

function startProcess(command, args, options) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: process.env,
    stdio: "inherit",
    windowsHide: false,
  });

  children.add(child);
  child.on("exit", () => children.delete(child));
  return child;
}

async function waitForBackend(child) {
  let exited = false;
  child.on("exit", () => {
    exited = true;
  });

  const deadline = Date.now() + 45000;
  while (Date.now() < deadline) {
    if (await requestHealth()) return true;
    if (exited) return false;
    await delay(500);
  }
  return false;
}

function stopChildren() {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

process.on("SIGINT", () => {
  stopChildren();
  process.exit(130);
});

process.on("SIGTERM", () => {
  stopChildren();
  process.exit(143);
});

if (await requestHealth()) {
  console.log(`Using existing FastAPI backend at http://${apiHost}:${apiPort}`);
} else if (await canConnect()) {
  console.error(
    `Port ${apiPort} is already in use, but http://${apiHost}:${apiPort}/health did not return OK.`,
  );
  console.error("Stop that process or set MODEL_TESTER_API_PORT before running npm run dev.");
  process.exit(1);
} else {
  const python = findPython();
  console.log(`Starting FastAPI backend at http://${apiHost}:${apiPort}`);
  const backend = startProcess(
    python,
    ["-m", "uvicorn", "app.api:app", "--reload", "--host", apiHost, "--port", String(apiPort)],
    { cwd: repoRoot },
  );

  if (!(await waitForBackend(backend))) {
    console.error("FastAPI did not become ready. Check the backend output above.");
    console.error("Common fix: install Python requirements, then run npm run dev again.");
    stopChildren();
    process.exit(1);
  }
}

const viteBin = path.join(webDir, "node_modules", "vite", "bin", "vite.js");
if (!existsSync(viteBin)) {
  console.error("Vite is not installed yet. Run npm install in app/web, then run npm run dev.");
  stopChildren();
  process.exit(1);
}

const viteArgs = [viteBin, "--host", viteHost];
if (vitePort) viteArgs.push("--port", String(vitePort));

const vite = startProcess(process.execPath, viteArgs, { cwd: webDir });
vite.on("exit", (code) => {
  stopChildren();
  process.exit(code ?? 0);
});
