#!/usr/bin/env python3

import asyncio
import argparse
import contextlib
import itertools
import os
import random
import socket
import sys
import time
from contextlib import closing


#Config
TARGET_HOST = "192.168.100.100"
TARGET_PORT = 2222
TARGET_USER = "ssh-user"
SSH_KEY    = "/root/.ssh/id_rsa"

COMMANDS = [
    "whoami", "hostname", "uptime", "ls -l /",
    "ps aux | head -n 20",
    "cat /etc/passwd | head -n 20",
    "dd if=/dev/zero bs=256K count=2 status=none | wc -c"
]

HTTP_URLS = [
    "https://example.com",
    "https://www.gnu.org",
    "https://www.python.org",
]

SCP_FILE = "scp_file.txt"


#Utilities
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def print_event(session_id, event, **kw):
    ts = now_str()
    items = " ".join(f"{k}={v}" for k, v in kw.items())
    if session_id is None:
        print(f"[{ts}] {event} {items}", flush=True)
    else:
        print(f"[{ts}] session={session_id} event={event} {items}", flush=True)

def pick_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

async def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            await asyncio.sleep(0.2)
    return False

async def run_proc(*args, env=None, timeout=None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        return 124, "", "timeout"
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")

def ensure_scp_file():
    if not os.path.exists(SCP_FILE):
        with open(SCP_FILE, "w") as f:
            f.write("Hello from traffic generator.\n")


#Preflight: block until SSH works once
async def preflight_check(max_wait: float) -> bool:
    env = dict(os.environ)
    env["SSH_AUTH_SOCK"] = ""
    start = time.time()
    attempt = 0
    while time.time() - start < max_wait:
        attempt += 1
        print_event(None, "preflight_try", attempt=attempt)
        rc, _, _ = await run_proc(
            "ssh",
            "-p", str(TARGET_PORT),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=3",
            "-o", "ConnectionAttempts=1",
            "-q",
            "-i", SSH_KEY,
            f"{TARGET_USER}@{TARGET_HOST}",
            "true",
            env=env,
            timeout=6,
        )
        if rc == 0:
            print_event(None, "preflight_ok", attempts=attempt)
            return True
        await asyncio.sleep(1.0)
    print_event(None, "preflight_timeout", waited_seconds=int(time.time() - start))
    return False


#Actions (TUNNELED ONLY)
async def run_scp(session_id, env, ssh_port):
    args = [
        "scp",
        "-P", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "IdentitiesOnly=yes",
        "-o", "LogLevel=ERROR",
        "-i", SSH_KEY,
        SCP_FILE,
        f"{TARGET_USER}@localhost:{SCP_FILE}"
    ]
    rc, out, err = await run_proc(*args, env=env, timeout=30)
    size = os.path.getsize(SCP_FILE) if os.path.exists(SCP_FILE) else 0
    info = (err.strip() or out.strip())[:200]
    return rc, size, info

async def run_http(session_id, socks_port):
    url = random.choice(HTTP_URLS)
    proxy = f"socks5h://localhost:{socks_port}"
    args = [
        "curl", "-x", proxy,
        "--retry", "2", "--retry-delay", "1",
        "--connect-timeout", "5", "--max-time", "25",
        "-sS", "-o", "/dev/null", "-w", "%{http_code}", url
    ]
    rc, out, err = await run_proc(*args, timeout=30)
    code = out.strip() if out else ""
    info = f"http={url} code={code}" if rc == 0 else (err.strip()[:200] or "curl_failed")
    return rc, len(out or ""), info

async def run_remote_cmd(session_id, env, ssh_port):
    cmd = random.choice(COMMANDS)
    args = [
        "ssh",
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "IdentitiesOnly=yes",
        "-o", "LogLevel=ERROR",
        "-i", SSH_KEY,
        f"{TARGET_USER}@localhost",
        cmd
    ]
    rc, out, err = await run_proc(*args, env=env, timeout=30)
    txt = (out if rc == 0 else err).strip()
    info = f"cmd={cmd!r} outlen={len(txt)}"
    return rc, len(out or ""), info


#Tunnel lifecycle
async def start_tunnel(local_ssh_port: int, socks_port: int, env) -> asyncio.subprocess.Process:
    args = [
        "ssh",
        "-i", SSH_KEY,
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2",
        "-o", "ConnectTimeout=5",
        "-p", str(TARGET_PORT),
        "-N",
        "-L", f"{local_ssh_port}:127.0.0.1:{TARGET_PORT}",
        "-D", f"{socks_port}",
        f"{TARGET_USER}@{TARGET_HOST}",
    ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE, env=env
    )
    return proc

async def stop_proc(proc: asyncio.subprocess.Process):
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
    except ProcessLookupError:
        pass


#One tunneled session
async def run_session(session_id: int, min_actions=3, max_actions=6):
    env = dict(os.environ)
    env["SSH_AUTH_SOCK"] = ""  # avoid agent side traffic
    ensure_scp_file()

    local_ssh_port = pick_free_port()
    socks_port = pick_free_port()
    print_event(session_id, "tunnel_starting", local_ssh_port=local_ssh_port, socks_port=socks_port)

    tunnel_proc = await start_tunnel(local_ssh_port, socks_port, env)
    ok_ssh = await wait_for_port("127.0.0.1", local_ssh_port, timeout=12)
    ok_socks = await wait_for_port("127.0.0.1", socks_port, timeout=12)

    if not (ok_ssh and ok_socks):
        err = ""
        if tunnel_proc and tunnel_proc.stderr:
            try:
                err = (await asyncio.wait_for(tunnel_proc.stderr.read(), timeout=0.5)).decode(errors="replace").strip()
            except Exception:
                pass
        await stop_proc(tunnel_proc)
        print_event(session_id, "tunnel_failed", ok_ssh=ok_ssh, ok_socks=ok_socks, stderr=(err[:200] if err else "-"))
        print_event(session_id, "session_end", reason="tunnel_not_ready")
        return

    print_event(session_id, "tunnel_ready", local_ssh_port=local_ssh_port, socks_port=socks_port)
    print_event(session_id, "session_start")

    try:
        actions = random.randint(min_actions, max_actions)
        choices = ["scp", "http", "cmd"]
        for idx in range(1, actions + 1):
            try:
                choice = random.choice(choices)
                if choice == "scp":
                    rc, b, info = await run_scp(session_id, env, ssh_port=local_ssh_port)
                elif choice == "http":
                    rc, b, info = await run_http(session_id, socks_port=socks_port)
                else:
                    rc, b, info = await run_remote_cmd(session_id, env, ssh_port=local_ssh_port)
            except Exception as e:
                rc, b, info = 70, 0, f"exception:{type(e).__name__}:{e}"
            print_event(session_id, "action", idx=idx, type=choice, rc=rc, bytes=b, info=info)
            await asyncio.sleep(random.uniform(0.2, 1.2))
    finally:
        await stop_proc(tunnel_proc)
        print_event(session_id, "tunnel_stopped")
        print_event(session_id, "session_end", reason="normal")


#Infinite orchestrator
async def worker_loop(worker_id: int, next_session_id, min_actions: int, max_actions: int, stop_event: asyncio.Event):
    print_event(None, "worker_started", worker=worker_id)
    while not stop_event.is_set():
        sid = next(next_session_id)
        await run_session(sid, min_actions=min_actions, max_actions=max_actions)
        # brief pause so workers don’t hammer immediately
        await asyncio.sleep(random.uniform(0.1, 0.5))
    print_event(None, "worker_stopped", worker=worker_id)

async def main():
    parser = argparse.ArgumentParser(description="Tunneled-only SSH traffic generator (runs until interrupted).")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent sessions (worker pool size).")
    parser.add_argument("--min-actions", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--preflight", action="store_true", default=True, help="Block until SSH is ready before starting.")
    parser.add_argument("--no-preflight", dest="preflight", action="store_false")
    parser.add_argument("--preflight-timeout", type=int, default=60, help="Seconds to wait for readiness.")
    args = parser.parse_args()

    print_event(None, "orchestrator_start", concurrency=args.concurrency)

    if args.preflight:
        ok = await preflight_check(args.preflight_timeout)
        if not ok:
            print_event(None, "abort", reason="preflight_failed")
            return

    stop_event = asyncio.Event()
    next_session_id = itertools.count(1)

    #Start workers
    workers = [
        asyncio.create_task(worker_loop(i+1, next_session_id, args.min_actions, args.max_actions, stop_event))
        for i in range(max(1, args.concurrency))
    ]

    print_event(None, "running", hint="Press Ctrl+C to stop")
    try:
        #Run forever until Ctrl+C
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print_event(None, "interrupt_received")
    finally:
        stop_event.set()
        for t in workers:
            t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*workers)
        print_event(None, "orchestrator_stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"[{now_str()}] KeyboardInterrupt – exiting.", file=sys.stderr)
