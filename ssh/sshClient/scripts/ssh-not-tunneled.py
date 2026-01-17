#!/usr/bin/env python3


import subprocess
import time
import random

print("[PLAIN] Traffic")

commands = [
    "whoami", "hostname", "uptime", "ls -l /",
    "ps aux", "cat /etc/passwd", "dd if=/dev/zero bs=512K count=2"
]
try:
    while True:
        for session in range(1, 101):
            print(f"\n[PLAIN] ➤ Session {session}")
            for _ in range(random.randint(3, 6)):
                cmd = random.choice(commands)

                subprocess.run([
                    "ssh",
                    "-i", "/root/.ssh/id_rsa", #hier ist der private key des clients
                    "-p", "2222",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "ssh-user@192.168.100.100",
                    cmd
                ],timeout=10)
            time.sleep(random.uniform(0.5, 1.5))

except KeyboardInterrupt:
    print("Keyboardinterrupt")
