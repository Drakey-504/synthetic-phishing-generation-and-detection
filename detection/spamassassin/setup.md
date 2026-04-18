# SpamAssassin Setup (macOS via Docker)

Homebrew no longer ships the `spamassassin` formula. We run SpamAssassin in
a Docker container and talk to it over TCP from Python — no local Perl
install needed.

## 1. Prerequisites

- Docker Desktop installed and running (whale icon in menu bar, not animating).

## 2. Start the container

```bash
docker run -d \
  --name spamd \
  -p 783:783 \
  --restart unless-stopped \
  instantlinux/spamassassin
```

This pulls the image on first run (~200 MB), then starts `spamd` inside
the container on port 783, exposed to your Mac's localhost.

## 3. Verify it's healthy

```bash
docker ps
```

You should see `spamd` with status `Up X seconds` and
`0.0.0.0:783->783/tcp` in the ports column.

```bash
docker logs spamd | tail -20
```

Look for `spamd: server started on IO::Socket::IP [0.0.0.0]:783`. Startup
takes ~20 seconds — the daemon loads ~100 MB of rules into memory.

You may see one harmless warning:
```
register: Error reading socket
Use of uninitialized value ... Razor2/Client/Config.pm
```
Razor2 is an optional network-based reputation service that can't reach
the outside network from inside the container. The local rule set (which
does 99% of the work) is unaffected. Ignore it.

## 4. Smoke test

The `eval.py` script talks to spamd directly over TCP, so no `spamc`
binary is needed. Test with Python:

```bash
python3 -c "
import socket
msg = b'Subject: test\r\nFrom: x@y.com\r\n\r\nhello world'
req = (b'REPORT SPAMC/1.5\r\n'
       b'Content-length: ' + str(len(msg)).encode() + b'\r\n\r\n' + msg)
s = socket.create_connection(('127.0.0.1', 783))
s.sendall(req); s.shutdown(socket.SHUT_WR)
print(s.recv(4096).decode(errors='replace'))
"
```

You should see a response that starts with `SPAMD/1.1 0 EX_OK` followed
by `Spam: False ; <score> / 5.0`. If you see that, you're ready to run
`eval.py`.

## 5. Stop / restart

```bash
docker stop spamd         # stops the container (state preserved)
docker start spamd        # starts it again
docker rm -f spamd        # removes it completely
```

Because we used `--restart unless-stopped`, the container will auto-start
when Docker Desktop launches. You don't need to restart it between
evaluation runs.

## Troubleshooting

**`docker ps` shows the container `Exited`** — check `docker logs spamd`
for the reason. Most common: port 783 already in use. Check with
`lsof -i :783` and either kill the conflicting process or change the port
via `-p 7830:783` and pass `--port 7830` to `eval.py`.

**`Connection refused` on port 783** — container isn't running. Start it
with `docker start spamd`.

**Rules out of date (scores look suspiciously low)** — the image ships
with a snapshot of the rules. To update:
```bash
docker exec spamd sa-update --nogpg
docker restart spamd
```
