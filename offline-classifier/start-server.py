#!/usr/bin/env python3
"""Tiny local server for the offline classifier (Python 3 standard library only).

  python3 start-server.py            -> http://localhost:8000      (PC; camera works on localhost)
  python3 start-server.py 9000       -> custom port
  python3 start-server.py --https    -> https://<LAN-IP>:8443      (phone on same Wi-Fi; self-signed cert,
                                        needs `openssl` installed; accept the browser warning)
"""
import http.server, socketserver, sys, os, socket, ssl, subprocess

args = [a for a in sys.argv[1:] if not a.startswith('--')]
use_https = '--https' in sys.argv
port = int(args[0]) if args else (8443 if use_https else 8000)
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)

class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.js': 'text/javascript', '.json': 'application/json', '': 'application/octet-stream'}

def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return '127.0.0.1'

ip = lan_ip()
httpd = socketserver.ThreadingTCPServer(('0.0.0.0', port), H)
scheme = 'http'
if use_https:
    cert, key = os.path.join(here, '.cert.pem'), os.path.join(here, '.key.pem')
    if not os.path.exists(cert):
        subprocess.check_call(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825',
                               '-keyout', key, '-out', cert, '-subj', '/CN=offline-classifier',
                               '-addext', f'subjectAltName=IP:{ip},DNS:localhost'])
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    scheme = 'https'
print(f'Serving {here}\n  PC:    {scheme}://localhost:{port}\n  Phone: {scheme}://{ip}:{port}\nCtrl+C to stop')
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
