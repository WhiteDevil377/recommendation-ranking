"""Local single-user and batch recommendation API."""
import argparse
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import parse_qs,urlparse
from recommend import Service


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8767)
    args=parser.parse_args()
    service=Service()

    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,body):
            data=json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed=urlparse(self.path)
            if parsed.path=="/health":
                self.respond(200,{"status":"ok","model":service.bundle["ranker"].name})
            elif parsed.path=="/recommend":
                try:
                    query=parse_qs(parsed.query)
                    result=service.recommend([int(query.get("user_id",[""])[0])],int(query.get("k",["10"])[0]))
                    self.respond(200,{"results":result})
                except (ValueError,TypeError) as exc:
                    self.respond(400,{"error":str(exc)})
            else:
                self.respond(404,{"error":"Use /health or /recommend"})

        def do_POST(self):
            if self.path!="/recommend":
                self.respond(404,{"error":"Use POST /recommend"})
                return
            try:
                length=int(self.headers.get("Content-Length","0"))
                if not 0<length<=65536:
                    raise ValueError("Invalid request size")
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):
                    raise ValueError("Expected a JSON object")
                self.respond(200,{"results":service.recommend(body.get("user_ids"),body.get("k",10))})
            except (ValueError,TypeError) as exc:
                self.respond(400,{"error":str(exc)})

    server=HTTPServer(("127.0.0.1",args.port),Handler)
    print(f"Local API: http://127.0.0.1:{args.port}",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
