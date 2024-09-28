from flask import request
from config import app
from main import title_tag


@app.route("/")
def main():
    return "hello world"


@app.get("/oldbaseline")
def answer():
    text = request.get_data(as_text=True)
    tag = title_tag(text)
    return text, tag


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
