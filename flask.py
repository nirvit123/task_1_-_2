from flask import Flask, request
app = Flask(__name__)
def reverse_name(name):
    return name[::-1]
def name_length(name):
    return len(name)
@app.route("/")
def home():
    name = request.args.get("name")
    if not name:
        return """
        <h2>❌ No Name Provided</h2>
        <p>Use: <b>?name=yourname</b></p>
        """

    name_upper = name.upper()
    name_reverse = reverse_name(name)
    length = name_length(name)
    return f"""
    <h1>🚀 Flask Name Processor</h1>
    <p><b>Original:</b> {name}</p>
    <p><b>Uppercase:</b> {name_upper}</p>
    <p><b>Reversed:</b> {name_reverse}</p>
    <p><b>Length:</b> {length}</p>
    """
if __name__ == "__main__":
    app.run(debug=True)