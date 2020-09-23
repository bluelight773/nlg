from flask import Flask, render_template, request

from aitextgen import aitextgen

app = Flask(__name__)

# Load CPU Shakespeare model
ai = aitextgen(model="large_models/gpu_gpt2_shakespeare/pytorch_model.bin",
               config="large_models/gpu_gpt2_shakespeare/config.json",
               vocab_file="large_models/gpu_gpt2_shakespeare/aitextgen-vocab.json",
               merges_file="large_models/gpu_gpt2_shakespeare/aitextgen-merges.txt")
               # to_gpu=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/", methods=['GET', 'POST'])
def streambyte():
    # your file processing code is here...
    prompt = str(request.form['prompt']).strip()
    if prompt:
        gen_text = ai.generate(1, prompt=prompt, return_as_list=True)[0]
    else:
        gen_text = ai.generate(1, return_as_list=True)[0]

    gen_html = gen_text.replace("\n", "</br>")

    return render_template("index.html", gen_html = gen_html)


if __name__ == "__main__":
    app.run(host="0.0.0.0")