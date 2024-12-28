from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Game categories and words
categories = {
    "Fruits": ["apple", "banana", "cherry", "mango", "orange", "grape", "pineapple"],
    "Animals": ["elephant", "tiger", "giraffe", "kangaroo", "zebra", "dolphin", "panda"],
    "Technology": ["python", "computer", "internet", "robotics", "software", "algorithm", "database"]
}

# Game state
score = 0
selected_category = None
word = None
jumbled_word = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start_game', methods=['GET'])
def start_game():
    global selected_category, word, jumbled_word
    selected_category = random.choice(list(categories.keys()))
    word = random.choice(categories[selected_category])
    jumbled_word = ''.join(random.sample(word, len(word)))

    return jsonify({
        "category": selected_category,
        "jumbled_word": jumbled_word
    })

@app.route('/check_guess', methods=['POST'])
def check_guess():
    global word, score
    guess = request.form['guess'].strip().lower()
    correct = guess == word
    if correct:
        score += 1
        return jsonify({
            "correct": True,
            "score": score,
            "message": f"Correct! The word was: {word}. Your score: {score}"
        })
    else:
        return jsonify({
            "correct": False,
            "message": f"Wrong! Try again. The jumbled word is: {jumbled_word}"
        })

if __name__ == "__main__":
    app.run(debug=True)
