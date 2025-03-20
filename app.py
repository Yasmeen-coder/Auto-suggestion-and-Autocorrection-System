from flask import Flask, render_template, request
import pandas as pd
import textdistance
import re
from collections import Counter

app = Flask(__name__)

# Read and process the text file to get the words and their frequencies
with open('words.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)
    words += words  # Duplicate the list to ensure all words are considered

# Create a set of unique words and their frequency dictionary
V = set(words)
words_freq_dict = Counter(words)
Total = sum(words_freq_dict.values())

# Calculate the probability of each word
probs = {k: v / Total for k, v in words_freq_dict.items()}

@app.route('/')
def index():
    return render_template('index.html', suggestions=None)

@app.route('/suggest', methods=['POST'])
def suggest():
    keyword = request.form['keyword'].lower()
    if keyword:
        # Calculate the Jaccard similarity between the keyword and each word in the dictionary
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, keyword) for v in words_freq_dict.keys()]
        
        # Create a DataFrame with words, their probabilities, and their similarities to the keyword
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df.columns = ['Word', 'Prob']
        df['Similarity'] = similarities
        
        # Sort the DataFrame by similarity and probability
        suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False)[['Word', 'Similarity']]
        
        # Convert the DataFrame to a list of dictionaries for rendering
        suggestions_list = suggestions.to_dict('records')
        
        return render_template('index.html', suggestions=suggestions_list)

if __name__ == '__main__':
    app.run(debug=True)
