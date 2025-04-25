from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Load the trained model
embedding_model = joblib.load('embedding_model.pkl')

# Load recipes
def load_recipes():
    try:
        with open('recipes_and_substitutes.json', 'r') as f:
            recipes = json.load(f)
            print("Loaded recipes:", list(recipes.keys()))
            return recipes
    except Exception as e:
        print(f"Error loading recipes: {e}")
        return {}

# Load FAQs
def load_faqs():
    try:
        with open('faqs.json', 'r') as f:
            faqs = json.load(f)
            print("Loaded FAQs:", list(faqs.keys()))
            return faqs
    except Exception as e:
        print(f"Error loading FAQs: {e}")
        return {}

RECIPES = {k.lower(): v for k, v in load_recipes().items()}  #convert keys to lowercase for better matching
FAQs = load_faqs()

greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

# Normalize input text
def normalize(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    return re.sub(r'\s+', ' ', text)

# Match input to known recipes/FAQs
def find_best_match(user_message):
    user_vector = embedding_model.encode([user_message])

    # FAQ match
    faq_questions = list(FAQs.keys())
    faq_vectors = embedding_model.encode(faq_questions)
    faq_similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    faq_best_index = faq_similarities.argmax()
    faq_score = faq_similarities[faq_best_index]

    # Recipe match
    recipe_questions = [
        f"how to make {r}" for r in RECIPES.keys()
    ] + [
        f"ingredients for {r}" for r in RECIPES.keys()
    ] + [
        f"steps for {r}" for r in RECIPES.keys()
    ]
    recipe_vectors = embedding_model.encode(recipe_questions)
    recipe_similarities = cosine_similarity(user_vector, recipe_vectors).flatten()
    recipe_best_index = recipe_similarities.argmax()
    recipe_score = recipe_similarities[recipe_best_index]

    print(f"FAQ match: {faq_questions[faq_best_index]} (score: {faq_score:.2f})")
    print(f"Recipe match: {recipe_questions[recipe_best_index]} (score: {recipe_score:.2f})")

    if faq_score > recipe_score and faq_score > 0.3:
        return ("faq", faq_questions[faq_best_index])
    elif recipe_score > 0.3:
        return ("recipe", recipe_questions[recipe_best_index])

    return (None, None)


# Extract ingredient and recipe from substitution query
def extract_recipe_and_ingredient(message):
    patterns = [
        r"substitute for (.+?) in (.+)",
        r"alternative to (.+?) in (.+)",
        r"replacement for (.+?) in (.+)",
        r"use instead of (.+?) in (.+)",
        r"instead of (.+?) in (.+)",
        r"what can i use instead of (.+?) in (.+)",
        r"what is a substitute for (.+?) in (.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            ingredient = normalize(match.group(1))
            recipe = normalize(match.group(2))
            return recipe, ingredient
    return None, None

def extract_recipe_property_query(user_message):
    patterns = {
        "cooking_time": [
            r"cooking time for (.+)",
            r"how long to bake (.+)",
            r"baking time for (.+)"
        ]
    }

    for prop, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, user_message)
            if match:
                recipe = normalize(match.group(1))
                return prop, recipe
    return None, None

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").lower().strip()
    normalized_user_msg = normalize(user_message)

    # Greeting
    if any(greet in normalized_user_msg for greet in greetings):
        return jsonify({"response": "Hello! How can I assist you?", "type": "greeting"})

    # Substitution query
    recipe_name, ingredient = extract_recipe_and_ingredient(normalized_user_msg)
    if recipe_name and ingredient:
        if recipe_name in RECIPES:
            substitutes = RECIPES[recipe_name].get("substitutes", {}).get(ingredient)
            if substitutes:
                return jsonify({
                    "response": f"You can substitute **{ingredient}** in {recipe_name} with: {substitutes}",
                    "type": "substitute"
                })
            else:
                return jsonify({
                    "response": f"Sorry, I couldn't find a substitute for **{ingredient}** in **{recipe_name}**.",
                    "type": "substitute"
                })
            
    prop, recipe = extract_recipe_property_query(normalized_user_msg)
    if prop and recipe in RECIPES:
        if prop == "cooking_time":
            return jsonify({
                "response": f"The cooking time for {recipe.title()} is {RECIPES[recipe]['cooking_time']}.",
                "type": "cooking_time"
            })

    # FAQ or recipe handling
    query_type, best_match = find_best_match(normalized_user_msg)

    if query_type == "faq":
        return jsonify({
            "response": FAQs[best_match],
            "type": "faq"
        })
    elif query_type == "recipe":
        if "how to make" in best_match or "steps for" in best_match:
            recipe_name = best_match.replace("how to make ", "").replace("steps for ", "")
            recipe_name = normalize(recipe_name)
            if recipe_name in RECIPES:
                steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(RECIPES[recipe_name]['steps'])])
                return jsonify({
                    "response": f"Here's how to make {recipe_name.title()}:\n{steps}",
                    "type": "recipe_steps"
                })

        elif "ingredients for" in best_match:
            recipe_name = best_match.replace("ingredients for ", "")
            recipe_name = normalize(recipe_name)
            if recipe_name in RECIPES:
                ingredients = "\n".join([f"- {ing}" for ing in RECIPES[recipe_name]['ingredients']])
                return jsonify({
                    "response": f"Here are the ingredients for {recipe_name.title()}:\n{ingredients}",
                    "type": "recipe_ingredients"
                })

    return jsonify({
        "response": "Sorry, I didn't catch that. Try asking about a recipe, ingredient, or FAQ.",
        "type": "unknown"
    })

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=10000)
