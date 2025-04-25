# 🤖 cozyBite(e-commerce-app) Chatbot

This is a smart chatbot built using **Python + Flask**, designed to respond to queries about:

- Recipes (steps, ingredients)
- Cooking time
- Ingredient substitutions
- FAQs (delivery, hours, etc.)
- Friendly greetings

It uses semantic search powered by the `sentence-transformers` library and handles user queries using cosine similarity matching.

---

## Dependencies

This chatbot uses the following libraries:

- `Flask`
- `flask-cors`
- `joblib`
- `scikit-learn`
- `sentence-transformers`
- `json`
- `re`
- `string`

### Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### Install the required packages:
```
pip install -r requirements.txt

pip install flask flask-cors joblib scikit-learn sentence-transformers 
```

### Before running the chatbot, you need to generate the embedding model and save it.
### To train chatbot:
```
python train_chatbot_model.py
```
### Run program:
```
python app.py
```
