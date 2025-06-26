import os
from openai import OpenAI
from flask import Flask, request, jsonify

# --- Initialize Flask App and OpenAI Client ---
app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- The core function to get tags ---
def get_image_tags(image_url):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and generate a list of 15-20 relevant, comma-separated tags. Focus on objects, setting, mood, and color. Do not provide any other text or explanation."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "low"},
                        }
                    ],
                }
            ],
            max_tokens=100
        )
        tags = response.choices[0].message.content.strip()
        return tags
    except Exception as e:
        return {"error": str(e)}

# --- API Endpoint ---
@app.route('/tag-image', methods=['POST'])
def tag_image_endpoint():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "Missing 'image_url' in request body"}), 400

    image_url = data['image_url']
    tags_result = get_image_tags(image_url)

    if isinstance(tags_result, dict) and "error" in tags_result:
        return jsonify(tags_result), 500

    tag_list = [tag.strip() for tag in tags_result.split(',')]
    return jsonify({"tags": tag_list})
# --- NEW: Function to find the best matching image ---
def find_best_match(search_text, image_data):
    """
    Calculates a match score for each image based on tag overlap with search text.
    
    Args:
        search_text (str): The user's input text.
        image_data (list): A list of dictionaries, e.g.,
                           [{'url': 'http://...', 'tags': ['dog', 'park', 'happy']}, ...]
                           
    Returns:
        str: The URL of the best matching image, or None if no match is found.
    """
    # Clean up the search text into a set of unique words for fast checking
    search_words = set(search_text.lower().split())
    
    best_image_url = None
    highest_score = -1

    for image in image_data:
        # Ensure the image has tags to compare against
        if 'tags' not in image or not image['tags']:
            continue

        # Create a set of the image's tags (all lowercase)
        image_tags = set(tag.lower() for tag in image['tags'])
        
        # Find the tags that are also in the search text (the "intersection")
        matching_tags = search_words.intersection(image_tags)
        
        # Calculate the score. We can simply use the number of matching tags.
        score = len(matching_tags)
        
        # If this image has a better score than the previous best, update it
        if score > highest_score:
            highest_score = score
            best_image_url = image.get('url') # Use .get() for safety
            
    return best_image_url

# --- NEW: API Endpoint for finding the best image ---
@app.route('/find-best-image', methods=['POST'])
def find_best_image_endpoint():
    data = request.get_json()
    
    # Validate the incoming data
    if not data or 'search_text' not in data or 'image_data' not in data:
        return jsonify({"error": "Request must include 'search_text' and 'image_data'"}), 400
        
    search_text = data['search_text']
    image_data = data['image_data']
    
    # Get the best match from our new function
    best_match_url = find_best_match(search_text, image_data)
    
    if best_match_url:
        return jsonify({"best_match_url": best_match_url})
    else:
        # If no image had any matching tags, return a not found error
        return jsonify({"error": "No suitable image found for the given text"}), 404

# --- Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
