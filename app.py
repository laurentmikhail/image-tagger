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


# --- Run the App ---
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
