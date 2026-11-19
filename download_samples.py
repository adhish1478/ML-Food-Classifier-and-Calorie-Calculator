import os
import urllib.request

SAMPLES = {
    "pizza.jpg": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=500&q=80",
    "burger.jpg": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500&q=80",
    "salad.jpg": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=500&q=80",
    "biryani.jpg": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=500&q=80",
    "dosa.jpg": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=500&q=80"
}

target_dir = os.path.join(os.path.dirname(__file__), "sample_images")
os.makedirs(target_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0'}
for filename, url in SAMPLES.items():
    dest = os.path.join(target_dir, filename)
    if not os.path.exists(dest):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp, open(dest, 'wb') as out_file:
                out_file.write(resp.read())
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed {filename}: {e}")
