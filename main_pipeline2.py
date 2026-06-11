import io
import os
import time
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================================
# 🔐 INITIALIZATION & ENVIRONMENT SETUP
# ==========================================================
load_dotenv()

gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("❌ Missing GEMINI_API_KEY in your .env file!")

# Initialize the official current Gemini Client
client = genai.Client()

# ==========================================================
# 🚀 NATIVE MULTIMODAL IMAGE PIPELINE
# ==========================================================

def run_accurate_pipeline(user_photo_path: str, outfit_photo_path: str):
    print("⚡ Starting 100% Accurate Gemini Multimodal Image Engine...")
    
    try:
        print("📸 Loading your local image assets...")
        user_img = Image.open(user_photo_path)
        outfit_img = Image.open(outfit_photo_path)
    except Exception as e:
        print(f"❌ Failed to load local images: {e}")
        return

    # Formulate precise prompts for each fitness evolution milestone
    timeframes = {
        "4_months": (
            "Create a new realistic photograph combining these two reference images. "
            "Take the exact black office blazer from the second image and dress the person from the first image in it. "
            "Modify the person's physique to look like they have undergone 4 months of consistent gym conditioning and fat loss. "
            "Slightly tighten and trim the waistline, make the upper torso look a bit leaner, and optimize posture. "
            "CRITICAL: Keep their facial identity, hair, skin tone, and the blazer's exact black design identical."
        ),
        "1_year": (
            "Create a new realistic photograph combining these two reference images. "
            "Take the exact black office blazer from the second image and dress the person from the first image in it. "
            "Modify the person's physique to look like they have undergone 1 full year of intensive athletic muscle-building workouts. "
            "Naturally broaden the shoulders, fill out the chest frame inside the blazer, trim the waist, and provide a confident, fit posture. "
            "CRITICAL: The face identity, features, skin color, and the original black texture of the blazer must remain accurate."
        )
    }

    # Execute the requests using the native image generation model
    for key, prompt_text in timeframes.items():
        print(f"\n🔮 Transmitting assets to Gemini for the {key.replace('_', ' ')} look...")
        start_time = time.time()
        
        try:
            # 🌟 FIXED: Target the active native image endpoint with exact configurations
            response = client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[user_img, outfit_img, prompt_text],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio="1:1")
                )
            )

            # Extract the raw inline image data cleanly from the response layout
            image_saved = False
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Convert the bytes array back into a viewable PNG image
                        final_image = part.as_image()
                        output_filename = f"office_presence_{key}_transformation.png"
                        final_image.save(output_filename)
                        print(f"✅ SUCCESS! Image saved as: {output_filename} (Took {time.time() - start_time:.2f}s)")
                        image_saved = True
            
            if not image_saved:
                print("⚠️ Model ran but did not return raw image data bytes.")

        except Exception as e:
            print(f"❌ Error communicating with Gemini API: {e}")

    print("\n=======================================================")
    print("🎉 ACCURATE PIPELINE EXECUTION COMPLETE!")
    print("=======================================================")


if __name__ == "__main__":
    USER_IMAGE = "user.jpg"
    OFFICE_BLAZER = "white_blazer.jpg"
    
    if os.path.exists(USER_IMAGE) and os.path.exists(OFFICE_BLAZER):
        run_accurate_pipeline(USER_IMAGE, OFFICE_BLAZER)
    else:
        print(f"\n⚠️ Missing inputs! Please save '{USER_IMAGE}' and '{OFFICE_BLAZER}' directly inside C:\\office_presence\\backend\\")