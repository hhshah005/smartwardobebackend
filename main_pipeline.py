import os
import io
import time
import torch
from PIL import Image
from dotenv import load_dotenv
from diffusers import AutoPipelineForInpainting
from google import genai

# ==========================================================
# 🔐 HARDWARE ARCHITECTURE INITIALIZATION
# ==========================================================
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("❌ Missing GEMINI_API_KEY inside your local .env configuration file!")

CUDA_ALIVE = torch.cuda.is_available()
COMPUTE_DEVICE = "cuda" if CUDA_ALIVE else "cpu"
PRECISION_DTYPE = torch.float16 if CUDA_ALIVE else torch.float32

print("=" * 60)
print("🚀 RUNNING SECURE CPU IMAGE-ADAPTER PIPELINE")
print(f"Target Hardware Cluster: {COMPUTE_DEVICE.upper()}")
print("=" * 60)

gemini_client = genai.Client()

# ==========================================================
# 🎨 STEP 1: LIGHTWEIGHT LOCAL IP-ADAPTER ENGINE
# ==========================================================
def execute_cpu_friendly_tryon(user_path: str, outfit_path: str) -> Image.Image:
    print("\n🧠 Loading streamlined local framework (CPU Optimized)...")
    
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=PRECISION_DTYPE
    ).to(COMPUTE_DEVICE)
    
    # 🌟 SECURE VULNERABILITY FIX: 
    # Swapped "ip-adapter_sd15.bin" to "ip-adapter_sd15.safetensors"
    # This completely bypasses the torch.load security check because safetensors are 100% safe!
    print("🧬 Activating Secure Image-Prompt feature extraction adapter...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="models", 
        weight_name="ip-adapter_sd15.safetensors"
    )
    
    if COMPUTE_DEVICE == "cpu":
        pipe.enable_attention_slicing()

    print("📸 Encoding texture layouts...")
    user_image = Image.open(user_path).convert("RGB").resize((512, 512))
    outfit_image = Image.open(outfit_path).convert("RGB").resize((512, 512))
    
    # Set up the bounding box layer over the torso region
    mask_image = Image.new("L", (512, 512), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask_image)
    draw.rectangle([50, 150, 462, 512], fill=255) 

    print("⚡ Rendering layers (Interactive Progress Tracking Active)...")
    
    # 70% of style and colors will be pulled natively from the blazer image layout
    pipe.set_ip_adapter_scale(0.7)
    
    # 🌟 COLOR-NEUTRAL PROMPT: No color words here, letting your input image set the palette!
    prompt = "A high quality realistic photograph of a professional person wearing this exact formal business corporate office blazer suit garment from the reference image, perfect fit, hyperrealistic"
    negative_prompt = "naked, bare skin, deformed clothes, contrasting random colors, low quality, distorted colors, blurry"
    
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=user_image,
        mask_image=mask_image,
        ip_adapter_image=outfit_image, 
        num_inference_steps=12,         
        guidance_scale=6.5,
        strength=0.8
    ).images[0]
    
    return result

# ==========================================================
# 🎬 MASTER PIPELINE WORKFLOW CONTROLLER
# ==========================================================
def main():
    USER = "user.jpg"
    BLAZER = "office_blazer.jpg"
    
    if not os.path.exists(USER) or not os.path.exists(BLAZER):
        print(f"\n⚠️ Quick configuration check: Verify '{USER}' and '{BLAZER}' are inside: C:\\office_presence\\backend\\")
        return

    # 1. Process Local Style Generation Loop
    try:
        start_time = time.time()
        output_image = execute_cpu_friendly_tryon(USER, BLAZER)
        output_image.save("1_tryon_local_output.png")
        print(f"\n✅ Step 1 Complete: Output written to '1_tryon_local_output.png' (Took {time.time() - start_time:.2f}s)")
    except Exception as e:
        print(f"❌ Local processing error: {e}")
        return

    # 2. Process Analytical Progress with Gemini Cloud
    print("\n--- Step 2: Running Gemini Cloud Projections ---")
    timeframes = {
        "4_months": (
            "Look at this image of a person wearing an office outfit. Regenerate this image to visually show "
            "what they look like after 4 months of consistent lean muscle building gym workouts. "
            "Slightly taper and slim down the waistline, tighten the upper arms inside the sleeves, and improve posture. "
            "CRITICAL: Keep the face, hair texture, skin tone, and office outfit layout 100% identical."
        ),
        "1_year": (
            "Look at this image of a person wearing an office outfit. Regenerate this image to visually show "
            "a complete 1-year transformation of dedicated high-intensity weight training. "
            "Give them naturally filled-out athletic chest/shoulders, a highly toned and lean midsection, and a strong confident posture. "
            "CRITICAL: The person's face identity and garment pattern design must remain 100% exactly the same."
        )
    }

    for key, prompt_text in timeframes.items():
        print(f"🔮 Forwarding localized generation frame to Gemini for {key.replace('_', ' ')} track...")
        try:
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[output_image, prompt_text]
            )
            if response.text:
                print(f"✅ Gemini successfully processed instructions for {key}!")
        except Exception as e:
            print(f"❌ Gemini analytical connection skipped: {e}")

    print("\n" + "=" * 60)
    print("🎉 LOCAL CPU-FRIENDLY PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()