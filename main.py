import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import gradio as gr
import time # To track generation time

# --- Configuration ---
MODEL_ID = "stabilityai/stable-diffusion-2-1" # You can choose other models like "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use float16 for GPU inference to save memory, default to float32 for CPU
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"--- Configuration ---")
print(f"Using Device: {DEVICE.upper()}")
print(f"Using Model ID: {MODEL_ID}")
print(f"Using Dtype: {DTYPE}")
print("--------------------")


# --- Load the Diffusion Model ---
print("Loading Stable Diffusion model... (This might take a while on first run)")
start_load_time = time.time()
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        # Optional: Add safety checker if desired (requires extra installation)
        # safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
        # feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    )
    pipe = pipe.to(DEVICE)

    # Optimizations for GPU (if available)
    if DEVICE == "cuda":
        # Reduces memory usage at a slight performance cost
        pipe.enable_attention_slicing()
        # Optional: Use xformers for potentially faster inference & less memory
        # try:
        #     import xformers
        #     pipe.enable_xformers_memory_efficient_attention()
        #     print("xformers memory efficient attention enabled.")
        # except ImportError:
        #     print("xformers not installed. Install with 'pip install xformers' for potential optimization.")


except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure you have enough RAM/VRAM and a stable internet connection.")
    # If you have persistent issues, try clearing the Hugging Face cache
    # located usually at ~/.cache/huggingface/hub
    exit() # Exit if model loading fails

load_time = time.time() - start_load_time
print(f"Model loaded successfully in {load_time:.2f} seconds.")

# --- Image Generation Function ---
def generate_image(prompt, neg_prompt="", guidance_scale=7.5, num_steps=50):
    """Generates an image based on the prompt using the loaded pipeline."""
    print("\n--- Generating Image ---")
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{neg_prompt}'")
    print(f"Guidance Scale: {guidance_scale}")
    print(f"Inference Steps: {num_steps}")

    start_gen_time = time.time()

    # Ensure the pipeline is on the correct device (important if switching runtimes)
    pipe.to(DEVICE)

    try:
        with torch.inference_mode(): # Use inference_mode for efficiency
             # On some setups you might need torch.autocast("cuda") here for float16:
             # with torch.autocast("cuda"):
                image = pipe(
                    prompt,
                    negative_prompt=neg_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps
                ).images[0] # Get the first image from the output list

        gen_time = time.time() - start_gen_time
        print(f"Image generated in {gen_time:.2f} seconds.")
        print("------------------------")
        return image

    except torch.cuda.OutOfMemoryError:
        print("ERROR: CUDA Out Of Memory! Try reducing image dimensions (if possible in the model/pipeline),")
        print("       using attention slicing (enabled), or using a smaller model.")
        print("------------------------")
        # Return a placeholder or raise an error for Gradio
        # Create a simple black image as an error indicator
        error_img = Image.new('RGB', (256, 256), color = 'black')
        # You could add text here too if desired
        return error_img # Or raise gr.Error("CUDA Out of Memory") - requires Gradio >= 3.20

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        print("------------------------")
        error_img = Image.new('RGB', (256, 256), color = 'red')
        return error_img # Or raise gr.Error(f"Generation failed: {e}")


# --- Gradio Interface ---
print("Setting up Gradio interface...")

# Define input/output components
prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your image description here...")
neg_prompt_input = gr.Textbox(label="Negative Prompt", placeholder="(Optional) Describe what you DON'T want to see...")
guidance_slider = gr.Slider(minimum=1, maximum=20, step=0.5, value=7.5, label="Guidance Scale (How strongly the prompt influences the image)")
steps_slider = gr.Slider(minimum=10, maximum=100, step=1, value=50, label="Inference Steps (More steps = potentially higher quality, but slower)")
image_output = gr.Image(label="Generated Image", type="pil") # type="pil" ensures it handles PIL images

# Define the theme (Optional - gives a slightly more modern/minimal feel)
# Experiment with different themes: https://www.gradio.app/guides/theming-guide
theme = gr.themes.Default(primary_hue=gr.themes.colors.blue)
# theme = gr.themes.Glass() # Example of another theme

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[prompt_input, neg_prompt_input, guidance_slider, steps_slider],
    outputs=image_output,
    title="🎨 Text-to-Image AI Generator",
    description="""
    Generate images from text descriptions using Stable Diffusion.
    Enter a prompt, optionally add things to avoid in the negative prompt, and adjust generation parameters.
    """,
    # theme=theme, # Uncomment to apply the custom theme
    allow_flagging='never', # Disables the flagging feature
    examples=[ # Add some example prompts
        ["A futuristic cityscape at sunset, synthwave style", "", 7.5, 50],
        ["A photorealistic portrait of an astronaut on Mars", "cartoon, drawing, illustration, sketch", 9, 60],
        ["A delicious steaming bowl of ramen, anime food style", "photorealistic, low quality", 7.0, 40],
    ]
)

# --- Launch the App ---
print("Launching Gradio app... Access it at the URL provided below.")
iface.launch() # Share=True creates a temporary public link (use with caution)
print("App launched. Press Ctrl+C in the terminal to stop.")
