# Create and activate a virtual environment (optional but recommended)
# python -m venv myenv
# source myenv/bin/activate  # On Linux/macOS
# myenv\Scripts\activate     # On Windows

# Install PyTorch first - VERY IMPORTANT!
# Go to https://pytorch.org/get-started/locally/ and choose your OS, package manager (pip),
# compute platform (CUDA if you have an NVIDIA GPU, CPU otherwise) and copy the command.
# Example for Linux/Windows with CUDA 11.8:
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Example for macOS (CPU or MPS):
# pip3 install torch torchvision torchaudio
# Example for Linux/Windows (CPU only):
# pip3 install torch torchvision torchaudio

# Install the other libraries
pip install diffusers transformers accelerate Pillow # accelerate helps optimize performance

# Optional: Install Gradio for the web interface
pip install gradio
