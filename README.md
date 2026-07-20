![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/banner_llama.jpg)
# Image-Gradio-Gguf
Status: Working
- Recently v1.28 - Now supporting both z-image-turbo and flux 2.
- Still updating readme for flux2...

### Description
A local python image generator from prompt using, 1 various compatible Encoders, and 2 choice of either, Z-Image-Turbo-GGUF or Flux-2-Klein-GGUF. This is a super, TXT to IMG and IMG to IMG image, generation project, covering what is possible currently to be most compitent through lightweight GGUF models that can fit on a 8GB GPU, but also scales to larger VRAM if you have that, and of course if you want you can also run it through CPU. The installer compiles for your specific CPU/GPU combo for the BEST speeds in inference you will find out there right now. As you can see it generates OK images, but has the typical issues associated with the libraries/models one would expect. The program uses a separate encoder file (optional but just do it), and it ensures to load this in one-shot mode, ensuring that both models will be loaded individually to the GPU, maxizing the potential size of the Image generation model. All that said, there may be a little learning if you are used to Grok or something of that level of simplicity.

### Output
- Preferences page, recently created and minimal, but strictly program/interface settings go here (v1.28)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/flux_results.jpg)

### Media
- Generation page showing a image to image, and other development images. (v1.28)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/generate_page.jpg)

- Configure page, recently refined/corrected and 1 detail moved to new Preference page (v1.21)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/configuration_page.jpg)

- Preferences page, recently created and minimal, but strictly program/interface settings go here (v1.28)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/preferences_page.jpg)

- Program starts something like this (A013)...
```
================================================================================
      Image-Gradio-Gguf: Starting Program...
================================================================================

  Versioning: Python 3.12.4; Gradio 6.19.0

  CPU     : AMD Ryzen 9 3900X 12-Core Processor
  Threads : 21 (85% of 24 logical cores)
  Vulkan  : True  (1.4.341)
    GPU0: NVIDIA GeForce GTX 1060 3GB
    GPU1: Radeon (TM) RX 470 Graphics

  llama-cli : C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\data\llama_cpp_binaries\llama-cli.exe
  sd        : C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\data\stable_diffusion_binaries\sd-cli.exe

  Encoder  : OK — G:/LargeModels/Text and Image/Qwen3-4b-Z-Image-Turbo-AbliteratedV1-GGUF/Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q4_K_M.gguf
  Diffusion: OK — G:/LargeModels/Text and Image/Z-Image-Turbo-GGUF/z_image_turbo-Q4_0.gguf
  VAE      : OK — G:\LargeModels\Text and Image\Z-Image-Turbo-GGUF\ae.safetensors

[gallery] Scanning for Thumbnails....
[gallery] Rescanned C:\Inference_Files\Image-Gradio-Gguf\Image-Gradio-Gguf\output: 3 images

* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.

```

### Requirements:
- Operating System - Deepseek assesed this as Windows 10-11, however I am testing on Windows 10 22h2.
- Python - Deepseek assesed this as Python versions 3.11-3.13, however I am testing on Python 3.12.
- Graphics- Testing on Vulkan 1.3 card with 1.4 driver. Also its a min DirectX 11.1 GPU for gradio/pyqt6 (browser interface).
- Processor - Detecting/supporting, SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX512, F16C, FMA, architecture features. Testing upon Zen 2 with AOCL installed. 
- Libraries - Required libraries is handled by the installer script, but they include both, Llama.Cpp and Stable Diffusion.
- Building - VS 2022 C++, specifically the Desktop Build Tools including CMake. Additionally the Vulkan 1.4 SDK. Additionally Windows 10/11 SDK relevant to your os version.

### Models (basic z-image-turbo):
I put Q# because it should support any quantization, the model variety will be expanded upon later....
- First you need an encoding model, get "Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf" from here [Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF](https://huggingface.co/LuffyTheFox/Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF). The encoder should be a Qwen3 4b Q4 or something of that level, we are mainly dealing with the positive/negative prompt, just dont write a book in those boxes.
- Second you need an image generation model, get "z_image_turbo-Q#.gguf" from [Vanilla Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF). The image generation model should be the highest quantization you can fit on your GPU. The filesize does not represent the loaded/operating size. Ask AI which is the largest quantization of said url Provided gguf model, that will safely load on your specified GPU, and download that one.
- Third the "ae.safetensors", this is only available from the [Vanilla Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) files, get it from there, use it with all z-image-turbo image generation model variants. One is also able to split this to CPU, for a little more space in VRAM, I do, but you might not if you had 12GB+ VRAM.

### Models

All encoders and image models are **GGUF** — on [HuggingFace](https://huggingface.co) look for the GGUF builds (any quantization, shown here as `Q#`). Each family also needs one `.safetensors` VAE, downloaded once.

**Encoder rule:** the encoder size must match the image model — Qwen3-**4B** for Z-Image-Turbo and Flux.2-klein-**4B**, Qwen3-**8B** for Flux.2-klein-**9B**. The program auto-detects this and warns on a mismatch. One Qwen3-4B encoder serves both Z-Image-Turbo and klein-4B. Qwen3-VL models work as text-only encoders (no mmproj needed); Qwen2.5 encoders are **not** compatible with Flux.2.

#### Z-Image-Turbo
Encoders — Qwen3-4B:
```
Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf
Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf
qwen_3_4b.Q#.gguf              (plain Qwen3-4B)
Qwen3-VL-4B*.Q#.gguf           (VL, text-only)
```
Image models:
```
z_image_turbo-Q#.gguf
darkBeastMar1526Latest_dbzit8SDAFOK-Q#.gguf
darkBeastMar2126Latest_dbzit9DIMRclaw-Q#.gguf
eventHorizon_zitV10-Q#.gguf
perfeczion_10BF16-Q#.gguf
smoothmixUltimate_zimageTurboV10-Q#.gguf
zImageTurboAnime_v10-Q#.gguf
zImageTurboNSFW_60BF16Diffusion-Q#.gguf
zImageTurboNSFW_61BF16Diffusion-Q#.gguf
```
VAE — from [Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF):
```
ae.safetensors
```

#### Flux.2-klein 4B
Encoders — Qwen3-4B (same set as Z-Image-Turbo):
```
Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf
Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf
qwen_3_4b.Q#.gguf              (plain Qwen3-4B)
Qwen3-VL-4B*.Q#.gguf           (VL, text-only)
```
Image models:
```
flux-2-klein-4b-Q#.gguf
flux-2-klein-base-4b-Q#.gguf
```
VAE — the `vae/` file from [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B):
```
diffusion_pytorch_model.safetensors
```

#### Flux.2-klein 9B
Encoders — Qwen3-8B:
```
Qwen3-8b-erotic-heretic-Q#.gguf
Qwen3-8B-Gemini-2.5-Flash-Uncensored-Q#.gguf
qwen_3_8b.Q#.gguf              (plain Qwen3-8B)
qwen3-vl-flux2-8b-Q#.gguf      (VL, text-only)
Qwen3-VL-8B*.Q#.gguf           (VL, text-only)
```
Image models:
```
flux-2-klein-9b-Q#.gguf
flux-2-klein-base-9b-Q#.gguf
```
VAE — the `vae/` file from [FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B):
```
diffusion_pytorch_model.safetensors
```

### Instructions:
Currently...
```
1. Ensure you have downloaded the correct models from Huggingface (see `Models:` section above), put them on your model storage drive (if you have one). You can put them in .\models if you like, but then this will cause a bloated folder, and you may not remember they are in there later. 
2. Ensure to download the latest release version to a suitable location, then unpack to the place you intend to have the program.
3. Run the program via right click run as admin on `Image-Gradio-Gguf.bat`, this will launch the batch menu.
4. Ensure that Python/Pip has internet access, these may request it during install (if that is an issue you may need to start install again), and the libraries/packages will install appropriately to the program folder, not globally. After which there will be a summary, and you will be returned to the batch menu. hopefully everything went ok for you. If there are any issues at this stage, I would suggest the installer should indicate what the issue was, so maybe paste that into AI with the installer script to fix your system compatibility.
5. Back on the batch menu, select 1 to run the application, the server will start up, and then the built-in browser will pop-up its own window with the Interface displayed. Ensure to go to Configuration page, to set model paths, and what is going to be loaded where, if there is not enough ram on the relating device, it will say in the output (see notes below). Adter configuring, go back to the Generation page, type in your positive prompt, and then hit generate. After you done your first image and everything is confirmed working, then possibly configure the settings further and produce a new prompt, and keep going til you have your images.
6. Upon exiting the program correctly through the exit button, the user will be returned to the batch menu, and one would then exit from there, or otherwise one could just click the [x] in the top right of all windows associated.
```

### Examples:
- If you want to test the image generation, then I suggested just write something like `A picture of a Woodchuck standing next to a pile of wood while juggling small logs of wood.`, or `A man walking his dog on the meadow on a sunny day.`, or if you want to do image to image then possibly `A photo-realistic version of the provided image.`.

### Notation:
- If you want to generate a 1024x1024 size image, be aware, this creates ~3GB of overhead on the GPU if thats where the Image Generation model is loaded, while a 768x768 image would have ~1.8GB of overhead...consider such things when it tells you it ran out of ram.
- The assessment by OPUS said, the reason why I could not fit Q4 ImageGen model with DP on Full while could fit Q8 ImageGen model with DP on Split, is because the difference between DP on Split or FUll, is up to 4.6GB extra on top. Keep in mind the models are done in 1-shot mode not m-lock.
- Something to consider is how much memory the Image model takes, image models need more space when loaded compared to a text model, if yo uneed more room for the image model then try Diffuser Placement is set to Split. So some tweaking settings may be requried with low VRAM. 
- 512x512 or less on flux 2 seems buggy, however, 512x768 or 768x512, seems to work good, while for me 768x768 will not fit in 8GB VRAM.

### Development:
Development is somewhat stopped for now due to having implemented, z-image-turbo (txt-img) and flux 2 (img-img, txt-img), however ideas for improvement will be here...
- Flux 2 model variants, possibly ungated or reinforced.

### Structure:
- Scripts structure is...
```
Image-Gradio-Gguf/
├── Image-Gradio-Gguf.bat      # Windows launcher batch file
├── launcher.py                    # Startup, Shutdown, Main Loop.
├── installer.py                   # Download, Setup & Build, Install, creation/rectiation of json, creation/recreation of constants.ini.
├── data/
│   ├── constants.ini              # System constants & defaults
│   └── persistent.json            # User configuration (auto-managed)
├── scripts/                       # All scripts have 6-10 letter names
│   ├── __init__.py                # Empty init file.
│   ├── configure.py                # Configuration, and all global variables/constants/maps/lists are here.
│   ├── display.py                  # Gradio, Browser, Python Displays.
│   ├── inference.py                  # Image generation, Model Handling, Text Generation.
│   ├── utilities.py                  # General code, and code that is not more appropriate to be in other scripts.
├── models/                        # Default model directory
└── output/                        # Generated images directory
```

### Disclaimer:
- While this program is designed to be able to create images without filtering, the idea being simpler less complex prompting in order to achieve intended result, for purposes such as for example illustrating a book, it may also generate images you dont intend, but you the "User" yourself are responsible for the contents/theme in the outputted images, by the action of the Editing of, the Positive Promt (which by default starts blank) and the modification of the Negative Prompt (which by default contains some helpful generic text segments not intended for image).
