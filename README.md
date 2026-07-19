![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/banner_llama.jpg)
# Image-Gradio-Gguf
Status: Beta (working but further development possible)
- Current Phase v1.19-v1.25 - Z-Image-Turbo support expanded with individual handling of "ae.safetensors". A new a Preferences page. Some fixes/improvements.
- v1.26-v1.27 - Small but significant improvments. Confirmed some of the Collection models are working.
- Next Phase ~v1.28+ - Two words "Flux Klein", enabling AI Image Editing. But, must be happy 100% with z-image-turbo performance first, then can just fall-back if its impossible.

### Description
A local python image generator from prompt using Qwen 3 Z-Image Engineer encoder and Z-Image Turbo. This is a simple image generation project, covering what is possible currently to be most compitent through GGUF models, through but eventually cover several encoders and image generation models. While this program work great for what it does, it is also example scripts for AI on how to do image inference with such libraries/models, and useful in the production of other progreams that require such things. As you can see it generates OK images, they can look pretty real, but has the typical issues one would exoect under such restrictions and with AI image generation in general, but for simple images I think it will be effective in generating your result in a few iterations, so long as the request is not too barmy.   

### Media
- Generation page, with the new interactive section titles (v1.27)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/generate_page.jpg)

- Configure page, recently refined/corrected and 1 detail moved to new Preference page (v1.21)...  
![image missing](https://raw.githubusercontent.com/wiseman-timelord/Image-Generator-Gguf/refs/heads/main/media/configuration_page.jpg)

- Preferences page, recently created and minimal, but strictly program/interface settings go here (v1.21)...  
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

### Models (basic):
I put Q# because it should support any quantization, the model variety will be expanded upon later....
- First you need an encoding model, get "Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf" from here [Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF](https://huggingface.co/LuffyTheFox/Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF). The encoder should be a Qwen3 4b Q4 or something of that level.
- Second you need an image generation model, get "z_image_turbo-Q#.gguf" from [Vanilla Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF). The image generation model should be the highest quantization you can fit on your GPU. The filesize does not represent the loaded/operating size. Ask AI which is the largest quantization of said url Provided gguf model, that will safely load on your specified GPU, and download that one, there is also diffuser split to consider for a little more space in VRAM, I would.
- Third the "ae.safetensors", this is only available from the [Vanilla Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) files, get it from there, use it with all z-image-turbo image generation model variants.

### Models (advanced):
- Image Encoder Models Supported, find them on [HuggingFace.Co](https://huggingface.co), but be sure to look for the GGUF versions:
```
Qwen3-4b-Z-Image-Turbo-AbliteratedV1.Q#.gguf
Qwen3-4b-Uncensored-Z-Image-Engineer-V4-Q#.gguf
Qwen3-8b-erotic-heretic-Q#.gguf
Qwen3-8B-Gemini-2.5-Flash-Uncensored-Q#.gguf
```
- Image Generation Models Supported, find them on [HuggingFace.Co](https://huggingface.co), but be sure to look for the GGUF versions:
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
- ae.safetensors,  you still going to need that from [Vanilla Z-Image-Turbo-GGUF](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF) 
```
ae.safetensors
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
- If you want to test the image generation, then I suggested just write something like `A picture of a Woodchuck standing next to a pile of wood while juggling small logs of wood.`, I cant remember the exact prompt, but you can compare it to my picture of a Woodch uck standing next to a pile of wood and juggling small logs of wood.

### Notation:
- If you want to generate a 1024x1024 size image, be aware, this creates ~3GB of overhead on the GPU if thats where the Image Generation model is loaded, while a 768x768 image would have ~1.8GB of overhead...there are such things to consider when it tells you it ran out of ram.
- The assessment by OPUS said, the reason why I could not fit Q4 ImageGen model with DP on Full while could fit Q8 ImageGen model with DP on Split, is because the difference between DP on Split or FUll, is up to 4.6GB extra on top. Keep in mind the models are done in 1-shot mode not m-lock.
- Something to consider is how much memory the Image model takes, image models need more space when loaded compared to a text model, if yo uneed more room for the image model then try Diffuser Placement is set to Split. So some tweaking settings may be requried with low VRAM. 
- If you have older hardware, then I strongly advise generating 256x256 images unless you need them larger, as optimally 256x256 will take little time compared to 512x512 images. This is not such a problem if you have newer/expensive hardware. 
- The Qwen3 encoder will do a good job of turning a bad prompt into something workable from a small input, but this is all experimental and an experiment too, so do not expect premium AI image quality, gguf versions of image generation models are sparse, you can see its slightly dated now because Qwen3 is the only encoder, but not that old that its naff. 

### Development:
Development is somewhat stopped for now due to, v1 done (working + nice) and funding issues, in the mean time you could check out my donation/sponsorship links on profile page, but there are still some improvements possible/planned...
- New Models Options for Z-Image-Turbo - https://huggingface.co/BigDannyPt/Z-Image-Turbo-GGUF-Collection
- Moved from notation.
```
more models supported for Flux Klein integration will be attempted after that, though this will be highly experimental, because I think it needs comfy UI, so will attempt to integrate it, but advanced features will possibly be limited, it may be generation only, or I may be able to have dynamic interface depending upon if using flux or z-image. We will see. I just want to get it working here, before I integrate it into my Agentic framework project, as a cheat sheet. Possibly I can just use the encoders here "Qwen3-Uncensored-TextEncoders-FLUX-Klein-Z-Image-Turbo-GGUF", for both, I have that encoder anyhow, downloading the Flux model "https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF" now. Start tomorrow. What I am hoping for, AI image editing and possibly better image generation, however Flux are models with guardrails, so both have their purposes. Oh, also I will test/fix the installer, hoping it works currently. 
```
- Attempting Flux Klein with flux encoder, this involves new panel for flux, but also hardcoding of common values to known good settings for most in order to decomplicate what would otherwise be the Comfy-UI, hence with reduced inputs we would have a functional image generation and editing, where the user may associate an image with their prompt, and then for example use the flux image to text encoder with the flux model and a prompt from the user to create filtered image, ie "Change the background to beaches, keep pose and central character, but put light brown shorts and t-shirt on the character".
- Done all noted improvements, though possibly will still noticinmg things that can be improved, see recent releases.
- Add AI edit image feature, ie outpaint, etc. Requires Flux-Klein support because Z-image-turbo only supports generation and not editing. The UI woudl also require a rethink, ie what are the options and how am I going to fit them in. It would also need a Qwen 3 Flux Klein Encoder. If you would like to see the project reach "add image editing via SD Klein and Qwen3 for SD", then please support via sponsor/donate via kofi/patreon, otherwise development may grind to a halt here at some unexpected point.

### Structure:
- Current plan for scripts is...
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
