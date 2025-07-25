# TransformersSharp

![Logo](media/logo.png)

A little wrapper for Hugging Face Transformers in C#. This is not a comprehensive 1:1 mapping of the whole HuggingFace [transformers](https://pypi.org/transformers) package, because the API is enormous.

If you need a specific feature, toggle or pipeline API clone this repo and make adjustments.

This project was created using [CSnakes](https://github.com/tonybaloney/CSnakes) and will fetch Python, PyTorch, and Hugging Face Transformers automatically, so you don't need to install them manually.

## Features

- Tokenizer API based on [`PreTrainedTokenizerBase`](https://huggingface.co/docs/transformers/v4.51.3/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)
- Tokenizer shim to the [Microsoft.ML.Tokenizers](https://learn.microsoft.com/dotnet/api/microsoft.ml.tokenizers.tokenizer?view=ml-dotnet-preview) base class
- [Generic Pipeline Factory](pipelines/index.md)
- [Text Generation Pipeline (`TextGenerationPipeline`)](pipelines/text_generation.md)
- [Text Classification Pipeline (`TextClassificationPipeline`)](pipelines/text_classification.md)
- [Image Classification Pipeline(`ImageClassificationPipeline`)](pipelines/image_classification.md)
- [Object Detection Pipeline (`ObjectDetectionPipeline`)](pipelines/object_detection.md)
- [Text to Audio Pipeline (`TextToAudioPipeline`)](pipelines/text_to_audio.md)
- [Text to Image Pipeline (`TextToImagePipeline`)](pipelines/text_to_image.md)
- [Automatic Speech Recognition (`AutomaticSpeechRecognitionPipeline`)](pipelines/auto_speech_recognition.md)
- [Sentence Transformers (Embeddings)](sentence_transformers.md)

## Project Structure

- **TransformersSharp.sln** - Main solution containing the TransformersSharp library and unit tests
- **DemoConsole/** - Console applications and demos demonstrating library features
  - **DemoConsole.sln** - Solution file for all demo projects
  - **ConsoleApp1-4** - Various example applications
  - **TransformersSharp.EnvironmentTest** - Environment validation and testing

## Architecture

- [Python Module Architecture](python_modules.md) - Modular Python wrapper design
- [CUDA Fix Summary](CUDA_FIX_SUMMARY.md) - Device compatibility and performance improvements