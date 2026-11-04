# Skribl

Handwritten digit and character recognition using convolutional neural networks.

**Author: Hamza**  
**CodeAlpha — Machine Learning Task 3**

## Features

- Separate digit and letter pages.
- Recognizes digits 0–9 and case-sensitive English letters A–Z / a–z.
- Draw handwriting or upload a PNG, JPEG, or WebP image.
- Detects separated characters on one horizontal line.
- Shows the predicted answer and top three guesses for each character.
- Saves confirmed answers and corrections for background learning.
- Generates training curves, confusion matrices, and measured evaluation results.

## Structure

| File | Purpose |
| --- | --- |
| backend.py | Flask API and local server |
| frontend/index.html | Landing page and recognition interface |
| frontend/style.css | Styling and animation |
| frontend/script.js | Drawing, uploads, predictions, and confirmation |
| frontend/logo.svg | App logo |
| train.py | Download datasets, train models, and generate results |
| model.py | CNN architecture and checkpoint loading |
| preprocessing.py | Image normalization and character splitting |
| feedback.py | Confirmed examples and background retraining |
| requirements.txt | Python dependencies |

Runtime folders are created automatically: data, models, reports, and instance.

## Setup

Extract the ZIP and open the project folder in VS Code. In its PowerShell terminal, run:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

Use "py -3.13" instead if Python 3.13 is installed. The dependency pair was checked on Python 3.13; Python 3.12 is supported by the pinned packages.

## Train

~~~powershell
.\.venv\Scripts\python.exe train.py --mode both --epochs 5
~~~

Initial downloads require internet. EMNIST is large; allow several GB of disk space. Training uses CPU, so the older NVIDIA card is not required. Keep the terminal open until training completes.

For digits first, use "--mode digits". Train letters later with "--mode letters".

## Run

~~~powershell
.\.venv\Scripts\python.exe backend.py
~~~

Open http://127.0.0.1:5000 in your browser. Keep the terminal open while using the app.

If your virtual environment is activated, these shorter commands work:

~~~powershell
pip install -r requirements.txt
python train.py --mode both --epochs 5
python backend.py
~~~

On later runs, only start backend.py. Setup and initial training do not need repeating.

## Updating an existing installation

This ZIP contains updated source files only. Preserve your existing models, data, and instance folders and your virtual environment. The checkpoint format is compatible with the earlier version.

Copy the updated source into your project, stop the old server with Ctrl+C, then start backend.py. Refresh the browser with Ctrl+F5. The old app.py and numbered batch files are no longer needed.

## How to use

Choose Digits or Letters. Draw or upload one line of separated, non-cursive characters. Do not mix digits and letters. Maximum: 20 characters.

Click "Read my handwriting". The main answer and top three guesses appear. For multiple characters, select each character to inspect its guesses.

Confirm the answer or enter a correction. Confirmation automatically saves the example for learning; there is no separate checkbox. A short notice explains this in the interface.

## Learning from feedback

Each installation stores its own confirmed examples locally. A hosted installation shares its model among its visitors; a GitHub clone learns only from its own examples.

The worker checks every minute. It considers retraining after 25 new unique examples per mode, with at least five training and five validation feedback examples. This means retraining may wait for more examples.

About 20% of feedback is reserved for validation using stable image hashes. Training replays original training samples to reduce forgetting. A candidate replaces the active model only if neither validation accuracy falls and at least one improves. The previous checkpoint is retained.

Confirmation saves the example immediately; model training happens in batches. The model continues serving predictions while a candidate trains. Duplicate normalized images keep their first confirmed label.

Temporary prediction crops expire after one hour and are cleaned by the running worker or a new prediction. Confirmed examples persist in instance/feedback.sqlite3.

Set SKRIBL_AUTO_RETRAIN=0 to disable background training. Confirmed examples are still stored. Stop the server before manually replacing models or running full initial training.

## Training details

Digits use MNIST. Letters use EMNIST ByClass filtered to 52 letter classes, preserving uppercase/lowercase. EMNIST images are transposed upright before shared 28×28 normalization.

The default experiment uses up to 2,000 training examples per class, a stratified 90/10 train/validation split, five epochs, AdamW, cross-entropy, and seed 42. The best validation checkpoint is selected before official test evaluation.

The default official test evaluation uses a seeded sample of up to 200 examples per class. Exact counts are recorded in reports. Test examples never enter training or checkpoint selection.

For all available examples:

~~~powershell
.\.venv\Scripts\python.exe train.py --mode both --epochs 10 --per-class 0 --test-per-class 0
~~~

Inspect measured results before claiming accuracy. More training does not guarantee better performance.

## Limitations

Auto-detection works best with clear gaps. Touching characters, shadows, disconnected strokes, cursive, and multiple lines can produce incorrect splits. An expected count checks the result; it does not force arbitrary splits.

Similar shapes such as C/c or I/l can be ambiguous. Confidence is an uncalibrated model score, not a guarantee. Unsupported inputs may still receive a prediction. Mixed alphanumeric input and cursive recognition are future extensions.

The source ZIP does not include trained weights, datasets, private session keys, or feedback. Predictions require initial training or your existing compatible checkpoints.

## GitHub

Suggested repository: CodeAlpha_HandwrittenCharacterRecognition.

Upload the source and keep .gitignore included. Exclude .venv, data, instance, and private feedback. Checkpoints are ignored by default. Share models separately with their evaluation details and matching replay checkpoints if learning is needed.

## Hosting

Run one application process with persistent writable models and instance folders. Start through "python backend.py" to enable the background worker. Configure HTTPS, a strong SKRIBL_SECRET_KEY, suitable access controls, retention controls, and request limits before public hosting. Localhost is the default. SKRIBL_HOST=0.0.0.0 is available for deliberate hosting configuration.

## References

- MNIST: https://yann.lecun.com/exdb/mnist/
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- EMNIST paper: https://arxiv.org/abs/1702.05373
- PyTorch: https://pytorch.org/

Dataset rights remain with their providers.
