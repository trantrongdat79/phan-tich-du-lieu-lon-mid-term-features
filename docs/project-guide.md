EE104 Features — Group Presentation Plan
1. Expected Outcome
A 10–20 minute presentation replicating the EE104 Features lecture by Lall & Boyd (Stanford). The deliverables are:

A complete slide deck covering the full lecture content
A live/pre-run demo notebook replicating key visual results from the slides
The demo runs alongside or after the slides as a ~3–5 min segment

Self-set scope: The presentation should be faithful to the original material, visually clean, and internally consistent. The demo focuses on recreating the plots and transforms from the slides — not full model training.
Success criteria: An audience unfamiliar with the material can follow the narrative from raw data → embeddings → feature engineering, and the demo visually reinforces the concepts.

2. Tasks Per Person
Person A — Slides: Raw Data to Vectors
Core message to deliver: Raw heterogeneous data must be transformed into comparable numeric vectors before any learning can happen.
Slides to build:

Title slide & agenda
Raw data, records, fields
Feature maps & embeddings (φ, ψ)
Faithful embeddings
Simple embeddings: Boolean, real, color
Categorical data & one-hot encoding (full and reduced)
Ordinal data
Standardization / z-scoring
Log transform + house price example

Responsibilities:

Set the slide template (font, colors, layout) — Person B must use the same
Define key terms clearly: record, field, embedding, feature map
End your last slide with a transition sentence: "Now that we have vectors, how do we make them better?"
Send template to Person B within the first 2 hours


Person B — Slides: Feature Engineering
Core message to deliver: Once we have feature vectors, we can deliberately transform them to improve model performance.
Slides to build:

Feature engineering concept & motivation
Types of transforms: gamma, clipping, powers
Positive/negative split
Interactions & product features
Quantizing / binning
Feature engineering pipeline
Automatic features (word2vec, vgg16) — brief conceptual mention only
Summary slide

Responsibilities:

Use the template from Person A — do not deviate
Open your first slide with a callback to Person A's transition sentence
Do not redefine terms Person A already defined
Send your draft slides to Person A for a quick read-through before final assembly


Person C — Demo Notebook
Core message to deliver: A visual, runnable demonstration of the key transforms covered in the slides.
Targets to replicate:

Z-scoring on a simple numeric dataset
Gamma-transform plot (γ = 0.5 and γ = 1.5)
Clipping / winsorizing plot
Powers transform plot (degree 1, 2, 3)
Positive/negative part split plot
Day-of-week circular embedding plot
Word2vec — load pretrained, show 3–5 word distance examples (skip if environment setup takes too long)

Responsibilities:

Use Python + matplotlib (+ gensim if doing word2vec)
Each plot should have a title matching the slide it corresponds to
Notebook must run clean from top to bottom — no broken cells
Prepare a short verbal explanation (2–3 sentences) for each plot
Skip vgg16 entirely — not worth the setup time


3. Expected Timeline
Assuming you start now (T = 0), deadline at T = 24 hours.
TimePerson APerson BPerson CT+0Begin slides Part 1Read through the PDF to understand Part 2 scopeStart demo environment setupT+2Send slide template to Person BBegin slides Part 2 using templateComplete plots 1–3 (z-score, gamma, clipping)T+6Complete draft of Part 1 slidesComplete draft of Part 2 slidesComplete plots 4–6 (powers, pos/neg, day-of-week)T+8Sync checkpoint — all 3 swap work, read each other's output, flag issuesT+10Revise Part 1 based on feedbackRevise Part 2 based on feedbackAttempt word2vec (if time allows)T+14Merge slides into one deckMerge slides into one deckFinalize notebook, ensure clean runT+16Full dry run — all 3 together, check transitions and demo flowT+18Final fixes to slidesFinal fixes to slidesFinal fixes to notebookT+20Buffer — handle anything that broke during dry runT+24Deadline

Critical warnings:

The T+8 sync checkpoint is non-negotiable — this is where disconnection gets caught
Person C should not wait for slides to be finished before starting — work in parallel from T+0
If word2vec setup exceeds 1 hour, drop it and add one more transform plot instead
