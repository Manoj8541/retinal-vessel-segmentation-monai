# retinal-vessel-segmentation-monai
## Retinal Vessel Segmentation — MONAI Hackathon

---

## 🔷 Section 1 — Dataset & Preprocessing

**Q: Why did you choose the DRIVE dataset?**  
DRIVE is the most widely used benchmark for retinal vessel segmentation with 40 annotated fundus images. It has a well-defined train/test split, expert-annotated ground truth masks and decades of literature to compare against — making evaluation objective and reproducible.

**Q: Why resize to 512×512?**  
Original DRIVE images are 565×584. Resizing to 512×512 standardises spatial resolution for batch processing, fits within GPU memory constraints and keeps resolution high enough to preserve thin capillary details that would be lost at lower resolutions like 256×256.

**Q: Why use NormalizeIntensityd per channel?**  
Fundus images have strong color channel variation due to different retinal pigmentation levels across patients. Per-channel normalisation using mean and standard deviation computed per image removes this bias, giving the model consistent intensity distributions regardless of camera settings or patient demographics.

**Q: What augmentations did you use and why those specifically?**  
- RandFlip and RandRotate90 — vessels appear in all orientations so the model must be rotation invariant
- RandZoom (0.9–1.1×) — simulates natural variation in image field of view and camera distance
- RandGaussianNoise — mimics image acquisition noise from different fundus camera hardware
- All augmentations applied jointly to image and mask to maintain spatial correspondence

**Q: How did you avoid overfitting with augmentation?**  
Conservative augmentation parameters were chosen — zoom limited to ±10%, noise std kept low. Aggressive augmentation like elastic deformation can distort vessel connectivity which would confuse the model. The goal is diversity not destruction.

---

## 🔷 Section 2 — Model Architecture

**Q: Explain UNet architecture briefly.**  
UNet has an encoder that progressively downsamples the input through convolution and pooling, capturing increasingly abstract features. A symmetric decoder upsamples back to original resolution. Skip connections directly copy encoder feature maps to corresponding decoder levels, preserving spatial detail lost during downsampling. The final output is a binary vessel probability map.

**Q: Why UNet for medical image segmentation specifically?**  
UNet was designed for biomedical segmentation with limited data. Skip connections preserve fine structural detail like thin vessel segments. It works well on small datasets — 20 training images in DRIVE — because the encoder-decoder weight sharing is efficient.

**Q: What do attention gates in Attention UNet do?**  
Attention gates learn to assign weights between 0 and 1 to each spatial location in the skip connection feature map before it reaches the decoder. Locations relevant to vessel prediction get high weights, background regions get suppressed. This theoretically helps with the severe class imbalance in DRIVE where vessels are less than 10 percent of pixels.

**Q: Then why did Attention UNet perform worse than Baseline UNet?**  
With only 20 training images, the attention gate parameters did not have enough data to learn meaningful vessel-specific attention. The gates likely learned noisy weights that degraded rather than improved skip connection quality. More training data or transfer learning would likely flip this result.

**Q: What is ASPP in DeepLabV3+ and why does it help?**  
Atrous Spatial Pyramid Pooling applies dilated convolutions with multiple dilation rates (6, 12, 18) in parallel. Each rate captures features at a different scale — small dilation captures fine capillary detail, large dilation captures thick artery context. The outputs are concatenated giving the model a multi-scale representation in a single layer without increasing resolution requirements.

**Q: Why did DeepLabV3+ underperform?**  
DeepLabV3+ with ResNet50 has 26.7 million parameters — 40 times more than Baseline UNet. Fine-tuning this many parameters on only 20 training images leads to severe overfitting. The model memorises training samples and fails to generalise. A much larger dataset or stronger regularisation would be needed for DeepLabV3+ to outperform UNet here.

---

## 🔷 Section 3 — Loss Functions & Training

**Q: Why Dice Loss for vessel segmentation?**  
Dice Loss directly optimises the overlap between predicted and ground-truth masks. It handles class imbalance inherently — because vessel pixels are rare, cross-entropy would be dominated by correct background predictions and never properly train vessel detection. Dice Loss treats both classes equally regardless of frequency.

**Q: What is the combined Dice + BCE loss and why use it?**  
Total Loss = 0.6 × Dice + 0.4 × BCE. Dice Loss optimises global overlap but provides sparse gradient signal in early training when predictions are mostly wrong. BCE provides dense pixel-level gradients that stabilise early training. The combination benefits from both — BCE drives early convergence, Dice drives final precision.

**Q: Why CosineAnnealingLR?**  
Cosine annealing decays the learning rate following a cosine curve from the initial value to near zero. This allows aggressive learning in early epochs and fine-grained weight updates in later epochs. It outperforms step decay for segmentation tasks because the smooth decay prevents oscillation near the loss minimum.

**Q: How does early stopping prevent overfitting?**  
If validation Dice does not improve for 10 consecutive epochs, training stops and the best checkpoint is restored. This prevents the model from continuing to memorise training samples after generalisation has peaked, saving both time and model quality.

---

## 🔷 Section 4 — Post-Processing

**Q: Explain each post-processing step.**  
- Morphological Closing fills small gaps within predicted vessel segments using a disk structuring element of radius 1. A vessel broken into two nearby segments gets reconnected.
- Connected Component Filtering removes isolated blobs smaller than 50 pixels. These are statistically unlikely to be real vessels — they are noise from imperfect sigmoid thresholding.
- Morphological Opening smooths rough boundaries and removes single-pixel spurs at vessel endpoints created by the closing operation.

**Q: Why did post-processing slightly reduce Dice for Baseline UNet?**  
The Baseline UNet already produces clean predictions with well-connected vessels. Morphological operations — designed to fix noisy predictions — end up eroding some true thin vessel segments that are correctly predicted. The cure is slightly worse than the disease for a strong model. For Attention UNet with noisier predictions the cleanup value is higher.

**Q: How do you prove post-processing helped?**  
Compare per-sample Dice scores before and after across all 20 test images. For Attention UNet the mean HD95 dropped from 49.15 to 38.59, indicating that boundary accuracy improved even though Dice changed marginally. HD95 is more sensitive to post-processing effects than Dice.

---

## 🔷 Section 5 — Evaluation Metrics

**Q: What does Dice score measure and what is its formula?**  
Dice = 2 × |Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|). It measures the overlap between predicted and true vessel masks. A score of 1 means perfect overlap, 0 means no overlap. It is the primary metric for segmentation tasks.

**Q: Why report Hausdorff95 distance?**  
Dice measures overall overlap but ignores boundary accuracy. A model could have good Dice but jagged, clinically unusable vessel boundaries. HD95 measures the 95th percentile of the directed Hausdorff distance between predicted and ground truth boundaries in pixels. Lower is better. It reveals boundary quality that Dice cannot.

**Q: What does high Recall but low Precision mean for Attention UNet?**  
Recall of 0.7074 means it detects 70 percent of true vessel pixels. Precision of 0.3374 means only 34 percent of its vessel predictions are actually vessels. The model is over-predicting — flagging many background pixels as vessels. It is sensitive but not specific.

**Q: Why is PR-AUC more informative than ROC-AUC here?**  
ROC-AUC accounts for true negatives — but background pixels vastly outnumber vessel pixels, so correctly predicting background inflates ROC-AUC artificially. PR-AUC only uses Precision and Recall — both computed only from positive vessel predictions. It is a stricter metric for class-imbalanced datasets like DRIVE.

**Q: What does the radar chart tell you at a glance?**  
It shows all seven metrics simultaneously for all four model conditions. The larger the enclosed area, the better the overall model. Baseline UNet Raw encloses the largest area confirming it is the best overall model. Attention UNet's shape is skewed toward Recall but shrinks in Precision, visually explaining the false-positive tendency.

---

## 🔷 Section 6 — Innovations

**Q: Explain Grad-CAM in simple terms.**  
Grad-CAM asks: for this prediction, which feature map channels mattered most? It computes gradients of the output with respect to the final convolutional layer, pools them spatially and uses them to weight the feature map activations. The result is a heatmap showing which spatial regions most influenced the prediction.

**Q: Why is Grad-CAM clinically important?**  
A model can achieve high Dice by learning spurious correlations — camera artifacts, bright optic disc regions. Grad-CAM lets a clinician visually verify the model is attending to actual vessel structures rather than image artifacts. It builds trust necessary for clinical deployment.

**Q: What is Monte Carlo Dropout and how is it different from standard dropout?**  
Standard dropout is active only during training and disabled during inference for deterministic predictions. MC Dropout keeps dropout active during inference and runs multiple forward passes. The variation across passes reflects model uncertainty — pixels where the model is consistently confident show low variance, uncertain regions show high variance.

**Q: What did uncertainty maps reveal in your results?**  
Vessel boundaries, thin capillary tips and low-contrast peripheral regions showed the highest uncertainty. This is clinically meaningful — these are exactly the regions where human expert disagreement is also highest. The uncertainty map can guide radiologists to review specific areas rather than checking the entire image.

**Q: What is TTA and what is its benefit?**  
Test-Time Augmentation runs the model on multiple augmented versions of the same test image and averages the output probability maps. For DRIVE, 8 views (flips, rotations, transpose) were used. Averaging reduces prediction variance caused by orientation bias — the model might be slightly more confident about vessels oriented horizontally due to training distribution. TTA corrects this for free.

---

## 🔷 Section 7 — Gradio App

**Q: Why Gradio over Streamlit?**  
Gradio is purpose-built for ML model demos with minimal boilerplate. It generates a public shareable link automatically with share=True — critical for hackathon live demonstrations where judges need remote access. Streamlit requires additional deployment configuration for public access.

**Q: What does your Gradio app allow users to do?**  
Upload any fundus image, select Baseline or Attention UNet, toggle post-processing on or off with adjustable closing radius and minimum component size, adjust overlay transparency and view six output panels simultaneously — original image, ground truth, raw prediction, post-processed mask, probability heatmap and vessel overlay — alongside the computed Dice score.

---

## 🔷 Section 8 — General & Critical Thinking

**Q: What would you do differently with more time?**  
- Train on larger augmented dataset using elastic deformation and color jitter
- Implement SwinUNETR which uses vision transformers for global context
- Use ensemble averaging across Baseline UNet and Attention UNet predictions
- Implement uncertainty-guided active learning to select the most informative samples for annotation
- Deploy the Gradio app permanently on Hugging Face Spaces

**Q: Why does Baseline UNet beat Attention UNet despite Attention UNet being architecturally more advanced?**  
Architectural sophistication requires data to justify it. Attention gates add parameters that must be learned — with 20 training images there is insufficient signal to train them meaningfully. The additional parameters become a liability rather than an asset. This is a common pattern in medical imaging where data is scarce: simpler models regularise better.

**Q: If you had BRATS data what would change in your pipeline?**  
BRATS is 3D MRI requiring spatial_dims=3 in MONAI UNet. Preprocessing would add Spacingd for voxel spacing normalisation and Orientationd for canonical orientation. Sliding window inference would replace batch inference to handle the large 3D volume. Loss function would need multi-class support for whole tumor, tumor core and enhancing tumor subregions.

**Q: How would you deploy this in a real hospital?**  
Containerise with Docker, deploy on a hospital GPU server, integrate DICOM input support using pydicom, add authentication and audit logging for HIPAA compliance, implement a feedback loop where radiologists can correct predictions to improve the model over time and generate structured reports alongside visual overlays.

---
