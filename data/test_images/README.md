# Test images

Drop your PlantVillage-style test photos into the per-class subfolders here.
The folder name **is** the ground-truth label, so it must match a class in
`model/class_names.json` exactly (already created for all 38 classes).

```
data/test_images/
├── Tomato___Late_blight/     img1.jpg  img2.jpg ...
├── Apple___Apple_scab/       ...
└── ...
```

- Any number of images per class (0 is fine — empty classes are skipped).
- Formats: .jpg, .jpeg, .png, .bmp, .webp.
- Images are auto-resized to 224×224 and ImageNet-normalized at inference.

Then run the real-image evaluation:

```bash
python agents/main.py --mode ensemble --retrieval on
```

`TEST_IMAGE_DIR` in `.env` points here (`data/test_images`). Change that one
value to evaluate a different image set.
