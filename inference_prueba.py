# infer_local_ckpt_fallback.py
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

CKPT_LOCAL = "/media/rcasal/PortableSSD/checkpoints_rt_detrv2/checkpoint-58208"
IMG  = "image4.jpg"
THR  = 0.03
OUT  = "out_pil4.jpg"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RTDetrV2ForObjectDetection.from_pretrained(CKPT_LOCAL).to(device).eval()
    # Fallback: si no hay preprocessor_config en el checkpoint local, usa el del modelo base
    
    processor = AutoImageProcessor.from_pretrained(CKPT_LOCAL)
    
        

    img = Image.open(IMG).convert("RGB")
    H, W = img.height, img.width
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(H, W)], device=device),
            threshold=THR
        )[0]
    print(results)
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except: font = None

    id2label = model.config.id2label
    for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = id2label[int(label_id)]
        txt = f"{cls} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
        tw = draw.textlength(txt, font=font) if hasattr(draw, "textlength") else 0
        th = 14
        draw.rectangle([x1, y1-(th+6), x1+(tw+8), y1], fill=(0,255,0))
        draw.text((x1+4, y1-(th+5)), txt, fill=(0,0,0), font=font)

    img.save(OUT)
    print(f"Saved: {OUT}")

if __name__ == "__main__":
    main()
