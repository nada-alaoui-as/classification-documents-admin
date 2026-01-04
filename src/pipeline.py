from pathlib import Path
from src.cv_module.gabarit_detector import GabaritDetector


def test_all_images(processed_dir: Path):
    detector = GabaritDetector()

    image_extensions = {".jpg", ".jpeg", ".png"}

    for image_path in processed_dir.rglob("*"):
        if image_path.suffix.lower() in image_extensions:
            category, conf, scores = detector.predict_from_image(str(image_path))

            print("\n📄 Image :", image_path)
            print(f"➡️ Catégorie : {category}")
            print(f"➡️ Confiance : {conf:.2%}")
            print(f"➡️ Scores : {scores}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    test_all_images(PROCESSED_DIR)
