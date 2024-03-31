from PIL import Image
import os

def resize_image(input_path, output_path):
    """Resizes an image to make dimensions divisible by 8."""
    with Image.open(input_path) as img:
        w, h = img.size
        w_new = (w // 8) * 8  # Make width divisible by 8
        h_new = (h // 8) * 8  # Make height divisible by 8
        img_resized = img.resize((w_new, h_new), Image.ANTIALIAS)
        img_resized.save(output_path)
        print(f"Resized image saved to {output_path}")

def convert_to_grayscale(input_path, output_path):
    """Converts an image to grayscale."""
    with Image.open(input_path) as img:
        img_gray = img.convert('L')  # Convert to grayscale
        img_gray.save(output_path)
        print(f"Grayscale image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    import os

    image_path = os.path.join(os.environ['DATA_DIR'], "TarTanAir-P002-Forest/image_left/000005_left.png")

    output_path_resized = "resized_image.png"
    output_path_gray = "grayscale_image_5.png"
    
    image = Image.open(image_path)
    print(image.size)  # Outputs (width, height)
    
    # resize_image(image_path, output_path_resized)
    # convert_to_grayscale(output_path_resized, output_path_gray)

    convert_to_grayscale(image_path, output_path_gray)
