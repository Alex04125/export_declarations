import openai
import os
import fitz
import os
from pdf2image import convert_from_path
import cv2
import pytesseract


def convert_pdf_id_data_folder_into_jpgs_in_data_parsed_pdf(
    pdf_to_parse_name="EX 1 - 984908_2.PDF",
    use_cache=True,
    read_only=False,
    user="-",
):
    pdf_folder = pdf_to_parse_name.lower().split(".")[0]
    cache_folder = f"data/{pdf_folder}"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)
    at_least_one_jpg_in_the_folder = [
        file for file in os.listdir(cache_folder) if "jpg" in file.lower()
    ]
    if (at_least_one_jpg_in_the_folder and use_cache) or read_only:
        pass  # the data is already parsed or in read only mode
    else:
        # Read the PDF:
        windows_poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
        if os.path.exists(windows_poppler_path):
            poppler_path = windows_poppler_path
        else:
            poppler_path = r"/usr/bin"
        pages = convert_from_path(
            f"data/{pdf_to_parse_name}", poppler_path=poppler_path
        )
        # Add the create timestamp and user file:
        # _set_file_process_state(
        #     pdf_to_parse_name, status="Converting PDF to JPGs...", read_only=read_only
        # )
        # add_create_date_txt_file(pdf_to_parse_name, read_only=read_only)
        # _set_user_that_created_the_file(pdf_to_parse_name, user=user)
        # Save the pages as JPGs:
        for i, page in enumerate(pages, start=1):
            print(
                f"- Saving PDF page {i+1}/{len(pages)} to JPG...             ",
                end="\r",
                flush=True,
            )
            page.save(f"{cache_folder}/output_page_{i}.jpg", "JPEG")
        deskew_and_center_images_in_folder(cache_folder)
        print("- PDF -> JPGs DONE!                ")


def deskew_and_center_images_in_folder(folder):
    """Processes all images in a folder and saves corrected images to an output folder."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if (
            os.path.isfile(file_path)
            and os.path.splitext(filename)[1].lower() in valid_extensions
        ):
            output_path = os.path.join(folder, filename)
            deskew_and_center_text(file_path, output_path)
        else:
            print(f"Skipping non-image file: {filename}")


def is_page_empty(image):
    """Detects if the page is empty by checking for text presence."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).strip()
    return len(text) == 0


def auto_rotate_image(image_path, output_path):
    """Detects text orientation, rotates image, and skips empty pages."""
    tesseract_path = os.getenv("Tesseract")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ ERROR: Could not load image {image_path}")
        return False, False

    if is_page_empty(image):
        print(f"⚠️ Skipping blank page: {image_path}")
        return False, False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    angle = osd["rotate"]
    print(f"🔄 Detected rotation angle: {angle}°")

    if angle == 0:
        return True, False  # Image is already correctly oriented

    if angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = image

    success = cv2.imwrite(output_path, rotated)
    if success:
        print(f"✅ Image successfully saved at: {output_path}")
        return True, True  # Image was rotated
    else:
        print(f"❌ ERROR: Could not save image at {output_path}")
        return False, False


def detect_skew_angle(image):
    """Detects skew angle using Hough Transform, with fallback to Tesseract OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect skew angle using Hough Transform
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # Keep angles within a natural range
                angles.append(angle)

    # Use median angle if valid angles exist
    hough_angle = np.median(angles) if angles else 0

    # OCR-based validation (fallback)
    osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    ocr_angle = osd["rotate"]

    # Normalize OCR angle to avoid 90, 180, 270 degrees
    if ocr_angle in [90, 180, 270]:
        ocr_angle = 0  # Ignore these rotations

    # Confidence check: If both methods give similar results, use it
    if abs(hough_angle - ocr_angle) <= 7:
        return hough_angle

    # If difference is large, trust OCR more but avoid 90-degree multiples
    return ocr_angle


def deskew_and_center_text(image_path, output_path, threshold=0):
    """Aligns the text in an image by deskewing only if the skew is significant."""

    # Auto-rotate the image first
    # rotated_path = "temp_rotated.jpg"
    rotation_success, image_rotated = auto_rotate_image(image_path, output_path)

    if not rotation_success:
        return False

    # If the image was rotated, skip further processing
    if image_rotated:
        print("✅ Image was rotated, skipping deskewing.")
        return True

    # Load the rotated (or original) image
    image = cv2.imread(output_path)
    if image is None:
        print(f"❌ ERROR: Could not load rotated image {output_path}")
        return False

    # Detect the skew angle
    angle = detect_skew_angle(image)
    print(f"🔄 Detected skew angle: {angle}°")

    # Only rotate if the skew is significant
    if abs(angle) > threshold:
        print("✅ Applying deskew correction...")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    else:
        print("✅ Skew is minimal or unreliable, no correction applied.")

    # Save the final image
    success = cv2.imwrite(output_path, image)

    if success:
        print(f"✅ Image saved at: {output_path}")
        return True
    else:
        print(f"❌ ERROR: Could not save image at {output_path}")
        return False


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def parse_customs_document(pdf_text, prompt, model="gpt-4o-mini"):
    # Load API key from environment variable
    api_key = os.getenv("CHAT_GPT_API")
    if not api_key:
        raise ValueError(
            "API key not found. Make sure CHAT_GPT_API is set in your environment variables."
        )

    client = openai.OpenAI(api_key=api_key)

    # Trim if needed (can make this dynamic if you're chunking later)
    pdf_chunk = pdf_text[:10000]

    # Send to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": pdf_chunk},
        ],
    )

    return response.choices[0].message.content


# Define the customs parsing prompt
customs_prompt = """
You are a customs document parser. Extract all relevant export declaration data from the following PDF or OCR'd text.

Return the following key information in a structured JSON or tabular format:
1. MRN (Movement Reference Number)
2. Exporter/Sender
   - Company name
   - Address
   - Country
   - VAT/ID (if available)
3. Recipient/Consignee
   - Company name
   - Address
   - Country
4. Customs Details
   - Date of acceptance
   - Export and exit customs offices
   - Incoterms
   - Country of export
   - Country of destination
   - Means of transport (mode, registration number if available)
5. Invoice & Shipment
   - Currency
   - Total invoice value
   - Gross and net weight
   - Number of packages
6. Goods (repeatable list)
   For each item:
   - Line number (if applicable)
   - Description
   - HS code (tariff code)
   - Quantity (unit count or declared value)
   - Gross weight
   - Net weight
   - Statistical value / Invoice value
7. Other Info (optional but useful)
   - IUT / LRN codes
   - Document references (Y903, N380, etc.)
   - Any mentions of pallet/packaging
   - Country of origin of goods

Be lenient to variations in language (e.g., English, Italian, German, Romanian) and handle minor OCR noise.
"""
