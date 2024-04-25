import os
from PIL import Image
import pytesseract 
import cv2
import numpy as np

def resize_to_300_dpi(image_path):
    # Open the image using PIL
    img = Image.open(image_path)
    
    # Calculate the new size for 300 DPI
    width_inch, height_inch = img.size
    new_width = int(width_inch * 300 / img.info['dpi'][0])
    new_height = int(height_inch * 300 / img.info['dpi'][1])
    
    # Resize the image to 300 DPI
    resized_img = img.resize((new_width, new_height))
    return resized_img

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image
        
def ocr_task():

    # Get the current directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the data folder
    data_folder = os.path.join(script_dir, '../data')

    # Create output.txt file in the parent folder
    output_file = os.path.join(os.path.dirname(script_dir), 'output.txt')
    file_list = []

    # Loop through all files in the data folder
    for file in os.listdir(data_folder):
        if file.endswith("_resized.png") or file.endswith("_processed.jpg"):
            continue
        
        file_path = os.path.join(data_folder, file)

        # Check if the resized image already exists
        resized_img_path = os.path.join(data_folder, f"{file}_resized.png")
        if not os.path.exists(resized_img_path):
            # Resize the image to 300 DPI
            resized_img = resize_to_300_dpi(file_path)
            resized_img.save(resized_img_path, dpi=(300, 300))

        # Check if the processed image already exists
        processed_img_path = os.path.join(data_folder, f"{file}_processed.jpg")
        if not os.path.exists(processed_img_path):
            # Process the image to remove noise
            cv_img = cv2.imread(resized_img_path)
            no_noise = noise_removal(cv_img)
            cv2.imwrite(processed_img_path, no_noise)

        # Perform OCR on the processed image
        im = Image.open(os.path.join(data_folder, f"{file}_processed.jpg"))
        ocr = pytesseract.image_to_string(im, config='--psm 10 --oem 3 -c tessedit_char_whitelist=-.0123456789')   
        file_list.append((file, ocr))  # Append file name and OCR to file_list

    # Sort file_list by file name
    file_list.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))

    with open(output_file, 'w') as f:
        for file, ocr in file_list:
            f.write(f"{file}: {ocr}\n")  # Write file name and OCR to output.txt

def delete_processed_images():
        # Get the current directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the relative path to the data folder
        data_folder = os.path.join(script_dir, '../data')
        
        # Loop through all files in the data folder
        for file in os.listdir(data_folder):
            if file.endswith("_processed.jpg") or file.endswith("_resized.png"):
                file_path = os.path.join(data_folder, file)
                os.remove(file_path)

if __name__ == "__main__": 
    ocr_task()
    delete_processed_images() #can be commented out to keep the processed images
    