from PIL import ImageGrab, Image
import datetime
import os

def get_screenshot(save_dir="screenshots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot = ImageGrab.grab()
        
    file_path = os.path.join(save_dir, f"screenshot_{timestamp}.png")
    screenshot.save(file_path)
    return file_path



if __name__ == "__main__":
    screenshot_path = get_screenshot()
    print(f"Screenshot saved at: {screenshot_path}")