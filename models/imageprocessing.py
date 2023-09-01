import cv2

def process_book_page(image_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    
    # Làm nét ảnh chữ
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return blurred_image

# Thay đổi đường dẫn tới ảnh của bạn
input_image_path = './uploads/test.jpg'

# Xử lí ảnh và lưu kết quả vào tệp mới
output_image = process_book_page(input_image_path)
cv2.imwrite('processed_image.jpg', output_image)
