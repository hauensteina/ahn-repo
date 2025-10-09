import qrcode

# Your PayPal or other payment URL
url = "https://www.paypal.com/ncp/payment/JH9F9ELKNLP4Y"

# Generate QR code
qr = qrcode.QRCode(
    version=1,                # controls size; 1 = smallest
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image
img = qr.make_image(fill_color="black", back_color="white")

# Save it
img.save("link2qr.png")

print("QR code saved as link2qr.png")

