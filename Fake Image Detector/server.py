from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# Define your FastAPI app
app = FastAPI()

# Load your PyTorch model
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Preprocess the image
def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Define the prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        result = "fake" if prediction == 0 else "real"

        # Return the prediction as HTML response
        result = f"""
<html>
    <head>
        <title>Prediction Result</title>
    </head>
    <body style="text-align: center; font-family: Arial, sans-serif; margin: 5rem; padding: 20px;">
    <h1>Prediction Result</h1>
    <p><b>Prediction:</b> {result}</p>
    <a href="/" style="display: inline-block; padding: 10px 20px; background-color: #007bff; color: #fff; text-decoration: none; border-radius: 5px;">Go   
 back</a>
</body>
</html>
"""
        return HTMLResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Define the HTML form endpoint
@app.get("/")
def main():
   content = """
  <html>
    <head>
        <title>Image Upload</title>
        <style>
            body {
                font-family: 'Poppins', Arial, sans-serif;
                background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
                margin: 10rem;
                padding: 0;
                height: 100vh;
                display: flex;
                # justify-content: center;
                align-items: center;
                text-align: center;
                color: #333;
            flex-direction: column;
            align-items: center;
            }

            h1 {
                font-size: 2.5rem;
                margin-bottom: 20px;
                margin-top: 5rem;
                color: #fff;
            }

            form {
                background-color: #fff;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                width: 100%;
                transition: transform 0.3s;
                margin-top:2.5rem;
            }

            form:hover {
                transform: translateY(-5px);
            }

            input[type="file"] {
                padding: 10px;
                margin-bottom: 25px;
                background-color: #f1f1f1;
                border: none;
                border-radius: 8px;
                width: 100%;
                cursor: pointer;
                transition: background-color 0.3s;
            }

            input[type="file"]:hover {
                background-color: #e1e1e1;
            }

            input[type="submit"] {
                background-color: #6a11cb;
                background-image: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                color: white;
                border: none;
                padding: 12px 25px;
                font-size: 1rem;
                border-radius: 50px;
                cursor: pointer;
                transition: background-color 0.3s, box-shadow 0.3s;
            }

            input[type="submit"]:hover {
                background-color: #4f7ac3;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }

            @media (max-width: 600px) {
                body {
                    padding: 20px;
                }

                form {
                    width: 100%;
                    padding: 20px;
                }

                h1 {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        <h1>Upload an Image for Classification</h1>
        <form action="/predict/" enctype="multipart/form-data" method="post">
            <input name="file" type="file" accept="image/*">
            <input type="submit" value="Upload and Classify">
        </form>
    </body>
</html>

    """
   return HTMLResponse(content=content)

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
