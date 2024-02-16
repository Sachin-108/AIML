# views.py
from django.shortcuts import render

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import subprocess
from django.http import JsonResponse
import os
from django.conf import settings

class_labels = ['Angeography Abnormal', 'Angeography Normal','CT Abnormal','CT Normal','ECG Abnormal','ECG Normal','MRI Abnormal','MRI Normal']
model_path = 'testimage/models/trained_model.h5'
model = load_model(model_path)

def predict_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        preprocessed_image = preprocess_image(uploaded_file)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        return render(request, 'result.html', {'predicted_label': predicted_label})

    return render(request, 'upload.html')

def preprocess_image(uploaded_file):
    img_content = uploaded_file.read()
    img = image.load_img(io.BytesIO(img_content), target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array



def run_script(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image = request.FILES['image']
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploads', 'temp_image.jpg').replace("\\", "/")
            with open(image_path, 'wb') as img_file:
                for chunk in image.chunks():
                    img_file.write(chunk)

            try:
                result = subprocess.check_output(["python", "testimage/models/detect.py", "--source", image_path], universal_newlines=True)
            except subprocess.CalledProcessError as e:
                return JsonResponse({'error': f'Error: {e.output}'}, status=500)

            # Extract the image name from the result (if available)
            image_name = 'test123.jpg'
            for line in result.split('\n'):
                if line.startswith("Image Path:"):
                    image_name = os.path.basename(line[len("Image Path:"):].strip())
                    break

            if image_name:
                # Construct the URL to serve the generated image
                image_url = '/media/uploads/processed/test123.jpg'

                # Return the image URL in JSON format
                return JsonResponse({'image_url': image_url})
            else:
                return JsonResponse({'error': 'Image path not found in the output of detect.py.'}, status=500)
        else:
            return JsonResponse({'error': 'Invalid image upload.'}, status=400)
    else:
        # Render the HTML page for image upload
        return render(request, 'script_result.html')
